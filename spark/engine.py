"""
Core Kinetic Monte Carlo engine implementing the BKL rejection-free algorithm.

Features:
  - BKL/VSSM rejection-free algorithm
  - Neighbor list for spatial events
  - Pairwise lateral interactions with per-site rates
  - Surface diffusion support
  - BEP (Bronsted-Evans-Polanyi) relations
  - Site type support

Algorithm per step:
  1. Compute cumulative rates (per-site when lateral interactions exist)
  2. Draw random numbers r1, r2, r3 ~ U(0,1)
  3. Advance time: dt = -ln(r1) / R_total
  4. Select process by binary search on cumulative rates using r2
  5. Select site (uniform or rate-weighted) using r3
  6. Execute process (update lattice + bookkeeping + affected rates)
"""

import numpy as np
import time as _time
from collections import defaultdict
from .rates import evaluate_rate_expression
from .units import kB, eV


class ParameterProxy:
    """
    Allows attribute-style access to model parameters.

    Usage:
        model.parameters.T = 550
        print(model.parameters.T)
    """

    def __init__(self, param_list, engine=None):
        object.__setattr__(self, '_params', {p.name: p for p in param_list})
        object.__setattr__(self, '_engine', engine)

    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        params = object.__getattribute__(self, '_params')
        if name in params:
            return params[name].value
        raise AttributeError(f"No parameter '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        params = object.__getattribute__(self, '_params')
        if name in params:
            params[name].value = value
            engine = object.__getattribute__(self, '_engine')
            if engine is not None:
                engine._update_rate_constants()
        else:
            raise AttributeError(f"No parameter '{name}'")

    def __repr__(self):
        params = object.__getattribute__(self, '_params')
        lines = ["Parameters:"]
        for name, p in params.items():
            lines.append(f"  {name} = {p.value}")
        return '\n'.join(lines)

    def as_dict(self):
        params = object.__getattribute__(self, '_params')
        return {name: p.value for name, p in params.items()}


class KMCEngine:
    """
    Lattice Kinetic Monte Carlo simulation engine.

    Parameters
    ----------
    project : Project
        The model definition.
    size : list of int
        Lattice dimensions, e.g. [50, 50] for a 50x50 2D lattice.
    print_rates : bool
        Print rate constants on initialization.
    banner : bool
        Print welcome banner.
    """

    def __init__(self, project, size=None, print_rates=True, banner=True):
        self.project = project
        self.size = size or [20, 20]
        self.ndim = project.meta.get('model_dimension', 2)

        if banner:
            print(f"spark KMC Engine - Model: "
                  f"{project.meta.get('model_name', 'unnamed')}")
            print(f"  Lattice: {self.size}, Dimension: {self.ndim}")

        # Species encoding
        self.nspecies = len(project.species_list)
        self.species_names = [sp.name for sp in project.species_list]
        self.species_id = {sp.name: sp.id for sp in project.species_list}

        # Detect empty species
        self._empty_species = 0
        for sp in project.species_list:
            if sp.name.lower() in ('empty', 'vacant', '*'):
                self._empty_species = sp.id
                break

        # Parameters (with auto-update callback)
        self.parameters = ParameterProxy(project.parameter_list, engine=self)

        # Initialize lattice (includes site types)
        self._init_lattice()

        # Build neighbor list
        self._build_neighbor_list()

        # Setup lateral interactions
        self._setup_lateral_interactions()

        # Setup BEP relations
        self._setup_bep_relations()

        # Defaults needed by _rebuild_avail_sites (called inside
        # _init_processes); callbacks are resolved right after, once
        # process names exist, and per-site rates are rebuilt then.
        self._rate_callback_by_id = {}
        self._per_site_active = self._has_lateral

        # Initialize processes
        self._init_processes()

        # Runtime rate callbacks: {process_name: fn(engine, proc_id,
        # site) -> rate}. Consulted FIRST in _compute_site_rate; used
        # for rate laws inexpressible as the static lateral/BEP forms
        # (e.g. Rogal II.D clipped diffusion barriers).
        for pname, fn in (getattr(project, 'rate_callbacks', None)
                          or {}).items():
            if pname not in self.process_names:
                raise KeyError(f'rate_callbacks: unknown process '
                               f'{pname!r}')
            self._rate_callback_by_id[
                self.process_names.index(pname)] = fn
        # Per-site rates are needed when either static laterals or
        # runtime callbacks are present.
        self._per_site_active = (self._has_lateral
                                 or bool(self._rate_callback_by_id))

        # Compute max offset range for neighbor updates
        self._compute_max_offset()

        # Rate constants
        self.rates = np.zeros(self.nproc)
        self._update_rate_constants()

        # Per-site rate tracking (lateral interactions or callbacks)
        if self._per_site_active:
            self._proc_total_rates = np.zeros(self.nproc)
            self._rebuild_per_site_rates()

        # Cumulative rates array
        self.accum_rates = np.zeros(self.nproc)

        # Simulation state
        self.kmc_time = 0.0
        self.kmc_step = 0
        self.procstat = np.zeros(self.nproc, dtype=np.int64)
        self._prev_procstat = np.zeros(self.nproc, dtype=np.int64)
        self._prev_time = 0.0

        # Wall time
        self._start_walltime = _time.time()

        if banner and self._has_lateral:
            n_lat = len(project.lateral_interactions)
            n_bep = len(self._bep_by_name)
            print(f"  Lateral interactions: {n_lat}, BEP relations: {n_bep}")
            print(f"  Neighbors per site: {len(self.neighbors[0])}")

        if print_rates:
            self.print_rates()

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def _init_lattice(self):
        """Initialize the lattice array with default species and site types.

        Multi-lattice (Hoffmann-Reuter-Scheffler 2015) support:

        - ``ncells = prod(lattice_size)`` is the number of physical unit cells.
        - ``spuck`` (Sites Per Unit Cell) is the total number of sites within
          one cell, summed across all layers. Layer membership is encoded in
          the site-within-cell index via cumulative offsets.
        - The global flat site index ``nr ∈ [0, ncells*spuck)`` decomposes as
          ``nr = cell_id * spuck + s_in_cell``.
        - Per-site defaults (default_species, site_type) are read from each
          ``Site`` object and tiled across cells, so different sites within
          the same cell can have different defaults.

        Single-layer 1-site models (the legacy use case) reduce to ``spuck=1``,
        which gives ``nsites = ncells`` and matches the pre-multi-lattice
        engine behavior byte-for-byte.
        """
        self.lattice_size = tuple(self.size[:self.ndim])
        self.ncells = int(np.prod(self.lattice_size))

        # spuck: sites per unit cell across all layers (>=1)
        self.spuck = max(self.project.lattice.spuck, 1)
        self.nsites = self.ncells * self.spuck

        # Per-site-in-cell defaults
        self._default_species_per_s = np.zeros(self.spuck, dtype=np.int32)
        self._site_type_per_s = np.zeros(self.spuck, dtype=np.int32)
        self._site_layer_id_per_s = np.zeros(self.spuck, dtype=np.int32)

        s_in_cell = 0
        for layer_id, layer in enumerate(self.project.lattice.layers):
            for site in layer.sites:
                ds_name = getattr(site, 'default_species', 'empty')
                if ds_name in self.species_id:
                    self._default_species_per_s[s_in_cell] = self.species_id[ds_name]
                stype = getattr(site, 'site_type', None)
                if isinstance(stype, int):
                    self._site_type_per_s[s_in_cell] = stype
                self._site_layer_id_per_s[s_in_cell] = layer_id
                s_in_cell += 1

        # Tile per-site-in-cell defaults across all cells -> global arrays
        if self.spuck >= 1:
            self.lattice = np.tile(self._default_species_per_s, self.ncells)
            self.site_types = np.tile(self._site_type_per_s, self.ncells)
        else:
            # Defensive: project has no layers/sites at all
            self.lattice = np.zeros(self.nsites, dtype=np.int32)
            self.site_types = np.zeros(self.nsites, dtype=np.int32)

    def _build_neighbor_list(self):
        """Build nearest-neighbor list for the lattice.

        Multi-lattice convention: "neighbor of site s" = the same
        ``s_in_cell`` at the nearest-neighbor cells. This matches paper §2.3
        of Hoffmann-Reuter 2015 — pairwise lateral interactions are
        within-layer (same site type) at NN cells. For ``spuck=1`` (legacy
        single-layer), this is identical to the previous cell-only NN list.
        """
        self.neighbors = [[] for _ in range(self.nsites)]
        nn_offsets = self._get_nn_offsets()

        for s in range(self.nsites):
            s_in_cell = self._site_in_cell(s)
            coord = self._site_to_coord(s)
            for offset in nn_offsets:
                nc = tuple(c + o for c, o in zip(coord, offset))
                ns = self._coord_to_site(nc, s_in_cell)
                if ns != s:
                    self.neighbors[s].append(ns)

    def _get_nn_offsets(self):
        """Get nearest-neighbor offsets for the lattice geometry."""
        if self.ndim == 1:
            return [(-1,), (1,)]
        elif self.ndim == 2:
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            return [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                    (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    def _setup_lateral_interactions(self):
        """Build lateral interaction lookup (4D array: sp1, st1, sp2, st2).

        site_type IDs come from Site.site_type at project setup; users
        allocate them as int32. When LateralInteraction.site_type1 or
        site_type2 is None, the interaction broadcasts over all site
        types for that species (backward compat with the prior 2D API).
        """
        # Determine site-type table size from project setup
        n_site_types = int(self._site_type_per_s.max()) + 1 \
            if self.spuck > 0 else 1
        self._n_site_types = n_site_types

        # 4D array: lateral_energy[sp1, st1, sp2, st2] = energy (0 = none)
        self._lateral_energy = np.zeros(
            (self.nspecies, n_site_types,
             self.nspecies, n_site_types))

        self._lateral_dict = {}
        if hasattr(self.project, 'lateral_interactions'):
            for li in self.project.lateral_interactions:
                sp1 = self.species_id.get(li.species1)
                sp2 = self.species_id.get(li.species2)
                if sp1 is None or sp2 is None:
                    continue
                st1 = li.site_type1
                st2 = li.site_type2

                # Resolve None to "all site types" (backward compat)
                st1_list = range(n_site_types) if st1 is None else [st1]
                st2_list = range(n_site_types) if st2 is None else [st2]

                for s1 in st1_list:
                    for s2 in st2_list:
                        self._lateral_energy[sp1, s1, sp2, s2] = li.energy
                        self._lateral_energy[sp2, s2, sp1, s1] = li.energy

                # Stable key per registered interaction (no double-count)
                key = (li.species1, st1, li.species2, st2)
                self._lateral_dict[key] = li.energy

        self._has_lateral = bool(np.any(self._lateral_energy != 0))

    def _setup_bep_relations(self):
        """Build BEP relation lookup from project definition."""
        self._bep_by_name = {}
        if hasattr(self.project, 'bep_relations'):
            self._bep_by_name = dict(self.project.bep_relations)

    def _init_processes(self):
        """Convert process definitions to numeric arrays for fast lookup.

        Each condition/action is stored as a 3-tuple
        ``(cell_offset, s_in_cell, sp_id)`` where ``s_in_cell`` is the
        layer-folded site-within-cell index resolved from ``Coord.layer`` and
        ``Coord.site``. For single-layer 1-site models, all coords resolve
        to ``s_in_cell = 0`` and behavior matches the pre-multi-lattice
        engine byte-for-byte.
        """
        self.nproc = len(self.project.process_list)
        self.process_names = [p.name for p in self.project.process_list]

        self._proc_conditions = []
        self._proc_actions = []
        self._proc_rate_exprs = []
        self._proc_tof_count = []
        self._proc_site_types = []

        lat = self.project.lattice
        for proc in self.project.process_list:
            conds = []
            for c in proc.conditions:
                offset = c.coord.offset[:self.ndim]
                sp_id = self.species_id[c.species]
                s_in_cell = self._resolve_s_in_cell(c.coord)
                conds.append((offset, s_in_cell, sp_id))
            self._proc_conditions.append(conds)

            acts = []
            for a in proc.actions:
                offset = a.coord.offset[:self.ndim]
                sp_id = self.species_id[a.species]
                s_in_cell = self._resolve_s_in_cell(a.coord)
                acts.append((offset, s_in_cell, sp_id))
            self._proc_actions.append(acts)

            self._proc_rate_exprs.append(proc.rate_constant)
            self._proc_tof_count.append(proc.tof_count)
            self._proc_site_types.append(getattr(proc, 'site_type', None))

        # Resolve BEP process IDs
        self._bep_proc_ids = {}
        for pid, name in enumerate(self.process_names):
            if name in self._bep_by_name:
                self._bep_proc_ids[pid] = self._bep_by_name[name]

        # Anchor site of each process: the s_in_cell of its first condition
        # (0 when there are no conditions). A process registers at exactly
        # one anchor site per satisfiable cell; without this constraint it
        # would register at all ``spuck`` sites of the cell, inflating the
        # total rate — and compressing kmc_time — by exactly ``spuck``.
        # For spuck=1 every anchor is 0 and behavior is unchanged.
        self._proc_anchor = [conds[0][1] if conds else 0
                             for conds in self._proc_conditions]
        self._procs_by_anchor = [[] for _ in range(self.spuck)]
        for pid, a in enumerate(self._proc_anchor):
            self._procs_by_anchor[a].append(pid)

        # Available sites bookkeeping: O(1) add/remove via swap-with-last
        self._avail_sites = [[] for _ in range(self.nproc)]
        self._site_in_avail = [dict() for _ in range(self.nproc)]
        # Parallel rate array for per-site rates (only used with lateral)
        self._avail_rates = [[] for _ in range(self.nproc)]

        self._rebuild_avail_sites()

    def _resolve_s_in_cell(self, coord):
        """Resolve a Coord's (layer, site) pair to a site-within-cell index.

        Returns 0 when layer/site are unspecified or when only one layer with
        one site exists (legacy single-layer models). Falls back gracefully
        for partial coords to preserve back-compat with legacy callers that
        didn't supply layer info explicitly.
        """
        lat = self.project.lattice
        if not lat.layers:
            return 0
        layer_name = getattr(coord, 'layer', None)
        site_name = getattr(coord, 'site', None)
        if layer_name is None and site_name is None:
            return 0
        if layer_name is None:
            layer_name = lat.layers[0].name
        try:
            return lat.site_in_cell_id(layer_name, site_name) if site_name \
                else lat.layer_offset(layer_name)
        except KeyError:
            return 0

    def _compute_max_offset(self):
        """Find the maximum cell-offset range across all process conditions."""
        max_r = 1
        for conds in self._proc_conditions:
            for offset, _s_in_cell, _sp in conds:
                for o in offset:
                    max_r = max(max_r, abs(o) + 1)
        for acts in self._proc_actions:
            for offset, _s_in_cell, _sp in acts:
                for o in offset:
                    max_r = max(max_r, abs(o) + 1)
        self._max_offset = max_r

    # ----------------------------------------------------------------
    # Coordinate conversion (with periodic boundary conditions)
    # ----------------------------------------------------------------

    def _site_to_coord(self, site):
        """Map global flat site index -> cell coord (cx, cy[, cz]).

        For multi-lattice, each cell contains ``spuck`` sites; this returns
        only the cell coord. Use ``self._site_in_cell(site)`` to recover the
        site-within-cell index. For single-layer 1-site models (spuck=1),
        this is a no-op identity.
        """
        cell_id = site // self.spuck
        if self.ndim == 1:
            return (cell_id,)
        elif self.ndim == 2:
            return (cell_id // self.lattice_size[1],
                    cell_id % self.lattice_size[1])
        else:
            Ly = self.lattice_size[1]
            Lz = self.lattice_size[2]
            return (cell_id // (Ly * Lz),
                    (cell_id % (Ly * Lz)) // Lz,
                    cell_id % Lz)

    def _coord_to_site(self, coord, s_in_cell=0):
        """Map (cell coord, site-within-cell) -> global flat site index.

        ``s_in_cell`` defaults to 0 for back-compat: legacy callers passing
        only a cell coord get the first site of that cell, which for
        single-layer 1-site models is the only site.
        """
        if self.ndim == 1:
            cell_id = coord[0] % self.lattice_size[0]
        elif self.ndim == 2:
            x = coord[0] % self.lattice_size[0]
            y = coord[1] % self.lattice_size[1]
            cell_id = x * self.lattice_size[1] + y
        else:
            x = coord[0] % self.lattice_size[0]
            y = coord[1] % self.lattice_size[1]
            z = coord[2] % self.lattice_size[2]
            cell_id = (x * self.lattice_size[1] * self.lattice_size[2]
                       + y * self.lattice_size[2] + z)
        return cell_id * self.spuck + s_in_cell

    def _site_in_cell(self, site):
        """Site-within-cell index (= layer-folded index) for a global site."""
        return site % self.spuck

    # ----------------------------------------------------------------
    # Available sites bookkeeping
    # ----------------------------------------------------------------

    def _check_process_at_site(self, proc_id, site):
        """Check if process can occur at site (anchor + species + site type)."""
        # The process is anchored at a single s_in_cell per cell
        if self._proc_anchor[proc_id] != site % self.spuck:
            return False

        # Check site type requirement
        site_type_req = self._proc_site_types[proc_id]
        if site_type_req is not None:
            if self.site_types[site] != site_type_req:
                return False

        # Check species conditions
        coord = self._site_to_coord(site)
        for offset, s_in_cell, sp_id in self._proc_conditions[proc_id]:
            neighbor = tuple(c + o for c, o in zip(coord, offset))
            if self.lattice[self._coord_to_site(neighbor, s_in_cell)] != sp_id:
                return False
        return True

    def _add_to_avail(self, proc_id, site):
        if site not in self._site_in_avail[proc_id]:
            idx = len(self._avail_sites[proc_id])
            self._site_in_avail[proc_id][site] = idx
            self._avail_sites[proc_id].append(site)
            if self._per_site_active and hasattr(self,
                                                 '_proc_total_rates'):
                rate = self._compute_site_rate(proc_id, site)
                self._avail_rates[proc_id].append(rate)
                self._proc_total_rates[proc_id] += rate
            else:
                self._avail_rates[proc_id].append(0.0)

    def _remove_from_avail(self, proc_id, site):
        idx_map = self._site_in_avail[proc_id]
        if site in idx_map:
            idx = idx_map[site]
            avail = self._avail_sites[proc_id]
            rates = self._avail_rates[proc_id]

            if self._per_site_active and hasattr(self,
                                                 '_proc_total_rates'):
                self._proc_total_rates[proc_id] -= rates[idx]

            # Swap with last
            last_site = avail[-1]
            avail[idx] = last_site
            rates[idx] = rates[-1]
            idx_map[last_site] = idx
            avail.pop()
            rates.pop()
            del idx_map[site]

    def _rebuild_avail_sites(self):
        """Full rebuild of available sites (used at initialization)."""
        for p in range(self.nproc):
            self._avail_sites[p] = []
            self._site_in_avail[p] = {}
            self._avail_rates[p] = []
        # Only the anchor site of each cell can host the process
        for p in range(self.nproc):
            anchor = self._proc_anchor[p]
            for cell in range(self.ncells):
                s = cell * self.spuck + anchor
                if self._check_process_at_site(p, s):
                    self._add_to_avail(p, s)

    def _rebuild_per_site_rates(self):
        """Recompute all per-site rates from scratch."""
        for p in range(self.nproc):
            total = 0.0
            for i, site in enumerate(self._avail_sites[p]):
                rate = self._compute_site_rate(p, site)
                self._avail_rates[p][i] = rate
                total += rate
            self._proc_total_rates[p] = total

    def _get_affected_sites(self, site, actions):
        """
        Get all sites whose process availability might change
        after executing actions rooted at site.
        Uses neighbor-list fast path for single-site processes.
        """
        # Fast path: single-site process with max_offset <= 1
        # Use neighbor list directly (5 sites in 2D vs 9 from grid).
        # spuck=1 only: the neighbor list is same-s_in_cell, so for spuck>1
        # it would miss anchor sites at other s_in_cell in affected cells.
        if len(actions) == 1 and self._max_offset <= 1 and self.spuck == 1:
            offset, s_in_cell, _sp = actions[0]
            if all(o == 0 for o in offset) and s_in_cell == self._site_in_cell(site):
                action_site = site
            else:
                coord = self._site_to_coord(site)
                nc = tuple(c + o for c, o in zip(coord, offset))
                action_site = self._coord_to_site(nc, s_in_cell)
            affected = set(self.neighbors[action_site])
            affected.add(action_site)
            return affected

        # General path: coordinate grid
        coord = self._site_to_coord(site)
        affected = set()
        # NOTE on per-site (lateral/callback) rate invalidation: a
        # rate at anchor A depends on sites up to max_cond_offset + 1
        # cells away (NN of a condition site). _compute_max_offset
        # already returns max|offset| + 1 >= max_cond_offset + 1, so
        # this radius refreshes every NN-dependent rate; proven
        # deterministically in stage21/test_runtime_engine.py.
        r = self._max_offset
        spuck = self.spuck

        changed_coords = []
        for offset, _s_in_cell, _sp in actions:
            nc = tuple(c + o for c, o in zip(coord, offset))
            changed_coords.append(nc)

        # For each affected cell, mark all spuck sites as potentially affected
        # (conservative; for spuck=1 this is identical to legacy single-site).
        if self.ndim == 1:
            for cc in changed_coords:
                for dx in range(-r, r + 1):
                    for k in range(spuck):
                        affected.add(self._coord_to_site((cc[0] + dx,), k))
        elif self.ndim == 2:
            for cc in changed_coords:
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        for k in range(spuck):
                            affected.add(self._coord_to_site(
                                (cc[0] + dx, cc[1] + dy), k))
        else:
            for cc in changed_coords:
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        for dz in range(-r, r + 1):
                            for k in range(spuck):
                                affected.add(self._coord_to_site(
                                    (cc[0] + dx, cc[1] + dy, cc[2] + dz), k))

        return affected

    def _update_avail_after_execution(self, affected_sites):
        """Update available sites and per-site rates for all affected sites."""
        for site in affected_sites:
            # Only processes anchored at this site's s_in_cell can live here
            for p in self._procs_by_anchor[site % self.spuck]:
                was_avail = site in self._site_in_avail[p]
                is_avail = self._check_process_at_site(p, site)

                if is_avail and not was_avail:
                    self._add_to_avail(p, site)
                elif not is_avail and was_avail:
                    self._remove_from_avail(p, site)
                elif is_avail and was_avail and self._per_site_active:
                    idx = self._site_in_avail[p][site]
                    old_rate = self._avail_rates[p][idx]
                    new_rate = self._compute_site_rate(p, site)
                    self._avail_rates[p][idx] = new_rate
                    self._proc_total_rates[p] += (new_rate - old_rate)

    # ----------------------------------------------------------------
    # Lateral interactions & BEP
    # ----------------------------------------------------------------

    def _get_temperature(self):
        """Get current temperature from parameters."""
        try:
            return float(self.parameters.T)
        except (AttributeError, TypeError):
            return 300.0

    def _compute_site_rate(self, proc_id, site):
        """
        Compute rate for a specific (process, site) pair including
        lateral interactions and BEP corrections.

        Without lateral interactions, returns the base rate.
        With lateral interactions:
          - Default: k = k_base * exp(+E_lat_react / (kB*T))
            where E_lat_react is the sum of pairwise interactions
            for adsorbates in the reactant state.
          - With BEP: k = k_base * exp(-alpha * delta_delta_H / (kB*T))
            where delta_delta_H = E_lat_product - E_lat_reactant.
        """
        cb = self._rate_callback_by_id.get(proc_id)
        if cb is not None:
            return cb(self, proc_id, site)

        base_rate = self.rates[proc_id]

        if not self._has_lateral:
            return base_rate

        T = self._get_temperature()
        if T <= 0:
            return base_rate

        beta_th = eV / (kB * T)

        # Compute interaction energy for reactant state
        E_lat_react = self._compute_interaction_energy(proc_id, site,
                                                       is_product=False)

        # BEP correction
        if proc_id in self._bep_proc_ids:
            bep = self._bep_proc_ids[proc_id]
            E_lat_prod = self._compute_interaction_energy(proc_id, site,
                                                          is_product=True)
            delta_delta_H = E_lat_prod - E_lat_react
            delta_Ea = bep.alpha * delta_delta_H
            return base_rate * np.exp(-delta_Ea * beta_th)

        # Default: reactant-state interaction modifies barrier
        # Positive E_lat (repulsion) → destabilized adsorbate → higher rate
        return base_rate * np.exp(E_lat_react * beta_th)

    def _compute_interaction_energy(self, proc_id, site, is_product=False):
        """
        Compute total pairwise lateral interaction energy for a process.

        Sums epsilon(species_i, species_neighbor) for all condition/action
        sites and their nearest neighbors (excluding intra-process sites).
        """
        coord = self._site_to_coord(site)
        entries = (self._proc_actions[proc_id] if is_product
                   else self._proc_conditions[proc_id])

        # Collect sites involved in this process (to exclude from counting)
        proc_sites = set()
        for offset, s_in_cell, _sp in entries:
            ps = self._coord_to_site(
                tuple(c + o for c, o in zip(coord, offset)), s_in_cell)
            proc_sites.add(ps)

        E_total = 0.0
        n_entries = len(entries)
        for offset, s_in_cell, sp_id in entries:
            if sp_id == self._empty_species:
                continue

            entry_site = self._coord_to_site(
                tuple(c + o for c, o in zip(coord, offset)), s_in_cell)
            st_entry = self.site_types[entry_site]

            for nn in self.neighbors[entry_site]:
                if n_entries > 1 and nn in proc_sites:
                    continue
                nn_sp = self.lattice[nn]
                st_nn = self.site_types[nn]
                # 4D lookup: (firing species, firing site_type,
                #             neighbor species, neighbor site_type)
                E_total += self._lateral_energy[sp_id, st_entry,
                                                 nn_sp, st_nn]

        return E_total

    # ----------------------------------------------------------------
    # Rate constants
    # ----------------------------------------------------------------

    def _update_rate_constants(self):
        """Evaluate all rate expressions with current parameters."""
        params = {p.name: p.value for p in self.project.parameter_list}
        for i, expr in enumerate(self._proc_rate_exprs):
            self.rates[i] = evaluate_rate_expression(expr, params)
        # Recompute per-site rates if per-site machinery is active
        if self._per_site_active and hasattr(self, '_proc_total_rates'):
            self._rebuild_per_site_rates()

    def _update_accum_rates(self):
        """Build cumulative rate array for process selection."""
        total = 0.0
        for p in range(self.nproc):
            if self._per_site_active:
                total += self._proc_total_rates[p]
            else:
                total += self.rates[p] * len(self._avail_sites[p])
            self.accum_rates[p] = total
        return total

    # ----------------------------------------------------------------
    # KMC step (BKL algorithm)
    # ----------------------------------------------------------------

    def do_kmc_step(self):
        """
        Execute one KMC step using the BKL rejection-free algorithm.

        Returns True if a step was executed, False if the system is frozen.
        """
        total_rate = self._update_accum_rates()

        if total_rate <= 0.0:
            return False

        # Three independent random numbers
        r_time = np.random.random()
        r_proc = np.random.random()
        r_site = np.random.random()

        # Time advancement (Poisson process)
        self.kmc_time += -np.log(r_time) / total_rate

        # Process selection (binary search on cumulative rates)
        proc_id = int(np.searchsorted(self.accum_rates,
                                       r_proc * total_rate))
        if proc_id >= self.nproc:
            proc_id = self.nproc - 1

        # Site selection
        n_avail = len(self._avail_sites[proc_id])
        if n_avail == 0:
            return False

        if self._per_site_active:
            # Select site proportional to per-site rate using cumulative sum
            rates_list = self._avail_rates[proc_id]
            total_proc = self._proc_total_rates[proc_id]
            if total_proc <= 0:
                return False
            target = r_site * total_proc
            cumul = 0.0
            site_idx = n_avail - 1
            for k in range(n_avail):
                cumul += rates_list[k]
                if cumul >= target:
                    site_idx = k
                    break
        else:
            # Uniform selection among available sites
            site_idx = min(int(r_site * n_avail), n_avail - 1)

        site = self._avail_sites[proc_id][site_idx]

        # Execute: update lattice
        coord = self._site_to_coord(site)
        for offset, s_in_cell, new_sp in self._proc_actions[proc_id]:
            neighbor = tuple(c + o for c, o in zip(coord, offset))
            self.lattice[self._coord_to_site(neighbor, s_in_cell)] = new_sp

        # Update bookkeeping
        affected = self._get_affected_sites(
            site, self._proc_actions[proc_id])
        self._update_avail_after_execution(affected)

        # Statistics
        self.kmc_step += 1
        self.procstat[proc_id] += 1

        return True

    def do_steps(self, n, progress=False):
        """
        Execute n KMC steps.

        Parameters
        ----------
        n : int
            Number of steps to execute.
        progress : bool
            Print progress at 10% intervals.
        """
        n = int(n)
        report_interval = max(1, n // 10)
        for i in range(n):
            if not self.do_kmc_step():
                print(f"System frozen at step {self.kmc_step}, "
                      f"time={self.kmc_time:.6e} s")
                break
            if progress and (i + 1) % report_interval == 0:
                pct = 100 * (i + 1) / n
                print(f"  [{pct:5.1f}%] step={self.kmc_step}, "
                      f"time={self.kmc_time:.6e} s")

    # ----------------------------------------------------------------
    # Observables
    # ----------------------------------------------------------------

    def get_coverage(self):
        """Get fractional coverage for each species."""
        coverage = {}
        for sp in self.project.species_list:
            coverage[sp.name] = float(np.sum(self.lattice == sp.id)) / \
                self.nsites
        return coverage

    def get_occupation(self):
        """Get occupation matrix [nspecies, nsites]."""
        occ = np.zeros((self.nspecies, self.nsites))
        for s in range(self.nsites):
            occ[self.lattice[s], s] = 1.0
        return occ

    def get_tof(self):
        """
        Get turn-over frequencies since last call.

        Returns dict of {observable_name: TOF in s^-1 per site}.
        """
        dt = self.kmc_time - self._prev_time
        if dt <= 0:
            return {}

        tof = defaultdict(float)
        for p in range(self.nproc):
            delta = self.procstat[p] - self._prev_procstat[p]
            for obs, coeff in self._proc_tof_count[p].items():
                tof[obs] += coeff * delta / (dt * self.nsites)

        self._prev_procstat = self.procstat.copy()
        self._prev_time = self.kmc_time
        return dict(tof)

    def get_process_stats(self):
        """Get execution count per process."""
        return {self.process_names[i]: int(self.procstat[i])
                for i in range(self.nproc)}

    def get_neighbor_coverages(self, site):
        """
        Get species counts among nearest neighbors of a site.

        Returns dict of {species_name: count}.
        """
        counts = defaultdict(int)
        for nn in self.neighbors[site]:
            sp = self.species_names[self.lattice[nn]]
            counts[sp] += 1
        return dict(counts)

    def get_lattice_2d(self):
        """
        Return 2D array representation of the lattice (for visualization).
        Only works for 2D lattices.
        """
        if self.ndim != 2:
            raise ValueError("get_lattice_2d only works for 2D lattices")
        return self.lattice.reshape(self.lattice_size)

    # ----------------------------------------------------------------
    # Site manipulation
    # ----------------------------------------------------------------

    def put(self, site_coord, species_name):
        """Set species at a specific lattice coordinate."""
        site = self._coord_to_site(tuple(site_coord[:self.ndim]))
        sp_id = self.species_id[species_name]
        self.lattice[site] = sp_id
        affected = self._get_affected_sites(
            site, [((0,) * self.ndim, sp_id)])
        self._update_avail_after_execution(affected)

    def get(self, site_coord):
        """Get species name at a specific lattice coordinate."""
        site = self._coord_to_site(tuple(site_coord[:self.ndim]))
        return self.species_names[self.lattice[site]]

    def set_site_type(self, site_coord, site_type):
        """Set the site type for a specific lattice site."""
        site = self._coord_to_site(tuple(site_coord[:self.ndim]))
        self.site_types[site] = site_type

    def set_site_types_region(self, region_func, site_type):
        """
        Set site type for all sites where region_func(coord) returns True.

        Parameters
        ----------
        region_func : callable
            Function taking a coordinate tuple and returning bool.
        site_type : int
            Site type to assign.
        """
        for s in range(self.nsites):
            coord = self._site_to_coord(s)
            if region_func(coord):
                self.site_types[s] = site_type
        self._rebuild_avail_sites()
        if self._has_lateral and hasattr(self, '_proc_total_rates'):
            self._rebuild_per_site_rates()

    # ----------------------------------------------------------------
    # State management
    # ----------------------------------------------------------------

    def reset(self):
        """Reset simulation to initial state (all sites = default species)."""
        default_sp = 0
        if self.project.lattice.layers:
            layer = self.project.lattice.layers[0]
            if layer.sites:
                ds_name = layer.sites[0].default_species
                if ds_name in self.species_id:
                    default_sp = self.species_id[ds_name]

        self.lattice[:] = default_sp
        self.kmc_time = 0.0
        self.kmc_step = 0
        self.procstat[:] = 0
        self._prev_procstat[:] = 0
        self._prev_time = 0.0
        self._rebuild_avail_sites()
        if self._has_lateral and hasattr(self, '_proc_total_rates'):
            self._rebuild_per_site_rates()

    def get_configuration(self):
        """Return a copy of the current lattice configuration."""
        return self.lattice.copy()

    def set_configuration(self, config):
        """Set lattice configuration and rebuild bookkeeping."""
        self.lattice[:] = config
        self._rebuild_avail_sites()
        if self._has_lateral and hasattr(self, '_proc_total_rates'):
            self._rebuild_per_site_rates()

    # ----------------------------------------------------------------
    # Printing
    # ----------------------------------------------------------------

    def print_rates(self):
        """Print all rate constants and available site counts."""
        print("\nRate constants:")
        print(f"  {'Process':<35s} {'k [s^-1]':>12s}  {'N_avail':>8s}")
        print(f"  {'-'*35} {'-'*12}  {'-'*8}")
        for i in range(self.nproc):
            n_avail = len(self._avail_sites[i])
            print(f"  {self.process_names[i]:<35s} "
                  f"{self.rates[i]:>12.4e}  {n_avail:>8d}")
        print()

    def print_coverages(self):
        """Print current surface coverages."""
        cov = self.get_coverage()
        print("Coverages:")
        for name, val in cov.items():
            if val > 1e-6:
                print(f"  {name}: {val:.6f}")

    # ----------------------------------------------------------------
    # Context manager
    # ----------------------------------------------------------------

    def deallocate(self):
        """Clean up resources (for kmos API compatibility)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.deallocate()

    def __repr__(self):
        lat_str = ''
        if self._has_lateral:
            lat_str = f', lateral={len(self._lateral_dict)}'
        return (f"KMCEngine(model='{self.project.meta.get('model_name')}', "
                f"size={list(self.lattice_size)}, "
                f"step={self.kmc_step}, time={self.kmc_time:.6e}"
                f"{lat_str})")
