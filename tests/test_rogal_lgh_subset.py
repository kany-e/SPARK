"""Rogal LGH cross-validation against site-type-aware LateralInteraction.

Verifies that SPARK's per-configuration lateral energy sum matches what
kmos Stage 1.4e bakes into each enumerated NN-config rate variant for
the Rogal 2008 LGH parameter set on PdO(sqrt5).

Site-type convention (test-local):
  site_type = 0 -> oxide bridge  (ox_br)
  site_type = 1 -> oxide hollow  (ox_hol)

V_ij values from reconkin parts/parameters.py L64-73:
  V_OO_brbr=0.08, V_OO_holhol=0.07, V_OO_brhol=0.08
  V_COCO_brbr=0.08, V_COCO_holhol=0.13, V_COCO_brhol=0.14
  V_OCO_brbr=0.06, V_OCO_holhol=0.11, V_OCO_brhol=0.13, V_OCO_holbr=0.12

LGH formula tested:
  E_lateral(target_site) = Σ over NN sites of V[target_sp, target_st,
                                                nn_sp, nn_st]

The factor of 2 from Rogal Eq 12 (applied at the rate-formula level in
kmos's CO_diff_brhol enumeration) is OUT OF SCOPE here. We test the bare
cluster-expansion sum, which is what SPARK's _compute_interaction_energy
returns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from spark.types import Project, Site, Condition, Action, LateralInteraction
from spark.engine import KMCEngine


# Rogal 2008 V_ij values (eV) — from reconkin parts/parameters.py
ROGAL_V = {
    ('O', 'O', 0, 0):     0.08,   # V_OO_brbr
    ('O', 'O', 1, 1):     0.07,   # V_OO_holhol
    ('O', 'O', 0, 1):     0.08,   # V_OO_brhol
    ('CO', 'CO', 0, 0):   0.08,   # V_COCO_brbr
    ('CO', 'CO', 1, 1):   0.13,   # V_COCO_holhol
    ('CO', 'CO', 0, 1):   0.14,   # V_COCO_brhol
    ('O', 'CO', 0, 0):    0.06,   # V_OCO_brbr
    ('O', 'CO', 1, 1):    0.11,   # V_OCO_holhol
    ('O', 'CO', 0, 1):    0.13,   # V_OCO_brhol — O@br, CO@hol
    ('O', 'CO', 1, 0):    0.12,   # V_OCO_holbr — O@hol, CO@br (asymmetric!)
}


BR = 0
HOL = 1


def _build_rogal_project():
    """Build a 2-layer-equivalent flat project with 2 site types per cell.
    Each cell: 1 ox_br (st=0) + 1 ox_hol (st=1). Species: empty, O, CO."""
    pt = Project()
    pt.set_meta(model_name='rogal_lgh_test', model_dimension=2)
    pt.add_species(name='empty')
    pt.add_species(name='O')
    pt.add_species(name='CO')

    layer = pt.add_layer(name='surf')
    layer.sites.append(Site(name='br',  default_species='empty', site_type=BR))
    layer.sites.append(Site(name='hol', default_species='empty', site_type=HOL))

    # Dummy processes anchored at br and at hol so we can invoke
    # _compute_interaction_energy via either entry site.
    c_br  = pt.lattice.generate_coord('br.(0,0,0).surf')
    c_hol = pt.lattice.generate_coord('hol.(0,0,0).surf')

    pt.add_process(
        name='probe_br',
        conditions=[Condition(c_br, 'empty')],
        actions=[Action(c_br, 'O')],
        rate_constant='1.0',
    )
    pt.add_process(
        name='probe_hol',
        conditions=[Condition(c_hol, 'empty')],
        actions=[Action(c_hol, 'O')],
        rate_constant='1.0',
    )

    # Register all 10 Rogal V_ij values
    for (sp1, sp2, st1, st2), V in ROGAL_V.items():
        pt.add_lateral_interaction(sp1, sp2, V,
                                   site_type1=st1, site_type2=st2)

    return pt


def _site_index(eng, cx, cy, site_in_cell):
    """Global site index for cell (cx, cy) and intra-cell index."""
    Lx = eng.lattice_size[0]
    cell_id = cx + Lx * cy
    return cell_id * eng.spuck + site_in_cell


def test_all_V_registered():
    """All 10 Rogal V_ij values are stored in _lateral_energy with
    the correct asymmetric handling for V_OCO_brhol vs V_OCO_holbr."""
    pt = _build_rogal_project()
    eng = KMCEngine(pt, size=[3, 3], banner=False, print_rates=False)

    spO  = eng.species_id['O']
    spCO = eng.species_id['CO']

    # Same-species, same site type (symmetric in site_type)
    assert eng._lateral_energy[spO, BR, spO, BR]   == 0.08
    assert eng._lateral_energy[spO, HOL, spO, HOL] == 0.07
    assert eng._lateral_energy[spCO, BR, spCO, BR]   == 0.08
    assert eng._lateral_energy[spCO, HOL, spCO, HOL] == 0.13

    # Same-species cross site type (symmetric: br-hol == hol-br for same species)
    assert eng._lateral_energy[spO, BR, spO, HOL]   == 0.08
    assert eng._lateral_energy[spO, HOL, spO, BR]   == 0.08
    assert eng._lateral_energy[spCO, BR, spCO, HOL] == 0.14
    assert eng._lateral_energy[spCO, HOL, spCO, BR] == 0.14

    # Cross-species, same site type (O-CO has its own br-br and hol-hol)
    assert eng._lateral_energy[spO,  BR,  spCO, BR]  == 0.06
    assert eng._lateral_energy[spO,  HOL, spCO, HOL] == 0.11

    # Cross-species, cross site type — ASYMMETRIC by Rogal convention
    assert eng._lateral_energy[spO,  BR,  spCO, HOL] == 0.13  # O@br, CO@hol
    assert eng._lateral_energy[spO,  HOL, spCO, BR]  == 0.12  # O@hol, CO@br

    # The symmetric swap of each ALSO populates these slots
    # (engine writes both (sp1,st1,sp2,st2) and (sp2,st2,sp1,st1) on register)
    assert eng._lateral_energy[spCO, HOL, spO,  BR]  == 0.13  # mirror of brhol
    assert eng._lateral_energy[spCO, BR,  spO,  HOL] == 0.12  # mirror of holbr


def test_lateral_sum_O_at_br_with_O_neighbors():
    """O at bridge surrounded by O atoms at varied site-types.
    Expected sum = Σ V_OO_<st_entry><st_nn> × n_neighbors_at_st_nn.
    """
    pt = _build_rogal_project()
    eng = KMCEngine(pt, size=[3, 3], banner=False, print_rates=False)

    spO = eng.species_id['O']
    BR_offset  = pt.lattice.site_in_cell_id('surf', 'br')
    HOL_offset = pt.lattice.site_in_cell_id('surf', 'hol')

    # Set up the lattice manually so we know the configuration:
    # Target: O at br(1,1)
    # Place neighbors: depends on whose NN list contains what.
    # We don't know the NN topology a priori — just place SOMETHING
    # and compute the expected sum from what _build_neighbor_list produced.

    eng.lattice[:] = eng._empty_species
    target_site = _site_index(eng, 1, 1, BR_offset)
    eng.lattice[target_site] = spO

    # Fill ALL br sites and ALL hol sites with O, then we know exactly
    # what species each NN of the target is.
    for cid in range(eng.ncells):
        eng.lattice[cid * eng.spuck + BR_offset]  = spO
        eng.lattice[cid * eng.spuck + HOL_offset] = spO
    # Target is part of "all br are O" so we don't need to set it separately

    # Count how many of target's NNs are br-typed and hol-typed
    n_nn_br  = sum(1 for nn in eng.neighbors[target_site]
                   if eng.site_types[nn] == BR)
    n_nn_hol = sum(1 for nn in eng.neighbors[target_site]
                   if eng.site_types[nn] == HOL)

    # Expected lateral sum:
    # All NN are O, target is O at br.
    # V[O@br, O@br] = 0.08 contributed per br neighbor
    # V[O@br, O@hol] = 0.08 contributed per hol neighbor
    expected = n_nn_br * 0.08 + n_nn_hol * 0.08

    # Use probe_br process (proc_id 0); is_product=True so the action
    # entry (which has sp_id=O at the br) gets used.
    E = eng._compute_interaction_energy(0, target_site, is_product=True)

    assert abs(E - expected) < 1e-12, \
        f"Expected {expected} (n_br_nn={n_nn_br}, n_hol_nn={n_nn_hol}), got {E}"


def test_lateral_sum_O_at_br_with_CO_neighbors():
    """O at bridge, all neighbors filled with CO.
    Expected: Σ V_OCO_<st_entry=br><st_nn> × n_nn_at_st_nn"""
    pt = _build_rogal_project()
    eng = KMCEngine(pt, size=[3, 3], banner=False, print_rates=False)

    spO  = eng.species_id['O']
    spCO = eng.species_id['CO']
    BR_offset  = pt.lattice.site_in_cell_id('surf', 'br')
    HOL_offset = pt.lattice.site_in_cell_id('surf', 'hol')

    # Target: O at br(1,1); fill all neighbors with CO
    eng.lattice[:] = eng._empty_species
    target_site = _site_index(eng, 1, 1, BR_offset)
    eng.lattice[target_site] = spO

    # Fill all sites OTHER than target with CO
    for cid in range(eng.ncells):
        s_br  = cid * eng.spuck + BR_offset
        s_hol = cid * eng.spuck + HOL_offset
        if s_br  != target_site: eng.lattice[s_br]  = spCO
        if s_hol != target_site: eng.lattice[s_hol] = spCO

    n_nn_br  = sum(1 for nn in eng.neighbors[target_site]
                   if eng.site_types[nn] == BR)
    n_nn_hol = sum(1 for nn in eng.neighbors[target_site]
                   if eng.site_types[nn] == HOL)

    # O@br + CO@br -> V_OCO_brbr = 0.06
    # O@br + CO@hol -> V_OCO_brhol = 0.13
    expected = n_nn_br * 0.06 + n_nn_hol * 0.13

    E = eng._compute_interaction_energy(0, target_site, is_product=True)
    assert abs(E - expected) < 1e-12, \
        f"Expected {expected} (n_br_nn={n_nn_br}, n_hol_nn={n_nn_hol}), got {E}"


def test_lateral_sum_O_at_hol_with_CO_neighbors():
    """O at hollow, all neighbors filled with CO.
    Critical for testing the OCO asymmetry: V_OCO_holbr=0.12 vs V_OCO_holhol=0.11."""
    pt = _build_rogal_project()
    eng = KMCEngine(pt, size=[3, 3], banner=False, print_rates=False)

    spO  = eng.species_id['O']
    spCO = eng.species_id['CO']
    BR_offset  = pt.lattice.site_in_cell_id('surf', 'br')
    HOL_offset = pt.lattice.site_in_cell_id('surf', 'hol')

    # Target: O at hol(1,1); fill all neighbors with CO
    eng.lattice[:] = eng._empty_species
    target_site = _site_index(eng, 1, 1, HOL_offset)
    eng.lattice[target_site] = spO

    for cid in range(eng.ncells):
        s_br  = cid * eng.spuck + BR_offset
        s_hol = cid * eng.spuck + HOL_offset
        if s_br  != target_site: eng.lattice[s_br]  = spCO
        if s_hol != target_site: eng.lattice[s_hol] = spCO

    n_nn_br  = sum(1 for nn in eng.neighbors[target_site]
                   if eng.site_types[nn] == BR)
    n_nn_hol = sum(1 for nn in eng.neighbors[target_site]
                   if eng.site_types[nn] == HOL)

    # O@hol + CO@br -> V_OCO_holbr = 0.12 (asymmetric!)
    # O@hol + CO@hol -> V_OCO_holhol = 0.11
    expected = n_nn_br * 0.12 + n_nn_hol * 0.11

    # Use probe_hol (proc_id 1) since target is at hol
    E = eng._compute_interaction_energy(1, target_site, is_product=True)
    assert abs(E - expected) < 1e-12, \
        f"Expected {expected} (n_br_nn={n_nn_br}, n_hol_nn={n_nn_hol}), got {E}"


def test_mixed_species_neighbors():
    """O at bridge with mixed O+CO+empty neighbors. Tests that empty
    species contribute 0 and that O and CO contributions are computed
    against the correct V terms."""
    pt = _build_rogal_project()
    eng = KMCEngine(pt, size=[3, 3], banner=False, print_rates=False)

    spO  = eng.species_id['O']
    spCO = eng.species_id['CO']
    BR_offset  = pt.lattice.site_in_cell_id('surf', 'br')
    HOL_offset = pt.lattice.site_in_cell_id('surf', 'hol')

    eng.lattice[:] = eng._empty_species
    target_site = _site_index(eng, 1, 1, BR_offset)
    eng.lattice[target_site] = spO

    # Alternate species at neighbor sites: half O, half CO, some empty
    nn_list = list(eng.neighbors[target_site])
    expected = 0.0
    for i, nn in enumerate(nn_list):
        st = eng.site_types[nn]
        if i % 3 == 0:
            eng.lattice[nn] = spO
            expected += ROGAL_V[('O', 'O', BR, int(st))]
        elif i % 3 == 1:
            eng.lattice[nn] = spCO
            expected += ROGAL_V[('O', 'CO', BR, int(st))]
        # i % 3 == 2: leave as empty, contributes 0

    E = eng._compute_interaction_energy(0, target_site, is_product=True)
    assert abs(E - expected) < 1e-12, \
        f"Expected {expected}, got {E}, nn_count={len(nn_list)}"


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'  PASS  {name}')
            except AssertionError as e:
                print(f'  FAIL  {name}: {e}')
                raise
    print()
    print('All Rogal LGH cross-validation tests: PASS')
