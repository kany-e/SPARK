"""Anchor-multiplicity regression tests (kmc_time semantics).

A process must register at exactly one anchor site per satisfiable cell
(the s_in_cell of its first condition), never at all ``spuck`` sites of
the cell. Otherwise the total rate is inflated by ``spuck`` and kmc_time
is compressed by the same factor.

Checks:
  1. Registered event count == ncells for an always-satisfiable process
     in a spuck=19 model (was ncells*spuck before the anchor fix).
  2. Toy 2-site-cell model, one always-available process at rate k:
     mean kmc_time step == 1/(ncells*k) within stochastic error
     (was 1/(2*ncells*k) before the fix).
  3. spuck=1 legacy model keeps the same time axis (guard against the
     fix disturbing single-site models).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from spark.types import Project, Site, Condition, Action
from spark.engine import KMCEngine


def _noop_process_project(nsites_per_cell, k=2.0):
    """Project with ``nsites_per_cell`` sites and one always-available
    process at rate k anchored at site 0 (action rewrites 'empty', so
    availability never changes and R_total stays ncells*k exactly)."""
    pt = Project()
    pt.set_meta(model_name='anchor_time_toy', model_dimension=2)
    pt.add_species(name='empty')
    pt.add_species(name='CO')
    pt.add_parameter(name='T', value=300.0)
    layer = pt.add_layer(name='L')
    for i in range(nsites_per_cell):
        layer.sites.append(Site(name=f's{i}', default_species='empty'))
    coord = pt.lattice.generate_coord('s0.(0,0,0).L')
    pt.add_process(name='tick',
                   conditions=[Condition(coord, 'empty')],
                   actions=[Action(coord, 'empty')],
                   rate_constant=str(k))
    return pt


def test_event_count_one_per_cell_spuck19():
    pt = _noop_process_project(19)
    eng = KMCEngine(pt, size=[3, 3], print_rates=False, banner=False)
    n_events = sum(len(eng._avail_sites[p]) for p in range(eng.nproc))
    assert n_events == eng.ncells, (
        f'expected {eng.ncells} anchored events, got {n_events} '
        f'(= ncells*spuck means the multiplicity bug is back)')


def test_mean_time_step_spuck2():
    np.random.seed(7)
    k = 2.0
    pt = _noop_process_project(2, k=k)
    eng = KMCEngine(pt, size=[5, 5], print_rates=False, banner=False)

    nsteps = 20000
    eng.do_steps(nsteps)
    mean_dt = eng.kmc_time / nsteps

    expected = 1.0 / (eng.ncells * k)          # 1/(25*2) = 0.02
    # dt ~ Exp(R): sigma of the mean = expected / sqrt(nsteps)
    tol = 4.0 * expected / np.sqrt(nsteps)
    assert abs(mean_dt - expected) < tol, (
        f'mean dt {mean_dt:.6e} vs expected {expected:.6e} '
        f'(+/- {tol:.2e}); factor {expected / mean_dt:.2f} off — '
        f'a factor ~spuck means anchor multiplicity is back')


def test_mean_time_step_spuck1_unchanged():
    np.random.seed(11)
    k = 3.0
    pt = _noop_process_project(1, k=k)
    eng = KMCEngine(pt, size=[6, 6], print_rates=False, banner=False)

    nsteps = 20000
    eng.do_steps(nsteps)
    mean_dt = eng.kmc_time / nsteps

    expected = 1.0 / (eng.ncells * k)
    tol = 4.0 * expected / np.sqrt(nsteps)
    assert abs(mean_dt - expected) < tol


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'  PASS  {name}')
    print('anchor time-semantics tests: ALL PASS')
