"""Stage 3.2 §2 dynamic regression gate: with the case-II pathway
DISABLED (reloc_shift=0.0), the model must be bit-identical to the
stage-3.0 state over >=10^4 steps (same seed, same init)."""
import sys
import numpy as np
sys.path.insert(0, '/home/user/SPARK')
sys.path.insert(0, '/home/user/SPARK/stage21')
sys.path.insert(0, '/home/user/reconkin/spark_port')
sys.path.insert(0, '/home/user/reconkin/stage20_correction')
from collapsed_project import build_project
from spark.engine import KMCEngine
import hr2015_state as H  # noqa


def make(with_flag):
    kw = dict(fig7_corrected=True, coadsorption_fix=True,
              interpatch_fix=True, flip_unfreeze=True)
    if with_flag:
        kw['reloc_shift'] = 0.0
    pt = build_project(dict(laterals=True, K_nearpatch=-1.10,
                            raise_oxide=0.5, raise_scope='CO'), **kw)
    eng = KMCEngine(pt, size=[20, 20], print_rates=False, banner=False)
    eng.parameters.T = 393.0
    eng.parameters.p_COgas = 5e-11
    eng.parameters.p_O2gas = 1e-30
    return eng


N = 20000
res = []
for with_flag in (False, True):
    eng = make(with_flag)
    np.random.seed(1)
    H.set_intact_oxide(eng, rebuild=False)
    from initial_state_20x20 import phase_flip_actions, flip_cell_exact
    LY = 20
    acts = phase_flip_actions()
    for cy in range(LY):
        flip_cell_exact(eng, 0, cy, acts)
    eng._rebuild_avail_sites()
    eng._rebuild_per_site_rates()
    eng.do_steps(N)
    res.append((eng.lattice.copy(), eng.procstat.copy(),
                eng.kmc_time, eng.kmc_step, np.random.get_state()))

(l0, p0, t0, s0, r0), (l1, p1, t1, s1, r1) = res
ok_lat = bool((l0 == l1).all())
ok_ps = bool((p0 == p1).all())
ok_t = (t0 == t1)
ok_rng = all((np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b)
             for a, b in zip(r0, r1))
print(f'steps={N}  lattice_identical={ok_lat}  procstat_identical={ok_ps}')
print(f'kmc_time: {t0!r} == {t1!r} -> {ok_t}   rng_state_identical={ok_rng}')
assert ok_lat and ok_ps and ok_t and ok_rng, 'REGRESSION GATE FAIL'
print(f'GATE PASS: reloc_shift=0.0 bit-identical to stage-3.0 build over {N} steps')
