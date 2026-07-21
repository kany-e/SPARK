"""Stage 2.1 P3 — A.0 geometry/PHASE_FLIP suite on the collapsed
runtime-rate model, laterals OFF and ON.

Reuses the Stage-1.4 A.0 tests verbatim (footprint / independence /
adjacent-mutex) by injecting a collapsed-project engine factory into
hr2015_state BEFORE importing the suite. Availability logic is
mode-independent; the suite must pass identically in both modes.

Usage: python3 run_a0_collapsed.py
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/user/reconkin/spark_port')

from collapsed_project import build_project  # noqa: E402
from spark.engine import KMCEngine  # noqa: E402
import hr2015_state  # noqa: E402

assert '_proc_anchor' in open(
    sys.modules['spark.engine'].__file__).read()


def _make_engine_collapsed(mode):
    def make_engine(T=393.0, p_co=5e-11, p_o2=1e-30, size=(20, 20),
                    seed=None):
        import numpy as np
        if seed is not None:
            np.random.seed(seed)
        pt = build_project(mode)
        eng = KMCEngine(pt, size=list(size), print_rates=False,
                        banner=False)
        eng.parameters.T = T
        eng.parameters.p_COgas = p_co
        eng.parameters.p_O2gas = p_o2
        return eng
    return make_engine


def run_mode(tag, mode):
    print(f'=== A.0 on collapsed model, {tag} ===')
    hr2015_state.make_engine = _make_engine_collapsed(mode)
    if 'spark_hr2015_a0' in sys.modules:
        del sys.modules['spark_hr2015_a0']
    a0 = importlib.import_module('spark_hr2015_a0')
    a0.FAILURES.clear()
    a0.test_footprint()
    a0.test_independence()
    a0.test_adjacent_mutex()
    ok = not a0.FAILURES
    print(f'=== {tag}: {"ALL PASS" if ok else a0.FAILURES} ===\n')
    return ok


if __name__ == '__main__':
    ok1 = run_mode('laterals OFF', dict(laterals=False))
    ok2 = run_mode('laterals ON (K=-1.10)',
                   dict(laterals=True, K_nearpatch=-1.10))
    sys.exit(0 if (ok1 and ok2) else 1)
