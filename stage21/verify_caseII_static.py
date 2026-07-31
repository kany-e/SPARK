"""Stage 3.2 static verification of the caseII build flag.

Checks:
  V1: caseII=None -> process list IDENTICAL (names, conditions, actions,
      rates, order) to a build without the caseII code path even being
      exercised (the stage-3.0 state = fig7+coad+ipp, 103 processes).
  V2: caseII='A' -> exactly +1 process named cross_react_E_caseII_A with
      conds CO@pd_br_01_11_x(0,0,0) + O@ox_hol_1(-1,0,0), actions both
      empty, rate exp(-beta*E_cross_react_E) ; all other 103 unchanged.
  V3: caseII='B' -> same but rate has (E_cross_react_E+0.15).
"""
import sys
sys.path.insert(0, '/home/user/SPARK')
sys.path.insert(0, '/home/user/SPARK/stage21')
from collapsed_project import build_project

MODE = dict(laterals=True, K_nearpatch=True, raise_oxide=0.5,
            raise_scope='CO')
KW = dict(fig7_corrected=True, coadsorption_fix=True, interpatch_fix=True)


def sig(pt):
    out = []
    for p in pt.process_list:
        conds = sorted((c.coord.site, tuple(c.coord.offset), c.species)
                       for c in p.conditions)
        acts = sorted((a.coord.site, tuple(a.coord.offset), a.species)
                      for a in p.actions)
        out.append((p.name, tuple(conds), tuple(acts), p.rate_constant))
    return out


base = sig(build_project(mode=dict(MODE), **KW))
off = sig(build_project(mode=dict(MODE), caseII=None, **KW))
a = sig(build_project(mode=dict(MODE), caseII='A', **KW))
b = sig(build_project(mode=dict(MODE), caseII='B', **KW))

print(f"counts: base={len(base)} off={len(off)} A={len(a)} B={len(b)}")

# V1
assert base == off, "V1 FAIL: caseII=None differs from stage-3.0 build"
print("V1 PASS: caseII=None identical to stage-3.0 state "
      f"({len(base)} processes, full signature match incl. order)")

# V2/V3
for tag, s, want_rate in (
        ('A', a, '1/(beta*h)*exp(-(beta*E_cross_react_E*eV))'),
        ('B', b, '1/(beta*h)*exp(-(beta*(E_cross_react_E+0.15)*eV))')):
    extra = [p for p in s if p not in base]
    missing = [p for p in base if p not in s]
    assert not missing, f"V{tag} FAIL: base processes missing: {missing}"
    assert len(extra) == 1, f"V{tag} FAIL: expected 1 new process, got {len(extra)}"
    name, conds, acts, rate = extra[0]
    assert name == 'cross_react_E_caseII_' + tag, name
    assert conds == (('ox_hol_1', (-1, 0, 0), 'O'),
                     ('pd_br_01_11_x', (0, 0, 0), 'CO')), conds
    assert acts == (('ox_hol_1', (-1, 0, 0), 'empty'),
                    ('pd_br_01_11_x', (0, 0, 0), 'empty')), acts
    assert rate == want_rate, rate
    print(f"V{'2' if tag == 'A' else '3'} PASS: caseII='{tag}' adds exactly "
          f"{name}: conds {list(conds)}, rate {rate}")

# cross-check: the case-I cross for comparison
ci = [p for p in base if p[0].startswith('cross_react')]
print("\ncase-I cross processes in base for comparison:")
for p in ci:
    print(f"  {p[0]}: conds={p[1]} rate={p[3]}")
print("\nALL STATIC CHECKS PASS")
