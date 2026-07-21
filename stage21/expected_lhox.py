"""Stage 2.2 Part A — expected LH_ox count in the P5 windows, from the
model's own rate constants (no logs needed; retires or keeps the
'oxide share overcorrected' worry on paper).

A CO visiting the intact oxide adsorbs at a bridge (hollows are
O-occupied). During its residence the competing channels are:
  - lateral-weakened desorption (Eq. 12 E_eff via rogal_rates),
  - LH_ox with an adjacent hollow O (Table III barriers),
  - br->hol / br->br diffusion (blocked/uphill on the intact oxide:
    hollows occupied; br partner empty -> br->br possible).
P(react | visit) = k_LH_tot / (k_LH_tot + k_des) is an upper bound
that IGNORES escape by diffusion (which only lowers it), so the
expected count is itself an upper bound.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'stage16'))

from rogal_rates import RogalRates, KB_EV  # noqa: E402

T = 393.0
rr = RogalRates(T, laterals=True, K_nearpatch=-1.10)

# intact-oxide bridge site: NN = 3 hollows (all O) + 1 br partner
# (empty)  [oxide_nn_full topology]
nn_br = [('O', 'hol')] * 3 + [('empty', 'br')]
e_eff = rr.e_eff('CO', 'br', nn_br)
k_des = rr.prefactor * math.exp(rr.beta * e_eff)

# LH_ox from CO@br with O@hol: barrier 0.9 eV (Table III, Ohol+CObr),
# one process per adjacent O pair -> 3 channels
k_lh = rr.prefactor * math.exp(-rr.beta * 0.9) * 3

p_react = k_lh / (k_lh + k_des)
visits = 178 + 178   # the two P5 seeds
mean = p_react * visits

print(f'E_eff(CO@br, 3 O-hol NN) = {e_eff:+.3f} eV '
      f'(bare -1.40; lateral weakening +{e_eff + 1.40:.2f})')
print(f'k_des = {k_des:.3e} /s   k_LH_tot = {k_lh:.3e} /s')
print(f'P(react|visit) <= {p_react:.3e}')
print(f'expected LH_ox over {visits} visits <= {mean:.3f}')
print(f'P(observe 0 | mean {mean:.3f}) = {math.exp(-mean):.3f}')
print()
print('Conclusion: 0 observed LH_ox events are CONSISTENT with the '
      'faithful model\'s own expectation (<1 event in the window). '
      'The >=60x suppression vs as-built stands; any claim about the '
      'Fig-10 oxide SHARE (paper ~23%) is untestable at this flip '
      'count and is withdrawn from the stage-2.1 phrasing where it '
      'read as an overcorrection finding.')
