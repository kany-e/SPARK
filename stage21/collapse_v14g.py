"""Stage 2.1 P1 — mechanical collapse of canonical 1.4g to the base
model (93 non-lateral + 8 collapsed CO_diff_ox_brhol bases = 101).

Every step is assertion-gated:

A1  count structure: 5213 = 93 + 5120; 8 families; sizes 4x1024+4x256.
A2  per family: identical actions and core conditions across variants;
    NN condition-site lists identical; species combinations over
    {O, Osub, empty, null}^k complete and unique (derived from the
    CONDITIONS, with the _all suffix only cross-checked).
A3  NN site lists match the audited geometry tables
    (model_io.oxide_nn, mirroring processes_stage14g.py:985-1001).
A4  full OFF-mode parity: every variant's kmos-exact rate at
    303/343/393 K equals RogalRates(laterals=False) evaluated on that
    variant's NN configuration (rtol 1e-9). 5120 x 3 comparisons.
A5  every non-enumerated runtime-lateral process (O-side diffusion,
    CO brbr/holhol, CO_des_ox, near-patch O) has a kmos rate equal to
    the bare-rule prediction (barrier == E_tab(+uphill E0 asymmetry),
    desorption == bare E0) at 3 T.
A6  the 65 remaining static processes each have a kmos rate at 3 T
    (passthrough; numeric parity was proven at the port phase).

Output: collapsed_model.json {processes, runtime_classes, provenance}.

Scope note (LATERAL_CLASS_TABLE.md): runtime classes are CO_diff_ox_*,
O_diff_ox_* (incl. near-patch, E_tab 1.1), CO_des_ox_*. The model has
no oxide O2 desorption process (census: Table-2-complete) — nothing to
make lateral there. Stage-1.6's strip engine kept near-patch static;
this stage applies the sec-II.D uphill rule to it uniformly (same
class, family E_tab preserved).
"""

import hashlib
import itertools
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
S16 = os.path.join(os.path.dirname(HERE), 'stage16')
sys.path.insert(0, S16)

from model_io import oxide_nn, site_kind  # noqa: E402
from rogal_rates import RogalRates, KB_EV, E0, E_TAB  # noqa: E402

XML = os.path.join(S16, 'vendor', 'multilattice_v14g.xml')
CANON_SHA = 'b456f39b9abb28fb25a3716238e39f0a832a00eaae9d6a6f9be6e785935cc740'
SPECIES_CHAR = {'O': 'O', 'Osub': 's', 'empty': 'e', 'null': 'n'}
TEMPS = (303.0, 343.0, 393.0)

RUNTIME_PREFIXES = ('CO_diff_ox_', 'O_diff_ox_', 'CO_des_ox_')


def load_xml():
    with open(XML, 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == CANON_SHA
    root = ET.parse(XML).getroot()
    procs = []
    for p in root.find('process_list'):
        conds = [(c.get('coord_name'),
                  tuple(int(x) for x in c.get('coord_offset').split()[:2]),
                  c.get('species')) for c in p.findall('condition')]
        acts = [(a.get('coord_name'),
                 tuple(int(x) for x in a.get('coord_offset').split()[:2]),
                 a.get('species')) for a in p.findall('action')]
        procs.append(dict(name=p.get('name'),
                          rate=p.get('rate_constant'),
                          tof=p.get('tof_count'),
                          conds=conds, acts=acts))
    return procs


def load_rates():
    out = {}
    for T in TEMPS:
        with open(os.path.join(S16, 'vendor',
                               f'kmos_rates_T{T:g}.json')) as f:
            out[T] = json.load(f)['rates']
    return out


def collapse():
    procs = load_xml()
    rates = load_rates()
    variants = [p for p in procs if '_lat_' in p['name']]
    base = [p for p in procs if '_lat_' not in p['name']]
    assert len(procs) == 5213 and len(variants) == 5120 \
        and len(base) == 93, (len(procs), len(variants), len(base))

    fams = {}
    for p in variants:
        stem = re.sub(r'_all[Oesn]+$', '', p['name'])
        fams.setdefault(stem, []).append(p)
    sizes = sorted(len(v) for v in fams.values())
    assert len(fams) == 8 and sizes == [256] * 4 + [1024] * 4, sizes
    print(f'A1 PASS: 5213 = 93 + 5120; 8 families {sizes}')

    collapsed = []
    n_off = 0
    for stem, vs in sorted(fams.items()):
        # src/dst from the actions: src -> empty, dst -> CO
        acts0 = vs[0]['acts']
        assert all(v['acts'] == acts0 for v in vs)
        (src_site, src_off), = [(s, o) for s, o, sp in acts0
                                if sp == 'empty']
        (dst_site, dst_off), = [(s, o) for s, o, sp in acts0
                                if sp == 'CO']
        core0 = [(s, o, sp) for s, o, sp in vs[0]['conds']
                 if (s, o) in ((src_site, src_off), (dst_site, dst_off))]
        assert sorted(core0) == sorted(
            [(src_site, src_off, 'CO'), (dst_site, dst_off, 'empty')])
        nn_sites0 = [(s, o) for s, o, sp in vs[0]['conds']
                     if (s, o) not in ((src_site, src_off),
                                       (dst_site, dst_off))]
        # A2: same NN sites in every variant; full unique species grid
        seen = set()
        for v in vs:
            nn = [(s, o) for s, o, sp in v['conds']
                  if (s, o) not in ((src_site, src_off),
                                    (dst_site, dst_off))]
            assert nn == nn_sites0, (stem, v['name'])
            cfg = tuple(sp for s, o, sp in v['conds']
                        if (s, o) not in ((src_site, src_off),
                                          (dst_site, dst_off)))
            assert cfg not in seen
            seen.add(cfg)
            suffix = re.search(r'_all([Oesn]+)$', v['name']).group(1)
            assert len(suffix) == len(nn_sites0)
        assert len(seen) == 4 ** len(nn_sites0)

        # A3: geometry completeness vs audited tables
        want = set()
        for site, off, partner in ((src_site, src_off,
                                    (dst_site, dst_off)),
                                   (dst_site, dst_off,
                                    (src_site, src_off))):
            for n, du, dv, kind in oxide_nn(site):
                abs_off = (off[0] + du, off[1] + dv)
                if (n, abs_off) != partner and (n, abs_off) != \
                        (site, off):
                    want.add((n, abs_off))
        assert want == set(nn_sites0), (stem, sorted(want),
                                        sorted(nn_sites0))

        # A4: OFF parity for every variant at 3 T
        sk, dk = site_kind(src_site), site_kind(dst_site)
        nn_src_sites = {(n, (src_off[0] + du, src_off[1] + dv)):
                        site_kind(n) for n, du, dv, _ in
                        oxide_nn(src_site)
                        if (n, (src_off[0] + du, src_off[1] + dv))
                        != (dst_site, dst_off)}
        nn_dst_sites = {(n, (dst_off[0] + du, dst_off[1] + dv)):
                        site_kind(n) for n, du, dv, _ in
                        oxide_nn(dst_site)
                        if (n, (dst_off[0] + du, dst_off[1] + dv))
                        != (src_site, src_off)}
        for v in vs:
            occ = {(s, o): sp for s, o, sp in v['conds']}
            for T in TEMPS:
                rr = RogalRates(T, laterals=False)
                nn_s = [(occ.get(k, 'empty'), kk)
                        for k, kk in nn_src_sites.items()]
                nn_d = [(occ.get(k, 'empty'), kk)
                        for k, kk in nn_dst_sites.items()]
                pred = rr.diffusion_rate('CO', sk, dk, nn_s, nn_d)
                ref = rates[T][v['name']]
                assert ref > 0 and abs(pred - ref) <= 1e-9 * ref, (
                    v['name'], T, pred, ref)
                n_off += 1

        cname = stem.replace('_lat_', '_')
        collapsed.append(dict(
            name=cname, rate=None, tof=vs[0]['tof'],
            conds=[(src_site, src_off, 'CO'),
                   (dst_site, dst_off, 'empty')],
            acts=acts0,
            runtime=dict(kind='diff', sp='CO', src=[src_site, src_off],
                         dst=[dst_site, dst_off],
                         E_tab=E_TAB[('CO', 'br', 'hol')])))
    print(f'A2/A3 PASS: 8 families, NN sites == audited tables')
    print(f'A4 PASS: OFF parity on {n_off} (variant,T) pairs, '
          f'rtol 1e-9')

    # A5: non-enumerated runtime classes — bare-rule parity
    runtime_bare = [p for p in base
                    if p['name'].startswith(RUNTIME_PREFIXES)]
    n5 = 0
    class_meta = {}
    for p in runtime_bare:
        name = p['name']
        if name.startswith('CO_des_ox'):
            (site, off), = [(s, o) for s, o, sp in p['conds']
                            if sp == 'CO']
            kind = site_kind(site)
            for T in TEMPS:
                rr = RogalRates(T, laterals=False)
                pred = rr.desorption_rate(kind, [])
                ref = rates[T][name]
                assert abs(pred - ref) <= 1e-9 * ref, (name, T, pred,
                                                       ref)
                n5 += 1
            class_meta[name] = dict(kind='des', sp='CO',
                                    src=[site, list(off)], dst=None,
                                    E_tab=None)
            continue
        sp = 'CO' if name.startswith('CO_') else 'O'
        (src_site, src_off), = [(s, o) for s, o, spc in p['acts']
                                if spc == 'empty']
        (dst_site, dst_off), = [(s, o) for s, o, spc in p['acts']
                                if spc == sp]
        sk, dk = site_kind(src_site), site_kind(dst_site)
        # family E_tab recovered from the 393 K rate (barrier =
        # tab + max(0, bare uphill)); verified vs expectations
        T = 393.0
        k = rates[T][name]
        barrier = -KB_EV * T * math.log(k / RogalRates(T, False)
                                        .prefactor)
        bare_up = max(0.0, E0[(sp, dk)] - E0[(sp, sk)])
        etab = barrier - bare_up
        if 'nearpatch' in name:
            assert abs(etab - 1.10) < 1e-6, (name, etab)
        else:
            assert abs(etab - E_TAB.get((sp, sk, dk),
                                        E_TAB.get((sp, dk, sk)))) \
                < 1e-6, (name, etab)
        # 3-T bare parity with the recovered E_tab
        for T in TEMPS:
            rr = RogalRates(T, laterals=False)
            pred = rr.prefactor * math.exp(
                -(etab + max(0.0, E0[(sp, dk)] - E0[(sp, sk)]))
                / (KB_EV * T))
            ref = rates[T][name]
            assert abs(pred - ref) <= 1e-9 * ref, (name, T, pred, ref)
            n5 += 1
        class_meta[name] = dict(kind='diff', sp=sp,
                                src=[src_site, list(src_off)],
                                dst=[dst_site, list(dst_off)],
                                E_tab=round(etab, 6))
    print(f'A5 PASS: bare-rule parity on {n5} (proc,T) pairs '
          f'({len(runtime_bare)} non-enumerated runtime processes)')

    # A6 + assemble output
    static = [p for p in base if p['name'] not in class_meta]
    for p in static:
        for T in TEMPS:
            assert p['name'] in rates[T]
    out_procs = []
    for p in base:
        entry = dict(name=p['name'], rate=p['rate'], tof=p['tof'],
                     conds=[[s, list(o), sp] for s, o, sp in
                            p['conds']],
                     acts=[[s, list(o), sp] for s, o, sp in p['acts']])
        if p['name'] in class_meta:
            entry['runtime'] = class_meta[p['name']]
        out_procs.append(entry)
    for c in collapsed:
        c2 = dict(c)
        c2['conds'] = [[s, list(o), sp] for s, o, sp in c['conds']]
        c2['acts'] = [[s, list(o), sp] for s, o, sp in c['acts']]
        out_procs.append(c2)
    n_rt = sum(1 for p in out_procs if 'runtime' in p)
    print(f'A6 PASS: {len(static)} static passthrough; collapsed model '
          f'{len(out_procs)} processes ({n_rt} runtime-lateral)')
    assert len(out_procs) == 101 and n_rt == 36, (len(out_procs), n_rt)

    with open(os.path.join(HERE, 'collapsed_model.json'), 'w') as f:
        json.dump(dict(provenance=dict(xml_sha256=CANON_SHA,
                                       total=5213, variants=5120),
                       processes=out_procs), f, indent=1)
    print('wrote collapsed_model.json')


if __name__ == '__main__':
    collapse()
