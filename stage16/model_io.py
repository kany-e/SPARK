"""Minimal kmos-XML parsing + sqrt5-oxide geometry for Stage 1.6.

Self-contained (no reconkin imports). The NN tables mirror the 1.4g
generator (processes_stage14g.py:985-1001) and the 1.5a audit.
"""

import ast
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, 'vendor', 'multilattice_v14g.xml')

# oxide in-cell/cross-cell NN membership: (site, du, dv)
BR_NN_HOLS = {
    'ox_br_0': [('ox_hol_1', 0, 0), ('ox_hol_0', 0, -1),
                ('ox_hol_0', 0, 0)],
    'ox_br_1': [('ox_hol_0', 0, 0), ('ox_hol_1', 0, 1),
                ('ox_hol_1', 0, 0)],
}
HOL_NN_BRS = {
    'ox_hol_0': [('ox_br_1', 0, 0), ('ox_br_0', 0, 0),
                 ('ox_br_0', 0, 1)],
    'ox_hol_1': [('ox_br_0', 0, 0), ('ox_br_1', 0, -1),
                 ('ox_br_1', 0, 0)],
}
HOL_NN_HOLS = {
    'ox_hol_0': [('ox_hol_1', 0, 0)],
    'ox_hol_1': [('ox_hol_0', 0, 0)],
}


def site_kind(name):
    if name.startswith('ox_br'):
        return 'br'
    if name.startswith('ox_hol'):
        return 'hol'
    return 'pd'


def oxide_nn(site, sp='CO'):
    """All lateral-relevant NN of an oxide site: [(name, du, dv, kind)].
    For both CO and O the NN set is the br<->hol + hol<->hol adjacency."""
    if site.startswith('ox_br'):
        lst = BR_NN_HOLS[site]
    else:
        lst = HOL_NN_BRS[site] + HOL_NN_HOLS[site]
    return [(n, du, dv, site_kind(n)) for n, du, dv in lst]


def parse_model(path=XML_PATH):
    """Parse the kmos v0.4 XML into plain dicts (subset of the Stage
    1.4 translator's schema; re-implemented here so stage16 has no
    dependency on the forfeited reconkin branch)."""
    root = ET.parse(path).getroot()
    params = {}
    for p in root.find('parameter_list'):
        v = p.get('value')
        try:
            params[p.get('name')] = float(v)
        except ValueError:
            params[p.get('name')] = eval(v)  # '3.894e-19/4' etc.
    layer = root.find('lattice').find('layer')
    sites = [{'name': s.get('type'),
              'default_species': s.get('default_species')}
             for s in layer.findall('site')]
    procs = []
    for p in root.find('process_list').findall('process'):
        def ent(tag):
            return [{'site': e.get('coord_name'),
                     'offset': tuple(int(x) for x in
                                     e.get('coord_offset').split()),
                     'species': e.get('species')}
                    for e in p.findall(tag)]
        tof = p.get('tof_count')
        procs.append({'name': p.get('name'),
                      'rate_constant': p.get('rate_constant'),
                      'tof_count': ast.literal_eval(tof) if tof else {},
                      'conditions': ent('condition'),
                      'actions': ent('action')})
    return {'params': params, 'sites': sites, 'processes': procs}
