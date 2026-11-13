# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
import json, sys
d=json.load(open(sys.argv[1]))
for r in d['rows']:
    print(r['preset'], r['fresh_bytes'])
    for k,v in r['steps'].items(): print('  %-26s %7.2f'%(k,v['median_ms']))
    for n in ('as_served','drop_to_1_tower'):
        v=r[n]; print(' ',n,v['bytes'],'level',v['level'],'err',v['max_abs_error'])
        for k in ('deserialize_file','decrypt','get_values','to_numpy','block_total'): print('    %-22s %7.2f'%(k,v[k]['median_ms']))
    c=r['client_drop_to_1_tower']; print('  client_drop: level_reduce %.2f block_total %.2f err %.3g (as served %.3g)'%(c['level_reduce']['median_ms'],c['block_total']['median_ms'],c['max_abs_error_20_runs'],c['as_served_max_abs_error_20_runs']))
