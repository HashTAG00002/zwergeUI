#!/usr/bin/env python3
"""Inject main-paper labels into supplementary.aux for cross-document references."""
with open('supplementary.aux', 'r') as f:
    aux = f.read()

labels = {
    'sec:probe': '2', 'sec:method:stage1': '2', 'sec:method': '4',
    'sec:method:stage2': '4.1', 'sec:method:ensemble': '4.2',
    'sec:analysis': '3', 'sec:analysis:layerwise': '3.1',
    'sec:analysis:serialization': '3.2', 'sec:analysis:counterfactual': '3.3',
    'sec:analysis:resolution': '3.4', 'sec:experiments': '5',
    'sec:experiments:main': '5.2', 'sec:experiments:sspro': '5.3',
    'tab:main_ensemble': '1', 'tab:resolution_comparison': '2',
    'fig:serialization_lens': '1', 'fig:counterfactual': '2',
    'eq:fusion': '2', 'eq:probe': '1', 'eq:gaussian': '2', 'eq:loss_s2': '3',
}

added = 0
for k, v in labels.items():
    if k not in aux:
        aux += '\\newlabel{' + k + '}{{' + v + '}{1}}\n'
        added += 1

with open('supplementary.aux', 'w') as f:
    f.write(aux)
print(f"Injected {added} labels into supplementary.aux")
