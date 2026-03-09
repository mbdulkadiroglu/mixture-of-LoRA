import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: f'{x:.6f}')

pkl = '/data/mehmet/projects/mixture-of-LoRA/data/routerbench_cache/datasets--withmartian--routerbench/snapshots/784021482c3f320c6619ed4b3bb3b41a21424fcb/routerbench_0shot.pkl'
df = pd.read_pickle(pkl)

print('11. COST STATISTICS PER MODEL')
print('=' * 80)
cost_cols = [c for c in df.columns if '|total_cost' in c]
rows = []
for col in cost_cols:
    m = col.replace('|total_cost', '')
    c = df[col]
    rows.append({'model': m, 'mean': c.mean(), 'std': c.std(), 'min': c.min(), 'med': c.median(), 'max': c.max(), 'total': c.sum()})
cdf = pd.DataFrame(rows).sort_values('mean', ascending=False)
print(cdf.to_string(index=False))
print()
print('COST-EFFICIENCY')
print('=' * 80)
ms = ['WizardLM/WizardLM-13B-V1.2','claude-instant-v1','claude-v1','claude-v2','gpt-3.5-turbo-1106','gpt-4-1106-preview','meta/code-llama-instruct-34b-chat','meta/llama-2-70b-chat','mistralai/mistral-7b-chat','mistralai/mixtral-8x7b-chat','zero-one-ai/Yi-34B-Chat']
for m in ms:
    s = pd.to_numeric(df[m], errors='coerce')
    tc = df[m+'|total_cost'].sum()
    sc = s.sum()
    cpc = tc/sc if sc > 0 else float('inf')
    print(f'  {m:50s} acc={s.mean():.4f} avg_cost={df[m+chr(124)+"total_cost"].mean():.6f} total={tc:.2f} per_correct={cpc:.6f}')

print()
print('SCORE VALUE DISTRIBUTION')
print('=' * 80)
for m in ms:
    vals = pd.to_numeric(df[m], errors='coerce')
    vc = vals.value_counts().sort_index()
    print(f'  {m}:')
    for v, cnt in vc.items():
        print(f'    {v:.2f}: {cnt}')
    print()
