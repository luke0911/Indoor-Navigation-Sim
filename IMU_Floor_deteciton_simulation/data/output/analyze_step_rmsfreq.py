#!/usr/bin/env python3
"""스텝 단위 피처 분리도 분석 — rmsZdivFreq, procRmsZ, stepFreqHz, stepVarH 등"""
import csv, math, os

def load_tsv(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def mean(v): return sum(v)/len(v) if v else 0
def std(v):
    if len(v)<2: return 0
    m=mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))

out_dir = os.path.dirname(os.path.abspath(__file__))
files = {
    '일반보행': os.path.join(out_dir, 'batch_Floor_sim_data_일반보행.tsv'),
    '살살걸음': os.path.join(out_dir, 'batch_Floor_sim_data_살살걸음.tsv'),
    '거칠게걸음': os.path.join(out_dir, 'batch_Floor_sim_data_거칠게 걸음.tsv'),
}

features = ['rmsZdivFreq', 'procRmsZ', 'stepFreqHz', 'stepVarH',
            'stairScore', 'varYZRatio', 'dirR1', 'dmag', 'recentStepVarYZ']

for label, fpath in files.items():
    if not os.path.exists(fpath): continue
    rows = load_tsv(fpath)
    # 스텝만 추출
    steps = [r for r in rows if r.get('isStep') == '1']

    gnd = [r for r in steps if r.get('floorGT') == 'ground']
    up  = [r for r in steps if r.get('floorGT') == 'up']

    print(f"\n{'='*75}")
    print(f"  {label}  (ground:{len(gnd)} steps,  up:{len(up)} steps)")
    print(f"{'='*75}")
    print(f"  {'feature':18s}  {'gnd_mean':>8s}  {'up_mean':>8s}  {'gnd_std':>8s}  {'up_std':>8s}  {'d':>6s}  sep")
    print(f"  {'-'*70}")

    for feat in features:
        try:
            vg = [float(r[feat]) for r in gnd if r.get(feat)]
            vu = [float(r[feat]) for r in up if r.get(feat)]
        except:
            continue
        if not vg or not vu: continue
        mg, sg = mean(vg), std(vg)
        mu, su = mean(vu), std(vu)
        pooled = math.sqrt((sg**2 + su**2)/2 + 1e-12)
        d = abs(mg - mu) / pooled
        stars = "★★★" if d>1.5 else ("★★" if d>0.8 else ("★" if d>0.5 else ""))
        print(f"  {feat:18s}  {mg:8.3f}  {mu:8.3f}  {sg:8.3f}  {su:8.3f}  {d:6.2f}  {stars}")
