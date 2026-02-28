#!/usr/bin/env python3
"""
스텝별 궤적 형태 분석 (localFwdY vs pureZ) — 순수 Python (no pandas/numpy)

각 스텝 구간 내 궤적에서 추출하는 피처:
  1. trajAngle  : PCA 주축 방향각 (deg) — 0°=수평(fwdY), 90°=수직(Z)
  2. aspectRatio: 궤적 종횡비 (장축/단축)
  3. zRange     : pureZ 진폭 (max-min)
  4. yRange     : localFwdY 진폭 (max-min)
  5. zDominance : zRange / (yRange + zRange)
  6. netZ       : 스텝 끝 - 시작의 Z 변위
  7. loopArea   : 궤적이 감싸는 면적 (Shoelace)
"""
import csv, math, os, sys
from collections import defaultdict

def load_tsv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append(r)
    return rows

def to_float(rows, cols):
    for r in rows:
        for c in cols:
            try: r[c] = float(r[c])
            except: r[c] = 0.0

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def std(vals):
    if len(vals) < 2: return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m)**2 for v in vals) / (len(vals) - 1))

def analyze_steps(rows, label):
    cols = ['localFwdY', 'pureZ', 'isStep', 'stepIdx', 'time_s']
    to_float(rows, cols)
    for r in rows:
        r['isStep'] = int(r['isStep'])
        r['stepIdx'] = int(float(r['stepIdx']))

    step_positions = [i for i, r in enumerate(rows) if r['isStep'] == 1]
    if len(step_positions) < 2:
        return []

    results = []
    for si in range(len(step_positions)):
        start = step_positions[si - 1] if si > 0 else 0
        end = step_positions[si]
        seg = rows[start:end+1]
        if len(seg) < 5:
            continue

        fY = [s['localFwdY'] for s in seg]
        pZ = [s['pureZ'] for s in seg]
        gt = seg[-1].get('floorGT', 'ground')
        step_num = seg[-1]['stepIdx']
        n = len(fY)

        # PCA via 2x2 covariance
        my = mean(fY)
        mz = mean(pZ)
        cy = [y - my for y in fY]
        cz = [z - mz for z in pZ]
        cov_yy = sum(a*a for a in cy) / n
        cov_zz = sum(a*a for a in cz) / n
        cov_yz = sum(a*b for a, b in zip(cy, cz)) / n

        # eigenvalues of 2x2 symmetric matrix
        tr = cov_yy + cov_zz
        det = cov_yy * cov_zz - cov_yz * cov_yz
        disc = max(0, tr*tr/4 - det)
        sqrt_disc = math.sqrt(disc)
        ev1 = tr/2 + sqrt_disc  # larger
        ev2 = tr/2 - sqrt_disc  # smaller

        # eigenvector for ev1
        if abs(cov_yz) > 1e-12:
            vy = cov_yz
            vz = ev1 - cov_yy
        elif abs(cov_yy - ev1) < abs(cov_zz - ev1):
            vy = 1.0; vz = 0.0
        else:
            vy = 0.0; vz = 1.0
        mag = math.sqrt(vy*vy + vz*vz + 1e-15)
        vy /= mag; vz /= mag

        trajAngle = math.degrees(math.atan2(abs(vz), abs(vy)))
        aspectRatio = math.sqrt(ev1 / (ev2 + 1e-12))

        zRange = max(pZ) - min(pZ)
        yRange = max(fY) - min(fY)
        zDominance = zRange / (yRange + zRange + 1e-9)

        netZ = pZ[-1] - pZ[0]
        netY = fY[-1] - fY[0]

        # Shoelace area
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += fY[i] * pZ[j] - fY[j] * pZ[i]
        loopArea = abs(area) / 2.0

        rmsY = math.sqrt(sum(y*y for y in fY) / n)
        rmsZ = math.sqrt(sum(z*z for z in pZ) / n)

        results.append({
            'dataset': label,
            'stepIdx': step_num,
            'gt': gt,
            'nSamples': n,
            'trajAngle': trajAngle,
            'aspectRatio': aspectRatio,
            'zRange': zRange,
            'yRange': yRange,
            'zDominance': zDominance,
            'netZ': netZ,
            'netY': netY,
            'loopArea': loopArea,
            'rmsY': rmsY,
            'rmsZ': rmsZ,
        })
    return results

# ── Main ──
out_dir = os.path.dirname(os.path.abspath(__file__))
files = {
    '일반보행': os.path.join(out_dir, 'batch_Floor_sim_data_일반보행.tsv'),
    '살살걸음': os.path.join(out_dir, 'batch_Floor_sim_data_살살걸음.tsv'),
    '거칠게걸음': os.path.join(out_dir, 'batch_Floor_sim_data_거칠게 걸음.tsv'),
}

all_results = []
for label, fpath in files.items():
    if not os.path.exists(fpath):
        print(f"SKIP: {fpath}")
        continue
    rows = load_tsv(fpath)
    result = analyze_steps(rows, label)
    all_results.extend(result)
    print(f"  {label}: {len(result)} steps analyzed")

# TSV 저장
out_path = os.path.join(out_dir, 'step_trajectory_analysis.tsv')
fieldnames = ['dataset','stepIdx','gt','nSamples','trajAngle','aspectRatio',
              'zRange','yRange','zDominance','netZ','netY','loopArea','rmsY','rmsZ']
with open(out_path, 'w') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    w.writeheader()
    for r in all_results:
        for k in ['trajAngle','aspectRatio','zRange','yRange','zDominance',
                   'netZ','netY','loopArea','rmsY','rmsZ']:
            r[k] = f"{r[k]:.4f}"
        w.writerow(r)
print(f"\nSaved: {out_path} ({len(all_results)} steps)")

# ── 통계 요약 ──
features = ['trajAngle','aspectRatio','zRange','yRange','zDominance','netZ','netY','loopArea']

print("\n" + "="*90)
print("  STEP TRAJECTORY ANALYSIS: ground vs up (stair)")
print("="*90)

datasets = sorted(set(r['dataset'] for r in all_results))
for ds in datasets:
    sub = [r for r in all_results if r['dataset'] == ds]
    print(f"\n{'─'*50}")
    print(f"  Dataset: {ds}")
    print(f"{'─'*50}")

    for gt_label in ['ground', 'up']:
        g = [r for r in sub if r['gt'] == gt_label]
        if not g:
            continue
        print(f"\n  [{gt_label}] ({len(g)} steps)")
        for col in features:
            vals = [float(r[col]) for r in g]
            m = mean(vals)
            s = std(vals)
            mn = min(vals)
            mx = max(vals)
            print(f"    {col:14s}:  mean={m:7.3f}  std={s:6.3f}  min={mn:7.3f}  max={mx:7.3f}")

# ── Ground vs UP 분리도 (Cohen's d) ──
print(f"\n{'='*90}")
print("  FEATURE SEPARATION: Cohen's d = |mean_gnd - mean_up| / pooled_std")
print("  ★★★ d>1.5 (완벽 분리)  ★★ d>0.8 (좋음)  ★ d>0.5 (보통)")
print("="*90)

for ds in datasets:
    sub = [r for r in all_results if r['dataset'] == ds]
    gnd = [r for r in sub if r['gt'] == 'ground']
    up  = [r for r in sub if r['gt'] == 'up']
    if not gnd or not up:
        continue
    print(f"\n  {ds}:")
    for col in features:
        vg = [float(r[col]) for r in gnd]
        vu = [float(r[col]) for r in up]
        mg, sg = mean(vg), std(vg)
        mu, su = mean(vu), std(vu)
        pooled = math.sqrt((sg**2 + su**2) / 2 + 1e-12)
        d = abs(mg - mu) / pooled
        stars = "★★★" if d > 1.5 else ("★★" if d > 0.8 else ("★" if d > 0.5 else "  "))
        print(f"    {col:14s}:  gnd={mg:7.3f}  up={mu:7.3f}  d={d:.2f} {stars}")
