#!/usr/bin/env python3
"""
교차 검증: 4가지 피처의 ground vs stair 분리력 분석
- procRmsZ (수직 충격량) → 실패 증명
- varYZRatio (궤적 집중도) → 성공 검증
- stepVarH (수평 에너지) → 성공 검증
- zPeakYRatio (엇박자 비율) → 성공 검증
+ zEnergyFrac (Z축 에너지 비율) → 추가 검증
"""
import os, math, csv, sys

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUT_DIR), "data", "output")

# ── 데이터셋 정의 ──
DATASETS = {
    "거칠게걸음": "step_analysis_Floor_sim_data_거칠게 걸음.tsv",
    "살살걸음":   "step_analysis_Floor_sim_data_살살걸음.tsv",
    "일반보행":   "step_analysis_Floor_sim_data_일반보행.tsv",
    "sample":    "step_analysis_Floor_sim_data_sample.tsv",
    "쿼터니언":   "step_analysis_Floor_sim_data_260224-163018.tsv",
    "sim_data":  "step_analysis_sim_data_260225-154018.tsv",
    "팔돌림":     "step_analysis_중간에 팔엄청 돌림.tsv",
}

# ── 통계 함수 (numpy 없이) ──
def mean(v): return sum(v) / max(len(v), 1)
def median(v):
    s = sorted(v); n = len(s)
    if n == 0: return 0
    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2
def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m)**2 for x in v) / max(len(v), 1))
def percentile(v, p):
    s = sorted(v); n = len(s)
    if n == 0: return 0
    k = (n - 1) * p / 100.0
    f = int(k); c = min(f + 1, n - 1)
    return s[f] + (k - f) * (s[c] - s[f])
def iqr(v): return percentile(v, 75) - percentile(v, 25)

def mann_whitney_u(a, b):
    """간이 Mann-Whitney U test (양측). 정규 근사."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan'), float('nan')
    combined = [(v, 'a') for v in a] + [(v, 'b') for v in b]
    combined.sort(key=lambda x: x[0])
    # Rank 계산 (tie 처리)
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    Ra = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == 'a')
    U1 = Ra - na * (na + 1) / 2
    U2 = na * nb - U1
    U = min(U1, U2)
    # 정규 근사
    mu = na * nb / 2.0
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    if sigma == 0:
        return U, float('nan')
    z = (U - mu) / sigma
    # 양측 p-value 근사 (표준정규 CDF)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return U, p

def effect_size_r(a, b):
    """rank-biserial correlation (effect size for Mann-Whitney)"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float('nan')
    U, _ = mann_whitney_u(a, b)
    # U is min(U1,U2); we need U1 for rank-biserial
    combined = [(v, 'a') for v in a] + [(v, 'b') for v in b]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    Ra = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == 'a')
    U1 = Ra - na * (na + 1) / 2
    r = 1 - 2 * U1 / (na * nb)  # rank-biserial
    return r

def overlap_pct(a, b):
    """두 분포의 IQR 겹침 정도 (0 = 완전 분리, 100 = 완전 겹침)"""
    if not a or not b: return float('nan')
    a25, a75 = percentile(a, 25), percentile(a, 75)
    b25, b75 = percentile(b, 25), percentile(b, 75)
    overlap_lo = max(a25, b25)
    overlap_hi = min(a75, b75)
    if overlap_lo >= overlap_hi: return 0.0
    overlap_range = overlap_hi - overlap_lo
    total_range = max(a75, b75) - min(a25, b25)
    if total_range == 0: return 100.0
    return overlap_range / total_range * 100

# ── 데이터 로딩 ──
def load_steps(filepath):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    return rows

print("=" * 90)
print("  교차 검증: 4+1 피처의 Ground vs Stair 분리력 분석")
print("=" * 90)

# ── 검증할 피처들 ──
FEATURES = [
    ("procRmsZ",    "❌ 수직 충격량",       "higher=stair?"),
    ("varYZRatio",  "🎯 궤적 집중도",       "lower=stair"),
    ("stepVarH",    "🛡️ 수평 에너지",       "lower=stair"),
    ("zPeakYRatio", "⚖️ 엇박자 비율",       "lower=stair"),
    ("zEnergyFrac", "🆕 Z에너지 비율",      "higher=stair"),
]

all_results = []  # for CSV export

for feat_key, feat_name, direction in FEATURES:
    print(f"\n{'─'*90}")
    print(f"  {feat_name} ({feat_key})  — 기대 방향: {direction}")
    print(f"{'─'*90}")
    print(f"  {'Dataset':<12} {'N_gnd':>5} {'N_up':>4}  {'Gnd_med':>8} {'Up_med':>8} {'Gap':>7}  {'IQR_ovlp':>8} {'p-value':>10} {'EffSize':>8} {'판정':>6}")

    combined_gnd = []
    combined_up = []

    for ds_name, ds_file in DATASETS.items():
        path = os.path.join(DATA_DIR, ds_file)
        if not os.path.exists(path):
            print(f"  {ds_name}: FILE NOT FOUND")
            continue
        steps = load_steps(path)

        gnd_vals = []
        up_vals = []
        for s in steps:
            gt = s.get('floorGT', '').strip()
            try:
                val = float(s[feat_key])
            except (KeyError, ValueError):
                continue
            if gt == 'ground':
                gnd_vals.append(val)
            elif gt in ('up', 'down', 'stair'):
                up_vals.append(val)

        combined_gnd.extend(gnd_vals)
        combined_up.extend(up_vals)

        if not gnd_vals or not up_vals:
            print(f"  {ds_name:<12} {len(gnd_vals):>5} {len(up_vals):>4}  — no stair data —")
            continue

        med_g = median(gnd_vals)
        med_u = median(up_vals)
        gap = med_u - med_g
        ovlp = overlap_pct(gnd_vals, up_vals)
        _, pval = mann_whitney_u(gnd_vals, up_vals)
        eff = effect_size_r(gnd_vals, up_vals)

        # 판정
        if pval < 0.01 and ovlp < 30:
            verdict = "✅강함"
        elif pval < 0.05:
            verdict = "⚠️약함"
        else:
            verdict = "❌실패"

        pval_str = f"{pval:.1e}" if not math.isnan(pval) else "N/A"

        print(f"  {ds_name:<12} {len(gnd_vals):>5} {len(up_vals):>4}  {med_g:>8.3f} {med_u:>8.3f} {gap:>+7.3f}  {ovlp:>7.1f}% {pval_str:>10} {eff:>+8.3f} {verdict}")

        all_results.append({
            'feature': feat_key, 'dataset': ds_name,
            'n_ground': len(gnd_vals), 'n_stair': len(up_vals),
            'ground_median': med_g, 'stair_median': med_u, 'gap': gap,
            'iqr_overlap_pct': ovlp, 'p_value': pval, 'effect_size_r': eff,
            'verdict': verdict,
        })

    # Combined
    if combined_gnd and combined_up:
        med_g = median(combined_gnd)
        med_u = median(combined_up)
        gap = med_u - med_g
        ovlp = overlap_pct(combined_gnd, combined_up)
        _, pval = mann_whitney_u(combined_gnd, combined_up)
        eff = effect_size_r(combined_gnd, combined_up)
        pval_str = f"{pval:.1e}" if not math.isnan(pval) else "N/A"

        if pval < 0.01 and ovlp < 30:
            verdict = "✅강함"
        elif pval < 0.05:
            verdict = "⚠️약함"
        else:
            verdict = "❌실패"

        print(f"  {'COMBINED':<12} {len(combined_gnd):>5} {len(combined_up):>4}  {med_g:>8.3f} {med_u:>8.3f} {gap:>+7.3f}  {ovlp:>7.1f}% {pval_str:>10} {eff:>+8.3f} {verdict}")

        all_results.append({
            'feature': feat_key, 'dataset': 'COMBINED',
            'n_ground': len(combined_gnd), 'n_stair': len(combined_up),
            'ground_median': med_g, 'stair_median': med_u, 'gap': gap,
            'iqr_overlap_pct': ovlp, 'p_value': pval, 'effect_size_r': eff,
            'verdict': verdict,
        })

# ── Claim 1 검증: procRmsZ로 threshold 2.0 적용 시 오류율 ──
print(f"\n\n{'='*90}")
print("  Claim 1 검증: procRmsZ > 2.0 → 계단? (절대 임계값의 실패)")
print(f"{'='*90}")
threshold = 2.0
for ds_name, ds_file in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path): continue
    steps = load_steps(path)
    tp = fp = tn = fn = 0
    for s in steps:
        gt = s.get('floorGT', '').strip()
        try: val = float(s['procRmsZ'])
        except: continue
        is_stair = gt in ('up', 'down', 'stair')
        pred_stair = val > threshold
        if is_stair and pred_stair: tp += 1
        elif not is_stair and pred_stair: fp += 1
        elif not is_stair and not pred_stair: tn += 1
        elif is_stair and not pred_stair: fn += 1
    total = tp + fp + tn + fn
    if total == 0: continue
    acc = (tp + tn) / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 0.001)
    print(f"  {ds_name:<12}  TP={tp:>3} FP={fp:>3} TN={tn:>3} FN={fn:>3}  Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

# ── Claim 2 검증: varYZRatio < 0.8 → 계단? ──
print(f"\n{'='*90}")
print("  Claim 2 검증: varYZRatio < 0.8 → 계단?")
print(f"{'='*90}")
threshold = 0.8
for ds_name, ds_file in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path): continue
    steps = load_steps(path)
    tp = fp = tn = fn = 0
    for s in steps:
        gt = s.get('floorGT', '').strip()
        try: val = float(s['varYZRatio'])
        except: continue
        is_stair = gt in ('up', 'down', 'stair')
        pred_stair = val < threshold
        if is_stair and pred_stair: tp += 1
        elif not is_stair and pred_stair: fp += 1
        elif not is_stair and not pred_stair: tn += 1
        elif is_stair and not pred_stair: fn += 1
    total = tp + fp + tn + fn
    if total == 0: continue
    acc = (tp + tn) / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 0.001)
    print(f"  {ds_name:<12}  TP={tp:>3} FP={fp:>3} TN={tn:>3} FN={fn:>3}  Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

# ── Claim 3 검증: stepVarH — ground > stair 방향 확인 ──
print(f"\n{'='*90}")
print("  Claim 3 검증: stepVarH — 거칠게걸음 Ground에서 stair 대비 얼마나 높은가?")
print(f"{'='*90}")
for ds_name, ds_file in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path): continue
    steps = load_steps(path)
    gnd = [float(s['stepVarH']) for s in steps if s.get('floorGT','').strip() == 'ground']
    up = [float(s['stepVarH']) for s in steps if s.get('floorGT','').strip() in ('up','down','stair')]
    if not gnd or not up: continue
    ratio = median(gnd) / max(median(up), 0.001)
    print(f"  {ds_name:<12}  Gnd_med={median(gnd):.3f}  Stair_med={median(up):.3f}  Gnd/Stair={ratio:.2f}x")

# ── CSV 저장 ──
csv_path = os.path.join(OUT_DIR, "cross_validation_results.tsv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'feature', 'dataset', 'n_ground', 'n_stair',
        'ground_median', 'stair_median', 'gap',
        'iqr_overlap_pct', 'p_value', 'effect_size_r', 'verdict'
    ], delimiter='\t')
    writer.writeheader()
    for r in all_results:
        writer.writerow(r)
print(f"\n결과 저장: {csv_path}")

# ── 요약 판정표 ──
print(f"\n\n{'='*90}")
print("  최종 교차 검증 요약 (COMBINED 기준)")
print(f"{'='*90}")
print(f"  {'Feature':<14} {'Ground_med':>10} {'Stair_med':>10} {'Gap':>8} {'IQR겹침':>8} {'p-value':>10} {'판정':>8}")
for r in all_results:
    if r['dataset'] == 'COMBINED':
        pval_str = f"{r['p_value']:.1e}" if not math.isnan(r['p_value']) else "N/A"
        print(f"  {r['feature']:<14} {r['ground_median']:>10.3f} {r['stair_median']:>10.3f} {r['gap']:>+8.3f} {r['iqr_overlap_pct']:>7.1f}% {pval_str:>10} {r['verdict']:>8}")
