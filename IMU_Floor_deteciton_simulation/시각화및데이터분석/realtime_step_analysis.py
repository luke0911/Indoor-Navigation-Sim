#!/usr/bin/env python3
"""
실시간 스텝 단위 분석 (Real-time Per-Step Analysis)

핵심 원칙:
  - 모든 피처는 **한 스텝 구간의 샘플만** 사용하여 계산 (전역 통계 없음)
  - 가속도 데이터가 실시간으로 들어온다고 가정
  - 스텝[n] 감지 시점에서 스텝[n-1]~스텝[n] 구간의 pureY, pureZ 샘플로 피처 계산

데이터셋: 일반보행, 살살걸음, 거칠게걸음 (3개만)

분석:
  1. 스텝별 피처 계산 (정확히 어떤 데이터로 계산하는지 명시)
  2. 단일 임계값 판별 테스트 (threshold sweep → precision/recall/F1)
  3. 스텝 내 궤적 형태 일관성 분석 (사용자 관찰 검증)
  4. 2-피처 조합 테스트
"""
import os, math, csv

DATA_DIR = "/Users/idohun/WorkSpace/KU-LAB/IMU_Floor_deteciton_simulation/data/output"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "일반보행":   "batch_Floor_sim_data_일반보행.tsv",
    "살살걸음":   "batch_Floor_sim_data_살살걸음.tsv",
    "거칠게걸음": "batch_Floor_sim_data_거칠게 걸음.tsv",
}

# ── 기본 통계 함수 (순수 Python) ──
def mean(v): return sum(v)/max(len(v),1)
def median_val(v):
    s = sorted(v); n = len(s)
    if n == 0: return 0
    return s[n//2] if n % 2 else (s[n//2-1]+s[n//2])/2
def std_val(v):
    m = mean(v)
    return math.sqrt(sum((x-m)**2 for x in v)/max(len(v),1))
def percentile(v, p):
    s = sorted(v); n = len(s)
    if n == 0: return 0
    k = (n-1)*p/100.0; f = int(k); c = min(f+1,n-1)
    return s[f]+(k-f)*(s[c]-s[f])
def iqr_overlap(a, b):
    if not a or not b: return float('nan')
    a25,a75 = percentile(a,25),percentile(a,75)
    b25,b75 = percentile(b,25),percentile(b,75)
    lo,hi = max(a25,b25),min(a75,b75)
    if lo >= hi: return 0.0
    span = max(a75,b75)-min(a25,b25)
    return (hi-lo)/span*100 if span > 0 else 0.0
def mann_whitney_p(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float('nan')
    combined = [(v,'a') for v in a]+[(v,'b') for v in b]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0]*len(combined); i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0]==combined[i][0]: j += 1
        ar = (i+1+j)/2.0
        for k in range(i,j): ranks[k] = ar
        i = j
    Ra = sum(ranks[i] for i in range(len(combined)) if combined[i][1]=='a')
    U1 = Ra - na*(na+1)/2; mu = na*nb/2.0
    sigma = math.sqrt(na*nb*(na+nb+1)/12.0)
    if sigma == 0: return float('nan')
    z = (min(U1,na*nb-U1)-mu)/sigma
    return 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))


# ══════════════════════════════════════════════════════════
#  STEP 1: 데이터 로딩 → 스텝 구간 분리
# ══════════════════════════════════════════════════════════
def load_step_segments(filepath):
    """
    batch TSV 로딩 → 스텝 구간별 분리

    스텝 구간 정의:
      step[n-1] 감지 시점 ~ step[n] 감지 시점 사이의 모든 샘플
      → 실시간: step[n] 감지 시 이전 구간 데이터가 확정됨
    """
    samples = []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            samples.append(row)

    # 스텝 경계 찾기
    step_boundaries = []
    for i, s in enumerate(samples):
        if s.get('isStep') == '1':
            step_boundaries.append((i, int(s['stepIdx']), s['floorGT']))

    # 각 스텝 구간: step[n]~step[n+1] 사이 샘플
    segments = []
    for si in range(len(step_boundaries) - 1):
        start_idx = step_boundaries[si][0]
        end_idx = step_boundaries[si+1][0]  # 다음 스텝 시점까지
        step_num = step_boundaries[si][1]
        gt = step_boundaries[si][2]

        seg_pY = []
        seg_pZ = []
        seg_gZ = []  # globalZ도 수집
        for j in range(start_idx, end_idx):
            try:
                seg_pY.append(float(samples[j]['pureY']))
                seg_pZ.append(float(samples[j]['pureZ']))
                seg_gZ.append(float(samples[j]['globalZ']))
            except:
                pass

        if len(seg_pY) < 3:
            continue

        segments.append({
            'step': step_num,
            'gt': gt,
            'pY': seg_pY,
            'pZ': seg_pZ,
            'gZ': seg_gZ,
            'n_samples': len(seg_pY),
        })

    return segments


# ══════════════════════════════════════════════════════════
#  STEP 2: 스텝별 피처 계산 (각 피처의 입력 데이터 명시)
# ══════════════════════════════════════════════════════════
"""
모든 피처는 아래 입력만 사용:
  - pY[0..N]: 이 스텝 구간의 pureY 샘플 N개
  - pZ[0..N]: 이 스텝 구간의 pureZ 샘플 N개
  - gZ[0..N]: 이 스텝 구간의 globalZ 샘플 N개

누적 상태(baseline, EWMA, 이전 스텝 정보) 없음.
한 스텝 데이터만으로 즉시 계산 가능 → 실시간 적용 가능
"""

def compute_step_features(seg):
    pY, pZ, gZ = seg['pY'], seg['pZ'], seg['gZ']
    n = len(pY)

    # ── 피처 1: zFrac (Z축 에너지 비율) ──
    # 입력: 이 스텝의 pureZ[0..N], pureY[0..N]
    # 계산: var(pZ) / (var(pZ) + var(pY))
    # 의미: 1에 가까울수록 Z축 변동이 지배적 → 계단 가능성
    varZ = sum((z - mean(pZ))**2 for z in pZ) / n
    varY = sum((y - mean(pY))**2 for y in pY) / n
    zFrac = varZ / (varZ + varY + 1e-9)

    # ── 피처 2: aspect (종횡비) ──
    # 입력: 이 스텝의 pureZ[0..N], pureY[0..N]
    # 계산: rangeZ / rangeY
    # 의미: >1이면 수직 방향 움직임이 큼
    rangeY = max(pY) - min(pY)
    rangeZ = max(pZ) - min(pZ)
    aspect = rangeZ / (rangeY + 1e-9)

    # ── 피처 3: rangeY (수평 진폭) ──
    # 입력: 이 스텝의 pureY[0..N]
    # 계산: max(pY) - min(pY)
    # 의미: 계단에서는 수평 진폭이 줄어드는 경향
    # (rangeY 자체)

    # ── 피처 4: rangeZ (수직 진폭) ──
    # 입력: 이 스텝의 pureZ[0..N]
    # 계산: max(pZ) - min(pZ)
    # (rangeZ 자체)

    # ── 피처 5: start_pZ (시작점 Z) ──
    # 입력: 이 스텝 첫 샘플의 pureZ
    # 의미: 계단 구간 시작점이 특정 Z값에 군집하는지
    start_pZ = pZ[0]
    start_pY = pY[0]

    # ── 피처 6: meanZ_offset (구간 내 Z 무게중심 - 시작점) ──
    # 입력: 이 스텝의 pureZ[0..N]
    # 계산: mean(pZ) - pZ[0]
    # 의미: 스텝 내에서 Z가 시작점 대비 어디로 이동했나
    meanZ_offset = mean(pZ) - start_pZ

    # ── 피처 7: maxZ_offset (구간 내 Z 최대 이탈) ──
    # 입력: 이 스텝의 pureZ[0..N]
    maxZ_offset = max(pZ) - start_pZ

    # ── 피처 8: rmsZ (구간 내 globalZ RMS) ──
    # 입력: 이 스텝의 globalZ[0..N]
    # 계산: sqrt(mean(gZ^2))
    # 의미: 중력 포함 Z축 RMS — 계단은 더 큰 충격
    rmsZ = math.sqrt(sum(z*z for z in gZ) / n) if gZ else 0

    # ── 피처 9: zPeakRatio (Z 피크 대 Y 피크 비) ──
    # 입력: 이 스텝의 pureZ[0..N], pureY[0..N]
    # 계산: max(|pZ|) / (max(|pY|) + max(|pZ|))
    maxAbsZ = max(abs(z) for z in pZ)
    maxAbsY = max(abs(y) for y in pY)
    zPeakRatio = maxAbsZ / (maxAbsY + maxAbsZ + 1e-9)

    # ── 피처 10: trajectory_compactness (궤적 밀집도) ──
    # 입력: 이 스텝의 pureY/pureZ
    # 계산: 궤적 경로 길이 / 궤적 범위 대각선
    #   밀집도 높음 → 좁은 영역에서 많이 움직임
    path_len = sum(math.sqrt((pY[i+1]-pY[i])**2+(pZ[i+1]-pZ[i])**2) for i in range(n-1))
    diag = math.sqrt(rangeY**2 + rangeZ**2) + 1e-9
    compactness = path_len / diag

    # ── 피처 11: z_dominance_ratio (구간 후반 Z 지배) ──
    # 입력: 이 스텝의 pureZ[0..N]
    # 계산: 후반부 abs(pZ) 평균 / 전반부 abs(pZ) 평균
    # 의미: 계단에서 후반에 Z가 커지는 패턴
    half = n // 2
    if half > 0:
        first_half_z = mean([abs(z) for z in pZ[:half]])
        second_half_z = mean([abs(z) for z in pZ[half:]])
        z_dom = second_half_z / (first_half_z + 1e-9)
    else:
        z_dom = 1.0

    return {
        'zFrac': zFrac,
        'aspect': aspect,
        'rangeY': rangeY,
        'rangeZ': rangeZ,
        'start_pZ': start_pZ,
        'start_pY': start_pY,
        'meanZ_offset': meanZ_offset,
        'maxZ_offset': maxZ_offset,
        'rmsZ': rmsZ,
        'zPeakRatio': zPeakRatio,
        'compactness': compactness,
        'z_dominance': z_dom,
        'n_samples': seg['n_samples'],
    }


# ══════════════════════════════════════════════════════════
#  STEP 3: 단일 임계값 판별 테스트
# ══════════════════════════════════════════════════════════
def threshold_sweep(gnd_vals, stair_vals, direction='gt', n_steps=200):
    """
    단일 피처 임계값 sweep → 최적 F1 찾기

    direction: 'gt' = 값이 클수록 stair, 'lt' = 값이 작을수록 stair
    반환: (best_threshold, best_precision, best_recall, best_f1)
    """
    all_vals = gnd_vals + stair_vals
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return (lo, 0, 0, 0)

    best = (0, 0, 0, 0)  # threshold, prec, rec, f1
    for i in range(n_steps + 1):
        t = lo + (hi - lo) * i / n_steps
        if direction == 'gt':
            tp = sum(1 for v in stair_vals if v > t)
            fp = sum(1 for v in gnd_vals if v > t)
            fn = sum(1 for v in stair_vals if v <= t)
        else:
            tp = sum(1 for v in stair_vals if v < t)
            fp = sum(1 for v in gnd_vals if v < t)
            fn = sum(1 for v in stair_vals if v >= t)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best[3]:
            best = (t, prec, rec, f1)

    return best


# ══════════════════════════════════════════════════════════
#  STEP 4: 궤적 형태 일관성 분석 (사용자 관찰 검증)
# ══════════════════════════════════════════════════════════
def analyze_trajectory_shape(segments):
    """
    사용자 관찰: "up 스텝일 때 시작점이 일정 구간에 멈추거나
                 한 스텝 구간 데이터가 일정한 모양으로 변화"

    분석:
    1. ground vs stair의 궤적 형태 분산 비교
       (stair가 더 일정한 형태라면 분산이 작아야 함)
    2. 스텝 내 pZ 패턴 유사성 (DTW 대신 간단한 형태 기술자)
    3. 시작점 군집 밀도 비교
    """
    gnd_segs = [s for s in segments if s['gt'] == 'ground']
    stair_segs = [s for s in segments if s['gt'] in ('up', 'down', 'stair')]

    if not gnd_segs or not stair_segs:
        return None

    result = {}

    # ── 시작점 군집 밀도 (std of start points) ──
    gnd_starts_pZ = [s['pZ'][0] for s in gnd_segs]
    stair_starts_pZ = [s['pZ'][0] for s in stair_segs]
    gnd_starts_pY = [s['pY'][0] for s in gnd_segs]
    stair_starts_pY = [s['pY'][0] for s in stair_segs]

    result['start_cluster'] = {
        'gnd_pZ_std': std_val(gnd_starts_pZ),
        'stair_pZ_std': std_val(stair_starts_pZ),
        'gnd_pY_std': std_val(gnd_starts_pY),
        'stair_pY_std': std_val(stair_starts_pY),
        'gnd_pZ_range': max(gnd_starts_pZ) - min(gnd_starts_pZ),
        'stair_pZ_range': max(stair_starts_pZ) - min(stair_starts_pZ),
    }

    # ── 궤적 형태 정규화 후 유사성 ──
    # 각 스텝의 pZ 궤적을 [0,1] 시간축으로 정규화 (10 포인트 리샘플)
    def normalize_trajectory(pZ_list, n_points=10):
        """1 스텝의 pZ 시계열 → n_points로 리샘플, min-max 정규화"""
        n = len(pZ_list)
        if n < 2:
            return [0.0] * n_points
        # 시간축 리샘플
        resampled = []
        for i in range(n_points):
            t = i / (n_points - 1) * (n - 1)
            idx = int(t)
            frac = t - idx
            if idx >= n - 1:
                resampled.append(pZ_list[-1])
            else:
                resampled.append(pZ_list[idx] * (1-frac) + pZ_list[idx+1] * frac)
        # min-max 정규화
        lo, hi = min(resampled), max(resampled)
        if hi - lo < 1e-9:
            return [0.5] * n_points
        return [(v - lo) / (hi - lo) for v in resampled]

    gnd_norms = [normalize_trajectory(s['pZ']) for s in gnd_segs]
    stair_norms = [normalize_trajectory(s['pZ']) for s in stair_segs]

    # 그룹 내 평균 궤적
    gnd_avg = [mean([g[i] for g in gnd_norms]) for i in range(10)]
    stair_avg = [mean([g[i] for g in stair_norms]) for i in range(10)]

    # 그룹 내 분산 (형태 일관성 = 분산이 작을수록 일관적)
    gnd_shape_var = mean([mean([(g[i] - gnd_avg[i])**2 for i in range(10)]) for g in gnd_norms])
    stair_shape_var = mean([mean([(s[i] - stair_avg[i])**2 for i in range(10)]) for s in stair_norms])

    result['shape_consistency'] = {
        'gnd_shape_var': gnd_shape_var,
        'stair_shape_var': stair_shape_var,
        'stair_more_consistent': stair_shape_var < gnd_shape_var,
        'gnd_avg_pattern': gnd_avg,
        'stair_avg_pattern': stair_avg,
    }

    # ── pY 궤적도 동일 분석 ──
    gnd_norms_pY = [normalize_trajectory(s['pY']) for s in gnd_segs]
    stair_norms_pY = [normalize_trajectory(s['pY']) for s in stair_segs]
    gnd_avg_pY = [mean([g[i] for g in gnd_norms_pY]) for i in range(10)]
    stair_avg_pY = [mean([g[i] for g in stair_norms_pY]) for i in range(10)]
    gnd_pY_var = mean([mean([(g[i] - gnd_avg_pY[i])**2 for i in range(10)]) for g in gnd_norms_pY])
    stair_pY_var = mean([mean([(s[i] - stair_avg_pY[i])**2 for i in range(10)]) for s in stair_norms_pY])

    result['shape_consistency_pY'] = {
        'gnd_pY_var': gnd_pY_var,
        'stair_pY_var': stair_pY_var,
        'stair_more_consistent': stair_pY_var < gnd_pY_var,
    }

    return result


# ══════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════
print("=" * 90)
print("  실시간 스텝 단위 분석 (Per-Step Real-Time Feature Analysis)")
print("  모든 피처: 한 스텝 구간의 pureY/pureZ/globalZ 샘플만으로 계산")
print("  누적 상태/전역 통계 없음 → 실시간 적용 가능")
print("=" * 90)

all_rows = []

for ds_name, ds_file in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path):
        print(f"\n  {ds_name}: FILE NOT FOUND ({path})"); continue

    segments = load_step_segments(path)
    gnd_segs = [s for s in segments if s['gt'] == 'ground']
    stair_segs = [s for s in segments if s['gt'] in ('up', 'down', 'stair')]

    print(f"\n\n{'='*90}")
    print(f"  {ds_name}: {len(segments)} steps (ground={len(gnd_segs)}, stair={len(stair_segs)})")
    print(f"{'='*90}")

    # 피처 계산
    for seg in segments:
        f = compute_step_features(seg)
        f['gt'] = seg['gt']
        f['step'] = seg['step']
        f['dataset'] = ds_name
        all_rows.append(f)

    gnd_feats = [r for r in all_rows if r['dataset']==ds_name and r['gt']=='ground']
    stair_feats = [r for r in all_rows if r['dataset']==ds_name and r['gt'] in ('up','down','stair')]

    # ── 피처별 통계 비교 ──
    FEATURES = [
        ('zFrac',        'gt',  'Z축 에너지 비율 = var(pZ)/(var(pZ)+var(pY))'),
        ('aspect',       'gt',  '종횡비 = rangeZ/rangeY'),
        ('rangeY',       'lt',  '수평 진폭 = max(pY)-min(pY)'),
        ('rangeZ',       'gt',  '수직 진폭 = max(pZ)-min(pZ)'),
        ('rmsZ',         'gt',  'globalZ RMS = sqrt(mean(gZ^2))'),
        ('zPeakRatio',   'gt',  'Z피크비 = max|pZ|/(max|pY|+max|pZ|)'),
        ('meanZ_offset', 'gt',  'Z중심이탈 = mean(pZ)-pZ[0]'),
        ('compactness',  'gt',  '궤적밀집도 = path_len/대각선'),
        ('z_dominance',  'gt',  '후반Z지배 = 후반|pZ|평균/전반|pZ|평균'),
    ]

    print(f"\n  {'피처':<18} {'입력데이터':<32} {'Gnd_med':>8} {'Str_med':>8} {'Gap':>8} {'IQR겹침':>7} {'p':>10} {'판정'}")
    print(f"  {'─'*105}")

    for fk, direction, desc in FEATURES:
        g = [f[fk] for f in gnd_feats]
        u = [f[fk] for f in stair_feats]
        if not g or not u: continue
        mg, mu = median_val(g), median_val(u)
        gap = mu - mg
        ovlp = iqr_overlap(g, u)
        p = mann_whitney_p(g, u)
        ps = f"{p:.1e}" if not math.isnan(p) else "N/A"
        tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
        print(f"  {fk:<18} {desc:<32} {mg:>8.3f} {mu:>8.3f} {gap:>+8.3f} {ovlp:>6.1f}% {ps:>10}  {tag}")

    # ── 단일 임계값 판별 테스트 ──
    print(f"\n  단일 임계값 판별 (threshold sweep → 최적 F1):")
    print(f"  {'피처':<18} {'방향':<6} {'임계값':>8} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'─'*60}")
    for fk, direction, desc in FEATURES:
        g = [f[fk] for f in gnd_feats]
        u = [f[fk] for f in stair_feats]
        if not g or not u: continue
        t, prec, rec, f1 = threshold_sweep(g, u, direction)
        dir_sym = ">" if direction == 'gt' else "<"
        print(f"  {fk:<18} {dir_sym}{t:>7.3f} {' ':>1} {prec:>5.2f}  {rec:>5.2f}  {f1:>5.2f}")

    # ── 궤적 형태 분석 ──
    shape = analyze_trajectory_shape(segments)
    if shape:
        sc = shape['start_cluster']
        print(f"\n  궤적 형태 분석:")
        print(f"    시작점 군집 밀도 (std가 작을수록 군집):")
        print(f"      pZ: Ground std={sc['gnd_pZ_std']:.3f}  Stair std={sc['stair_pZ_std']:.3f}  (stair {'더 군집' if sc['stair_pZ_std'] < sc['gnd_pZ_std'] else '덜 군집'})")
        print(f"      pY: Ground std={sc['gnd_pY_std']:.3f}  Stair std={sc['stair_pY_std']:.3f}  (stair {'더 군집' if sc['stair_pY_std'] < sc['gnd_pY_std'] else '덜 군집'})")
        print(f"      pZ range: Ground {sc['gnd_pZ_range']:.3f}  Stair {sc['stair_pZ_range']:.3f}")

        sh = shape['shape_consistency']
        print(f"\n    pZ 궤적 형태 일관성 (정규화 후 분산, 작을수록 일관적):")
        print(f"      Ground 형태분산: {sh['gnd_shape_var']:.4f}")
        print(f"      Stair  형태분산: {sh['stair_shape_var']:.4f}")
        print(f"      → Stair가 {'더 일관적' if sh['stair_more_consistent'] else 'ground보다 덜 일관적'}")

        # 평균 궤적 패턴 표시
        print(f"\n    정규화 평균 pZ 궤적 (10-point, 0.0=min, 1.0=max):")
        print(f"      Ground: {' → '.join(f'{v:.2f}' for v in sh['gnd_avg_pattern'])}")
        print(f"      Stair:  {' → '.join(f'{v:.2f}' for v in sh['stair_avg_pattern'])}")

        shY = shape['shape_consistency_pY']
        print(f"\n    pY 궤적 형태 일관성:")
        print(f"      Ground 형태분산: {shY['gnd_pY_var']:.4f}")
        print(f"      Stair  형태분산: {shY['stair_pY_var']:.4f}")
        print(f"      → Stair가 {'더 일관적' if shY['stair_more_consistent'] else 'ground보다 덜 일관적'}")


# ══════════════════════════════════════════════════════════
#  COMBINED 분석 (3개 합산)
# ══════════════════════════════════════════════════════════
print(f"\n\n{'='*90}")
print(f"  COMBINED (3개 데이터셋 합산)")
print(f"{'='*90}")

gnd_all = [r for r in all_rows if r['gt']=='ground']
stair_all = [r for r in all_rows if r['gt'] in ('up','down','stair')]
print(f"  총 {len(gnd_all)} ground + {len(stair_all)} stair steps")

FEATURES = [
    ('zFrac',        'gt',  'Z축 에너지 비율 = var(pZ)/(var(pZ)+var(pY))'),
    ('aspect',       'gt',  '종횡비 = rangeZ/rangeY'),
    ('rangeY',       'lt',  '수평 진폭 = max(pY)-min(pY)'),
    ('rangeZ',       'gt',  '수직 진폭 = max(pZ)-min(pZ)'),
    ('rmsZ',         'gt',  'globalZ RMS = sqrt(mean(gZ^2))'),
    ('zPeakRatio',   'gt',  'Z피크비 = max|pZ|/(max|pY|+max|pZ|)'),
    ('meanZ_offset', 'gt',  'Z중심이탈 = mean(pZ)-pZ[0]'),
    ('compactness',  'gt',  '궤적밀집도 = path_len/대각선'),
    ('z_dominance',  'gt',  '후반Z지배 = 후반|pZ|평균/전반|pZ|평균'),
]

print(f"\n  {'피처':<18} {'Gnd_med':>8} {'Str_med':>8} {'Gap':>8} {'IQR겹침':>7} {'p':>10} {'판정'}")
print(f"  {'─'*70}")
for fk, direction, desc in FEATURES:
    g = [f[fk] for f in gnd_all]
    u = [f[fk] for f in stair_all]
    mg, mu = median_val(g), median_val(u)
    gap = mu - mg
    ovlp = iqr_overlap(g, u)
    p = mann_whitney_p(g, u)
    ps = f"{p:.1e}" if not math.isnan(p) else "N/A"
    tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
    print(f"  {fk:<18} {mg:>8.3f} {mu:>8.3f} {gap:>+8.3f} {ovlp:>6.1f}% {ps:>10}  {tag}")

# ── COMBINED 임계값 테스트 ──
print(f"\n  COMBINED 단일 임계값 판별:")
print(f"  {'피처':<18} {'방향':<6} {'임계값':>8} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print(f"  {'─'*60}")
for fk, direction, desc in FEATURES:
    g = [f[fk] for f in gnd_all]
    u = [f[fk] for f in stair_all]
    t, prec, rec, f1 = threshold_sweep(g, u, direction)
    dir_sym = ">" if direction == 'gt' else "<"
    print(f"  {fk:<18} {dir_sym}{t:>7.3f} {' ':>1} {prec:>5.2f}  {rec:>5.2f}  {f1:>5.2f}")


# ══════════════════════════════════════════════════════════
#  STEP 5: 2-피처 조합 테스트 (AND 조건)
# ══════════════════════════════════════════════════════════
print(f"\n\n{'='*90}")
print(f"  2-피처 조합 테스트 (COMBINED, AND 조건)")
print(f"{'='*90}")

# 상위 피처 조합 테스트
TOP_FEATS = ['zFrac', 'aspect', 'rmsZ', 'rangeY', 'zPeakRatio']
combos_result = []

for i in range(len(TOP_FEATS)):
    for j in range(i+1, len(TOP_FEATS)):
        fk1, fk2 = TOP_FEATS[i], TOP_FEATS[j]
        # 각 피처 최적 임계값
        dir1 = next(d for f,d,_ in FEATURES if f==fk1)
        dir2 = next(d for f,d,_ in FEATURES if f==fk2)
        g1 = [f[fk1] for f in gnd_all]
        u1 = [f[fk1] for f in stair_all]
        g2 = [f[fk2] for f in gnd_all]
        u2 = [f[fk2] for f in stair_all]

        # 2D sweep (각 피처 50 스텝)
        best_combo = (0, 0, 0, 0, 0, 0)  # t1, t2, prec, rec, f1
        all_v1 = g1 + u1
        all_v2 = g2 + u2
        lo1, hi1 = min(all_v1), max(all_v1)
        lo2, hi2 = min(all_v2), max(all_v2)

        for s1 in range(51):
            t1 = lo1 + (hi1 - lo1) * s1 / 50
            for s2 in range(51):
                t2 = lo2 + (hi2 - lo2) * s2 / 50

                tp, fp, fn = 0, 0, 0
                for r in stair_all:
                    c1 = (r[fk1] > t1) if dir1=='gt' else (r[fk1] < t1)
                    c2 = (r[fk2] > t2) if dir2=='gt' else (r[fk2] < t2)
                    if c1 and c2: tp += 1
                    else: fn += 1
                for r in gnd_all:
                    c1 = (r[fk1] > t1) if dir1=='gt' else (r[fk1] < t1)
                    c2 = (r[fk2] > t2) if dir2=='gt' else (r[fk2] < t2)
                    if c1 and c2: fp += 1

                prec = tp/(tp+fp) if (tp+fp)>0 else 0
                rec = tp/(tp+fn) if (tp+fn)>0 else 0
                f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
                if f1 > best_combo[4]:
                    best_combo = (t1, t2, prec, rec, f1)

        t1, t2, prec, rec, f1 = best_combo
        d1s = ">" if dir1=='gt' else "<"
        d2s = ">" if dir2=='gt' else "<"
        combos_result.append((fk1, fk2, t1, t2, d1s, d2s, prec, rec, f1))
        print(f"  {fk1}{d1s}{t1:.3f} AND {fk2}{d2s}{t2:.3f}  → Prec={prec:.2f} Rec={rec:.2f} F1={f1:.2f}")

# 정렬해서 Top 5 표시
combos_result.sort(key=lambda x: -x[-1])
print(f"\n  Top 5 조합:")
for rank, (fk1, fk2, t1, t2, d1s, d2s, prec, rec, f1) in enumerate(combos_result[:5], 1):
    print(f"    #{rank}: {fk1}{d1s}{t1:.3f} AND {fk2}{d2s}{t2:.3f}  → Prec={prec:.2f} Rec={rec:.2f} F1={f1:.2f}")


# ══════════════════════════════════════════════════════════
#  COMBINED 궤적 형태 분석
# ══════════════════════════════════════════════════════════
print(f"\n\n{'='*90}")
print(f"  COMBINED 궤적 형태 분석")
print(f"{'='*90}")

all_segs = []
for ds_name, ds_file in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_file)
    if os.path.exists(path):
        all_segs.extend(load_step_segments(path))

shape = analyze_trajectory_shape(all_segs)
if shape:
    sc = shape['start_cluster']
    print(f"\n  시작점 군집 밀도:")
    print(f"    pZ: Ground std={sc['gnd_pZ_std']:.3f} range={sc['gnd_pZ_range']:.3f}")
    print(f"    pZ: Stair  std={sc['stair_pZ_std']:.3f} range={sc['stair_pZ_range']:.3f}")
    print(f"    → Stair 시작점이 {'더 군집' if sc['stair_pZ_std'] < sc['gnd_pZ_std'] else '덜 군집'} (pZ)")
    print(f"    pY: Ground std={sc['gnd_pY_std']:.3f}")
    print(f"    pY: Stair  std={sc['stair_pY_std']:.3f}")
    print(f"    → Stair 시작점이 {'더 군집' if sc['stair_pY_std'] < sc['gnd_pY_std'] else '덜 군집'} (pY)")

    sh = shape['shape_consistency']
    print(f"\n  pZ 궤적 형태 일관성 (정규화 후 분산):")
    print(f"    Ground: {sh['gnd_shape_var']:.4f}")
    print(f"    Stair:  {sh['stair_shape_var']:.4f}")
    print(f"    → Stair 궤적이 {'더 일관적 ✅' if sh['stair_more_consistent'] else 'ground보다 덜 일관적 ❌'}")

    print(f"\n  정규화 평균 pZ 궤적 패턴:")
    print(f"    Ground: {' → '.join(f'{v:.2f}' for v in sh['gnd_avg_pattern'])}")
    print(f"    Stair:  {' → '.join(f'{v:.2f}' for v in sh['stair_avg_pattern'])}")

    shY = shape['shape_consistency_pY']
    print(f"\n  pY 궤적 형태 일관성:")
    print(f"    Ground: {shY['gnd_pY_var']:.4f}")
    print(f"    Stair:  {shY['stair_pY_var']:.4f}")
    print(f"    → Stair 궤적이 {'더 일관적 ✅' if shY['stair_more_consistent'] else 'ground보다 덜 일관적 ❌'}")


# ── TSV 저장 ──
tsv_path = os.path.join(OUT_DIR, "realtime_step_features.tsv")
keys = ['dataset','step','gt',
        'zFrac','aspect','rangeY','rangeZ','rmsZ','zPeakRatio',
        'meanZ_offset','maxZ_offset','compactness','z_dominance',
        'start_pY','start_pZ','n_samples']
with open(tsv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys, delimiter='\t', extrasaction='ignore')
    writer.writeheader()
    for r in all_rows:
        writer.writerow(r)
print(f"\n\n스텝별 피처 TSV 저장: {tsv_path}")
print("(모든 피처는 해당 스텝 구간의 pureY/pureZ/globalZ 샘플만으로 계산됨)")
