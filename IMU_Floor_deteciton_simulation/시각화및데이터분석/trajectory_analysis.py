#!/usr/bin/env python3
"""
pureY vs pureZ 4사분면 궤적 분석 (3개 데이터셋: 일반보행, 살살걸음, 거칠게걸음)

분석 항목:
1. 스텝 시작점 (pY, pZ) 위치 분포 — ground vs stair
2. 스텝 구간 내 궤적 형태 특징 (방향, 면적, 종횡비)
3. 스텝 구간 내 pZ range, pY range, 진행 방향
4. 연속 스텝 간 시작점 이동 패턴
"""
import os, math, csv

DATA_DIR = "/Users/idohun/WorkSpace/KU-LAB/IMU_Floor_deteciton_simulation/data/output"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "일반보행":   "batch_Floor_sim_data_일반보행.tsv",
    "살살걸음":   "batch_Floor_sim_data_살살걸음.tsv",
    "거칠게걸음": "batch_Floor_sim_data_거칠게 걸음.tsv",
}

def mean(v): return sum(v)/max(len(v),1)
def median(v):
    s = sorted(v); n = len(s)
    if n == 0: return 0
    return s[n//2] if n % 2 else (s[n//2-1]+s[n//2])/2
def std(v):
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
    return (hi-lo)/(max(a75,b75)-min(a25,b25))*100
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


# ── 데이터 로딩: 매 샘플 + 스텝 구간 분리 ──
def load_batch(filepath):
    """batch TSV 로딩 → 스텝 구간별로 분리"""
    samples = []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            samples.append(row)

    # 스텝 경계 찾기
    step_boundaries = []  # (sample_index, stepIdx, gt_label)
    for i, s in enumerate(samples):
        if s.get('isStep') == '1':
            step_boundaries.append((i, int(s['stepIdx']), s['floorGT']))

    # 스텝 구간: step[n] ~ step[n+1]-1 사이의 모든 샘플
    segments = []
    for si in range(len(step_boundaries)):
        start_idx = step_boundaries[si][0]
        end_idx = step_boundaries[si+1][0] if si+1 < len(step_boundaries) else len(samples)-1
        step_num = step_boundaries[si][1]
        gt = step_boundaries[si][2]

        seg_pY = []
        seg_pZ = []
        for j in range(start_idx, end_idx+1):
            try:
                seg_pY.append(float(samples[j]['pureY']))
                seg_pZ.append(float(samples[j]['pureZ']))
            except:
                pass

        if len(seg_pY) < 3:
            continue

        segments.append({
            'step': step_num,
            'gt': gt,
            'pY': seg_pY,
            'pZ': seg_pZ,
            'start_pY': seg_pY[0],
            'start_pZ': seg_pZ[0],
            'n_samples': len(seg_pY),
        })

    return segments


# ── 스텝 구간 피처 계산 ──
def compute_segment_features(seg):
    pY, pZ = seg['pY'], seg['pZ']
    n = len(pY)

    # 시작점
    s_pY, s_pZ = pY[0], pZ[0]

    # 범위
    rangeY = max(pY) - min(pY)
    rangeZ = max(pZ) - min(pZ)
    aspect = rangeZ / (rangeY + 1e-6)  # 종횡비 (>1이면 Z방향 우세)

    # 무게중심
    cY, cZ = mean(pY), mean(pZ)

    # 시작점 → 무게중심 방향 (angle from Y axis, 0=오른쪽, 90=위)
    dy = cY - s_pY
    dz = cZ - s_pZ
    angle_deg = math.degrees(math.atan2(dz, dy))  # -180~180

    # 궤적 둘러싼 면적 (shoelace approximation)
    area = 0
    for i in range(n-1):
        area += pY[i]*pZ[i+1] - pY[i+1]*pZ[i]
    area = abs(area) / 2.0

    # 궤적 총 길이 (path length)
    path_len = sum(math.sqrt((pY[i+1]-pY[i])**2+(pZ[i+1]-pZ[i])**2) for i in range(n-1))

    # pZ의 변동 대 pY의 변동 비율
    varZ = sum((z - mean(pZ))**2 for z in pZ) / n
    varY = sum((y - mean(pY))**2 for y in pY) / n
    zFrac = varZ / (varZ + varY + 1e-6)  # 0~1, >0.5이면 Z 우세

    # pZ 평균 레벨 (시작 기준)
    meanZ_offset = mean(pZ) - s_pZ
    maxZ_offset = max(pZ) - s_pZ

    # 시작점이 어느 사분면에 있는지 (pY=X, pZ=Y 기준)
    quadrant = 0
    if s_pY >= 0 and s_pZ >= 0: quadrant = 1
    elif s_pY < 0 and s_pZ >= 0: quadrant = 2
    elif s_pY < 0 and s_pZ < 0: quadrant = 3
    else: quadrant = 4

    return {
        'start_pY': s_pY, 'start_pZ': s_pZ,
        'rangeY': rangeY, 'rangeZ': rangeZ,
        'aspect': aspect,
        'center_pY': cY, 'center_pZ': cZ,
        'angle_deg': angle_deg,
        'area': area,
        'path_len': path_len,
        'zFrac': zFrac,
        'meanZ_offset': meanZ_offset,
        'maxZ_offset': maxZ_offset,
        'quadrant': quadrant,
        'n_samples': seg['n_samples'],
    }


# ══════════════════════════════════════════════════════════════
#  분석 실행
# ══════════════════════════════════════════════════════════════
all_features_rows = []  # for TSV export

for ds_name, ds_file in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path):
        print(f"  {ds_name}: NOT FOUND"); continue

    segments = load_batch(path)
    print(f"\n{'='*85}")
    print(f"  {ds_name}  ({len(segments)} step segments)")
    print(f"{'='*85}")

    gnd_feats = []
    up_feats = []
    for seg in segments:
        f = compute_segment_features(seg)
        f['gt'] = seg['gt']
        f['step'] = seg['step']
        f['dataset'] = ds_name
        all_features_rows.append(f)
        if seg['gt'] == 'ground':
            gnd_feats.append(f)
        elif seg['gt'] in ('up', 'down', 'stair'):
            up_feats.append(f)

    if not gnd_feats or not up_feats:
        print("  No stair segments!"); continue

    # ── 피처별 비교 ──
    FEAT_LIST = [
        ('start_pZ',     '스텝 시작점 pZ (수직 위치)'),
        ('start_pY',     '스텝 시작점 pY (수평 위치)'),
        ('rangeZ',       '구간 내 pZ 범위'),
        ('rangeY',       '구간 내 pY 범위'),
        ('aspect',       '종횡비 (rangeZ/rangeY)'),
        ('zFrac',        '구간 내 var비율 (Z/(Z+Y))'),
        ('area',         '궤적 면적 (shoelace)'),
        ('path_len',     '궤적 경로 길이'),
        ('meanZ_offset', '구간 pZ 무게중심 offset'),
        ('maxZ_offset',  '구간 pZ max offset'),
        ('angle_deg',    '시작→무게중심 방향(deg)'),
    ]

    print(f"\n  {'Feature':<20} {'Gnd_med':>8} {'Up_med':>8} {'Gap':>8} {'IQR겹침':>7} {'p':>10} {'판정'}")
    print(f"  {'─'*75}")
    for fk, fname in FEAT_LIST:
        g = [f[fk] for f in gnd_feats]
        u = [f[fk] for f in up_feats]
        mg, mu = median(g), median(u)
        gap = mu - mg
        ovlp = iqr_overlap(g, u)
        p = mann_whitney_p(g, u)
        ps = f"{p:.1e}" if not math.isnan(p) else "N/A"
        tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
        print(f"  {fk:<20} {mg:>8.3f} {mu:>8.3f} {gap:>+8.3f} {ovlp:>6.1f}% {ps:>10}  {tag} {fname}")

    # ── 사분면 분포 ──
    print(f"\n  사분면 분포 (pY=X축, pZ=Y축 기준 시작점):")
    for q in [1,2,3,4]:
        gc = sum(1 for f in gnd_feats if f['quadrant']==q)
        uc = sum(1 for f in up_feats if f['quadrant']==q)
        gp = gc/len(gnd_feats)*100
        up_ = uc/len(up_feats)*100
        print(f"    Q{q}: Ground {gc:>3} ({gp:>5.1f}%)   Stair {uc:>3} ({up_:>5.1f}%)")

    # ── 연속 스텝 시작점 이동 분석 ──
    print(f"\n  연속 스텝 시작점 이동 (step[n] → step[n+1]):")
    gnd_dY, gnd_dZ, up_dY, up_dZ = [], [], [], []
    gnd_dist, up_dist = [], []
    for i in range(1, len(segments)):
        if segments[i-1]['gt'] != segments[i]['gt']:
            continue  # 전환 구간 스킵
        dy = segments[i]['start_pY'] - segments[i-1]['start_pY']
        dz = segments[i]['start_pZ'] - segments[i-1]['start_pZ']
        d = math.sqrt(dy*dy + dz*dz)
        if segments[i]['gt'] == 'ground':
            gnd_dY.append(dy); gnd_dZ.append(dz); gnd_dist.append(d)
        else:
            up_dY.append(dy); up_dZ.append(dz); up_dist.append(d)

    if gnd_dY and up_dY:
        print(f"    Ground: ΔpY med={median(gnd_dY):+.3f} std={std(gnd_dY):.3f}  |  ΔpZ med={median(gnd_dZ):+.3f} std={std(gnd_dZ):.3f}  |  dist med={median(gnd_dist):.3f}")
        print(f"    Stair:  ΔpY med={median(up_dY):+.3f} std={std(up_dY):.3f}  |  ΔpZ med={median(up_dZ):+.3f} std={std(up_dZ):.3f}  |  dist med={median(up_dist):.3f}")
        # ΔpY, ΔpZ의 분산 차이
        print(f"    ΔpY std비교: Gnd={std(gnd_dY):.3f} vs Stair={std(up_dY):.3f}  (ratio={std(gnd_dY)/max(std(up_dY),0.001):.2f}x)")
        print(f"    ΔpZ std비교: Gnd={std(gnd_dZ):.3f} vs Stair={std(up_dZ):.3f}  (ratio={std(gnd_dZ)/max(std(up_dZ),0.001):.2f}x)")
        # 이동 거리 비교
        p_dist = mann_whitney_p(gnd_dist, up_dist)
        ovlp_dist = iqr_overlap(gnd_dist, up_dist)
        ps = f"{p_dist:.1e}" if not math.isnan(p_dist) else "N/A"
        print(f"    이동 거리: p={ps}  IQR겹침={ovlp_dist:.1f}%")


# ══════════════════════════════════════════════════════════════
#  3개 합산 분석
# ══════════════════════════════════════════════════════════════
print(f"\n\n{'='*85}")
print(f"  COMBINED (3개 합산)")
print(f"{'='*85}")

gnd_all = [f for f in all_features_rows if f['gt']=='ground']
up_all = [f for f in all_features_rows if f['gt'] in ('up','down','stair')]

FEAT_LIST_EXT = [
    ('start_pZ',     '시작점 pZ'),
    ('start_pY',     '시작점 pY'),
    ('rangeZ',       'pZ 범위'),
    ('rangeY',       'pY 범위'),
    ('aspect',       '종횡비 Z/Y'),
    ('zFrac',        'var비율 Z/(Z+Y)'),
    ('area',         '궤적 면적'),
    ('path_len',     '경로 길이'),
    ('meanZ_offset', 'pZ 중심 offset'),
    ('maxZ_offset',  'pZ max offset'),
]

print(f"\n  {'Feature':<20} {'Gnd_med':>8} {'Up_med':>8} {'Gap':>8} {'IQR겹침':>7} {'p':>10} {'판정'}")
print(f"  {'─'*75}")
for fk, fname in FEAT_LIST_EXT:
    g = [f[fk] for f in gnd_all]
    u = [f[fk] for f in up_all]
    mg, mu = median(g), median(u)
    gap = mu - mg
    ovlp = iqr_overlap(g, u)
    p = mann_whitney_p(g, u)
    ps = f"{p:.1e}" if not math.isnan(p) else "N/A"
    tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
    print(f"  {fk:<20} {mg:>8.3f} {mu:>8.3f} {gap:>+8.3f} {ovlp:>6.1f}% {ps:>10}  {tag} {fname}")

# ── TSV 저장 ──
tsv_path = os.path.join(OUT_DIR, "trajectory_features_per_step.tsv")
keys = ['dataset','step','gt','start_pY','start_pZ','rangeY','rangeZ','aspect',
        'zFrac','area','path_len','meanZ_offset','maxZ_offset','angle_deg','quadrant','n_samples']
with open(tsv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys, delimiter='\t', extrasaction='ignore')
    writer.writeheader()
    for r in all_features_rows:
        writer.writerow(r)
print(f"\n스텝별 궤적 피처 저장: {tsv_path}")
