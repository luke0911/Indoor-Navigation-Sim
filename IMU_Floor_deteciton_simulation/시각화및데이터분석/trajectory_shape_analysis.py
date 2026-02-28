#!/usr/bin/env python3
"""
pureY-pureZ 궤적 형태 상세 분석

한 스텝 구간의 (pureY, pureZ) 시계열이 2D 평면에서 그리는 궤적을
다각도로 분석:

1. 궤적 진행 방향 분석 (시간 순 어디로 이동하는지)
2. 궤적 회전 방향 (시계/반시계)
3. 궤적 구간별 분해 (전반/후반 패턴)
4. 궤적 주축 방향 (PCA 없이 — 공분산 기반)
5. 궤적 대칭성/비대칭성
6. 정규화 궤적 평균 패턴 + 시각적 비교

모든 피처: 스텝 구간 pureY[0..N], pureZ[0..N]만 사용 (실시간 가능)
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


def load_step_segments(filepath):
    samples = []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            samples.append(row)
    step_boundaries = []
    for i, s in enumerate(samples):
        if s.get('isStep') == '1':
            step_boundaries.append((i, int(s['stepIdx']), s['floorGT']))
    segments = []
    for si in range(len(step_boundaries) - 1):
        start_idx = step_boundaries[si][0]
        end_idx = step_boundaries[si+1][0]
        step_num = step_boundaries[si][1]
        gt = step_boundaries[si][2]
        seg_pY, seg_pZ = [], []
        for j in range(start_idx, end_idx):
            try:
                seg_pY.append(float(samples[j]['pureY']))
                seg_pZ.append(float(samples[j]['pureZ']))
            except: pass
        if len(seg_pY) < 5: continue
        segments.append({'step': step_num, 'gt': gt, 'pY': seg_pY, 'pZ': seg_pZ, 'n': len(seg_pY)})
    return segments


# ══════════════════════════════════════════════════════════
#  궤적 형태 피처 계산
# ══════════════════════════════════════════════════════════
def analyze_trajectory(seg):
    """한 스텝의 (pY, pZ) 궤적 형태를 다각도로 분석"""
    pY, pZ = seg['pY'], seg['pZ']
    n = len(pY)

    # 무게중심 기준으로 중심화
    cY, cZ = mean(pY), mean(pZ)
    cy = [y - cY for y in pY]
    cz = [z - cZ for z in pZ]

    result = {}

    # ── 0. 바운딩 박스: 가로폭, 세로폭, 종횡비 ──
    rangeY = max(pY) - min(pY)  # 가로폭 (Y축 = 수평)
    rangeZ = max(pZ) - min(pZ)  # 세로폭 (Z축 = 수직)
    aspect_ratio = rangeZ / (rangeY + 1e-9)  # > 1 = 세로 긴 (계단)
    bbox_area = rangeY * rangeZ
    result['rangeY'] = rangeY
    result['rangeZ'] = rangeZ
    result['aspect_ratio'] = aspect_ratio
    result['bbox_area'] = bbox_area

    # ── 1. 궤적 진행 방향: 시작→1/4→1/2→3/4→끝 ──
    # 궤적을 시간 5등분해서 각 지점의 (pY, pZ) 위치
    quarters = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(frac * (n-1)), n-1)
        quarters.append((pY[idx], pZ[idx]))
    result['quarters'] = quarters

    # 시작→끝 벡터 (순이동)
    net_dY = pY[-1] - pY[0]
    net_dZ = pZ[-1] - pZ[0]
    net_dist = math.sqrt(net_dY**2 + net_dZ**2)
    net_angle = math.degrees(math.atan2(net_dZ, net_dY))  # -180~180
    result['net_dY'] = net_dY
    result['net_dZ'] = net_dZ
    result['net_dist'] = net_dist
    result['net_angle'] = net_angle

    # ── 2. 궤적 회전 방향 (signed area = 시계/반시계) ──
    # shoelace의 부호: 양수 = 반시계, 음수 = 시계
    signed_area = 0
    for i in range(n-1):
        signed_area += cy[i]*cz[i+1] - cy[i+1]*cz[i]
    signed_area /= 2.0
    result['signed_area'] = signed_area
    result['rotation'] = 'CCW' if signed_area > 0 else 'CW'

    # ── 3. 전반부 vs 후반부 궤적 분해 ──
    half = n // 2

    # 전반부: 시작 → 중간
    first_dY = pY[half] - pY[0]
    first_dZ = pZ[half] - pZ[0]
    first_angle = math.degrees(math.atan2(first_dZ, first_dY))
    first_dist = math.sqrt(first_dY**2 + first_dZ**2)

    # 후반부: 중간 → 끝
    second_dY = pY[-1] - pY[half]
    second_dZ = pZ[-1] - pZ[half]
    second_angle = math.degrees(math.atan2(second_dZ, second_dY))
    second_dist = math.sqrt(second_dY**2 + second_dZ**2)

    result['first_angle'] = first_angle
    result['first_dist'] = first_dist
    result['second_angle'] = second_angle
    result['second_dist'] = second_dist

    # 전반/후반 방향 꺾임 (turning angle)
    turn = second_angle - first_angle
    if turn > 180: turn -= 360
    if turn < -180: turn += 360
    result['turn_angle'] = turn

    # ── 4. 주축 방향 (공분산 기반 PCA 대체) ──
    # cov(pY, pZ), var(pY), var(pZ) → 주축 각도
    varY = sum(y*y for y in cy) / n
    varZ = sum(z*z for z in cz) / n
    covYZ = sum(cy[i]*cz[i] for i in range(n)) / n

    # 주축 각도: 0.5 * atan2(2*cov, varY - varZ)
    if abs(varY - varZ) < 1e-12 and abs(covYZ) < 1e-12:
        principal_angle = 0
    else:
        principal_angle = math.degrees(0.5 * math.atan2(2*covYZ, varY - varZ))
    result['principal_angle'] = principal_angle

    # 주축 분산비 (elongation) — eigenvalue ratio
    trace = varY + varZ
    det = varY * varZ - covYZ * covYZ
    disc = max(trace*trace/4 - det, 0)
    lam1 = trace/2 + math.sqrt(disc)
    lam2 = max(trace/2 - math.sqrt(disc), 1e-12)
    result['elongation'] = lam1 / lam2  # 1이면 원형, 크면 길쭉

    # ── 5. 궤적 대칭성 ──
    # 전반부 경로 길이 vs 후반부 경로 길이
    first_path = sum(math.sqrt((pY[i+1]-pY[i])**2+(pZ[i+1]-pZ[i])**2) for i in range(half))
    second_path = sum(math.sqrt((pY[i+1]-pY[i])**2+(pZ[i+1]-pZ[i])**2) for i in range(half, n-1))
    result['path_symmetry'] = first_path / (second_path + 1e-9)  # 1이면 대칭

    # 전반부 vs 후반부 Z 지배력
    first_z_energy = sum(cz[i]**2 for i in range(half)) / max(half, 1)
    second_z_energy = sum(cz[i]**2 for i in range(half, n)) / max(n - half, 1)
    result['z_energy_shift'] = second_z_energy / (first_z_energy + 1e-9)

    # ── 6. pZ 피크 위치 (전체 구간 대비) ──
    # pZ가 최대/최소인 시점이 구간의 어디쯤인지 (0~1)
    max_z_pos = pZ.index(max(pZ)) / (n - 1)
    min_z_pos = pZ.index(min(pZ)) / (n - 1)
    result['max_z_pos'] = max_z_pos  # 0=시작, 1=끝
    result['min_z_pos'] = min_z_pos

    # pZ 피크가 먼저 오는지 (max_z_pos < min_z_pos)
    result['z_peak_first'] = max_z_pos < min_z_pos

    # ── 7. 궤적 복잡도 (path / convex hull 대각선) ──
    total_path = sum(math.sqrt((pY[i+1]-pY[i])**2+(pZ[i+1]-pZ[i])**2) for i in range(n-1))
    rangeY = max(pY) - min(pY)
    rangeZ = max(pZ) - min(pZ)
    diag = math.sqrt(rangeY**2 + rangeZ**2) + 1e-9
    result['complexity'] = total_path / diag

    # ── 8. 시작→피크 경로 vs 피크→끝 경로 ──
    peak_idx = pZ.index(max(pZ))
    if peak_idx > 0 and peak_idx < n-1:
        rise_path = sum(math.sqrt((pY[i+1]-pY[i])**2+(pZ[i+1]-pZ[i])**2) for i in range(peak_idx))
        fall_path = sum(math.sqrt((pY[i+1]-pY[i])**2+(pZ[i+1]-pZ[i])**2) for i in range(peak_idx, n-1))
        result['rise_fall_ratio'] = rise_path / (fall_path + 1e-9)
    else:
        result['rise_fall_ratio'] = 1.0

    # ── 9. Y 방향 "왕복" 패턴 분석 ──
    # pY의 부호 변경 횟수 (중심 기준)
    y_crossings = sum(1 for i in range(n-1) if cy[i]*cy[i+1] < 0)
    result['y_crossings'] = y_crossings

    # Z 방향 부호 변경 횟수
    z_crossings = sum(1 for i in range(n-1) if cz[i]*cz[i+1] < 0)
    result['z_crossings'] = z_crossings

    # ── 10. 시간 가중 중심 (궤적이 어느 쪽에 오래 머무는지) ──
    # 시간 1/3씩 나눈 체류 영역
    t1 = n // 3
    t2 = 2 * n // 3
    early_cZ = mean(pZ[:t1]) if t1 > 0 else pZ[0]
    mid_cZ = mean(pZ[t1:t2]) if t2 > t1 else pZ[t1]
    late_cZ = mean(pZ[t2:]) if t2 < n else pZ[-1]
    result['early_z'] = early_cZ
    result['mid_z'] = mid_cZ
    result['late_z'] = late_cZ

    early_cY = mean(pY[:t1]) if t1 > 0 else pY[0]
    mid_cY = mean(pY[t1:t2]) if t2 > t1 else pY[t1]
    late_cY = mean(pY[t2:]) if t2 < n else pY[-1]
    result['early_y'] = early_cY
    result['mid_y'] = mid_cY
    result['late_y'] = late_cY

    # Z의 초기→중기→후기 변화 패턴
    result['z_trend'] = f"{early_cZ:+.1f}→{mid_cZ:+.1f}→{late_cZ:+.1f}"

    return result


# ── 정규화 궤적 (2D resampling for average shape) ──
def normalize_2d_trajectory(pY, pZ, n_points=20):
    """궤적을 n_points로 리샘플 + 무게중심 원점, 최대 범위 1로 정규화"""
    n = len(pY)
    rY, rZ = [], []
    for i in range(n_points):
        t = i / (n_points - 1) * (n - 1)
        idx = int(t); frac = t - idx
        if idx >= n - 1:
            rY.append(pY[-1]); rZ.append(pZ[-1])
        else:
            rY.append(pY[idx]*(1-frac) + pY[idx+1]*frac)
            rZ.append(pZ[idx]*(1-frac) + pZ[idx+1]*frac)
    # 중심화
    cY, cZ = mean(rY), mean(rZ)
    rY = [y - cY for y in rY]
    rZ = [z - cZ for z in rZ]
    # 최대 범위로 정규화
    scale = max(max(abs(y) for y in rY), max(abs(z) for z in rZ), 1e-9)
    rY = [y / scale for y in rY]
    rZ = [z / scale for z in rZ]
    return rY, rZ


def format_p(p):
    if math.isnan(p): return "N/A"
    return f"{p:.1e}"


def _ascii_plot(tY, tZ, w=41, h=15):
    """간단한 ASCII 2D trajectory plot"""
    grid = [[' ']*w for _ in range(h)]
    minY, maxY = min(tY), max(tY)
    minZ, maxZ = min(tZ), max(tZ)
    rY = maxY - minY + 1e-9
    rZ = maxZ - minZ + 1e-9
    for i in range(len(tY)):
        col = int((tY[i] - minY) / rY * (w - 1))
        row = h - 1 - int((tZ[i] - minZ) / rZ * (h - 1))
        col = max(0, min(w-1, col))
        row = max(0, min(h-1, row))
        if i == 0:
            grid[row][col] = 'S'
        elif i == len(tY) - 1:
            grid[row][col] = 'E'
        else:
            marker = str(i) if i < 10 else chr(ord('a') + i - 10)
            if grid[row][col] == ' ':
                grid[row][col] = marker
    for row in grid:
        print(f"      |{''.join(row)}|")
    print(f"      +{'-'*w}+")


# ══════════════════════════════════════════════════════════
#  메인 분석
# ══════════════════════════════════════════════════════════
print("=" * 95)
print("  pureY-pureZ 궤적 형태 상세 분석")
print("  모든 분석: 한 스텝 구간의 (pureY, pureZ) 시계열만 사용")
print("=" * 95)

all_rows = []

for ds_name, ds_file in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path):
        print(f"\n  {ds_name}: NOT FOUND"); continue

    segments = load_step_segments(path)
    gnd_segs = [s for s in segments if s['gt'] == 'ground']
    stair_segs = [s for s in segments if s['gt'] in ('up', 'down', 'stair')]

    print(f"\n\n{'='*95}")
    print(f"  {ds_name}: {len(segments)} steps (ground={len(gnd_segs)}, stair={len(stair_segs)})")
    print(f"{'='*95}")

    gnd_feats, stair_feats = [], []
    for seg in segments:
        f = analyze_trajectory(seg)
        f['gt'] = seg['gt']
        f['step'] = seg['step']
        f['dataset'] = ds_name
        f['n'] = seg['n']
        all_rows.append(f)
        if seg['gt'] == 'ground':
            gnd_feats.append(f)
        elif seg['gt'] in ('up', 'down', 'stair'):
            stair_feats.append(f)

    if not gnd_feats or not stair_feats:
        print("  No stair segments!"); continue

    # ── A. 궤적 진행 방향 (전반/후반) ──
    print(f"\n  [A] 궤적 진행 방향 (전반 → 후반)")
    print(f"  {'':>20} {'Ground (median)':>20} {'Stair (median)':>20} {'IQR겹침':>8} {'p':>10}")
    print(f"  {'─'*80}")

    for fk, label in [
        ('first_angle',   '전반부 방향(°)'),
        ('first_dist',    '전반부 이동거리'),
        ('second_angle',  '후반부 방향(°)'),
        ('second_dist',   '후반부 이동거리'),
        ('turn_angle',    '꺾임각(°)'),
        ('net_angle',     '순이동 방향(°)'),
        ('net_dist',      '순이동 거리'),
    ]:
        g = [f[fk] for f in gnd_feats]
        u = [f[fk] for f in stair_feats]
        mg, mu = median_val(g), median_val(u)
        ovlp = iqr_overlap(g, u)
        p = mann_whitney_p(g, u)
        tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
        print(f"  {label:<20} {mg:>+12.2f}         {mu:>+12.2f}         {ovlp:>6.1f}% {format_p(p):>10} {tag}")

    # ── B. 궤적 회전 방향 분포 ──
    gnd_cw = sum(1 for f in gnd_feats if f['rotation']=='CW')
    gnd_ccw = len(gnd_feats) - gnd_cw
    st_cw = sum(1 for f in stair_feats if f['rotation']=='CW')
    st_ccw = len(stair_feats) - st_cw
    print(f"\n  [B] 궤적 회전 방향")
    print(f"    Ground: CW {gnd_cw} ({gnd_cw/len(gnd_feats)*100:.0f}%)  CCW {gnd_ccw} ({gnd_ccw/len(gnd_feats)*100:.0f}%)")
    print(f"    Stair:  CW {st_cw} ({st_cw/len(stair_feats)*100:.0f}%)  CCW {st_ccw} ({st_ccw/len(stair_feats)*100:.0f}%)")

    # ── C. 주축/elongation/대칭성 ──
    print(f"\n  [C] 궤적 형태 특성")
    print(f"  {'':>20} {'Ground (median)':>20} {'Stair (median)':>20} {'IQR겹침':>8} {'p':>10}")
    print(f"  {'─'*80}")

    for fk, label in [
        ('rangeY',          '가로폭(Y range)'),
        ('rangeZ',          '세로폭(Z range)'),
        ('aspect_ratio',    '종횡비(Z/Y)'),
        ('bbox_area',       'BBox 면적'),
        ('principal_angle', '주축 방향(°)'),
        ('elongation',      '길쭉함(주축비)'),
        ('path_symmetry',   '경로 대칭(전/후)'),
        ('z_energy_shift',  'Z에너지 후반/전반'),
        ('complexity',      '궤적 복잡도'),
        ('rise_fall_ratio', '상승/하강 경로비'),
    ]:
        g = [f[fk] for f in gnd_feats]
        u = [f[fk] for f in stair_feats]
        mg, mu = median_val(g), median_val(u)
        ovlp = iqr_overlap(g, u)
        p = mann_whitney_p(g, u)
        tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
        print(f"  {label:<20} {mg:>12.3f}         {mu:>12.3f}         {ovlp:>6.1f}% {format_p(p):>10} {tag}")

    # ── D. pZ 피크 위치 ──
    print(f"\n  [D] pZ 피크 타이밍 (0.0=스텝시작, 1.0=스텝끝)")
    for fk, label in [
        ('max_z_pos', 'pZ 최대 위치'),
        ('min_z_pos', 'pZ 최소 위치'),
    ]:
        g = [f[fk] for f in gnd_feats]
        u = [f[fk] for f in stair_feats]
        mg, mu = median_val(g), median_val(u)
        ovlp = iqr_overlap(g, u)
        p = mann_whitney_p(g, u)
        tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
        print(f"    {label}: Ground {mg:.3f}  Stair {mu:.3f}  IQR겹침={ovlp:.1f}%  p={format_p(p)} {tag}")

    # pZ max → min 순서
    gnd_peak_first = sum(1 for f in gnd_feats if f['z_peak_first'])
    st_peak_first = sum(1 for f in stair_feats if f['z_peak_first'])
    print(f"    pZ 최대가 먼저(상승→하강 패턴): Ground {gnd_peak_first}/{len(gnd_feats)} ({gnd_peak_first/len(gnd_feats)*100:.0f}%)  Stair {st_peak_first}/{len(stair_feats)} ({st_peak_first/len(stair_feats)*100:.0f}%)")

    # ── E. Y/Z 축 교차 (진동 특성) ──
    print(f"\n  [E] 중심축 교차 횟수 (진동 패턴)")
    for fk, label in [
        ('y_crossings', 'pY 중심축 교차'),
        ('z_crossings', 'pZ 중심축 교차'),
    ]:
        g = [f[fk] for f in gnd_feats]
        u = [f[fk] for f in stair_feats]
        mg, mu = median_val(g), median_val(u)
        ovlp = iqr_overlap(g, u)
        p = mann_whitney_p(g, u)
        tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
        print(f"    {label}: Ground {mg:.1f}  Stair {mu:.1f}  IQR겹침={ovlp:.1f}%  p={format_p(p)} {tag}")

    # ── F. 시간 3등분 Z 체류 ──
    print(f"\n  [F] 시간 3등분 pZ 체류 (초기/중기/후기 평균 pZ)")
    for phase, fk in [('초기','early_z'), ('중기','mid_z'), ('후기','late_z')]:
        g = [f[fk] for f in gnd_feats]
        u = [f[fk] for f in stair_feats]
        mg, mu = median_val(g), median_val(u)
        ovlp = iqr_overlap(g, u)
        p = mann_whitney_p(g, u)
        tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
        print(f"    {phase}: Ground {mg:>+7.2f}  Stair {mu:>+7.2f}  gap={mu-mg:>+6.2f}  IQR겹침={ovlp:.1f}%  p={format_p(p)} {tag}")

    # ── G. 평균 정규화 2D 궤적 (ASCII 시각화) ──
    gnd_norm = [normalize_2d_trajectory(s['pY'], s['pZ']) for s in gnd_segs]
    stair_norm = [normalize_2d_trajectory(s['pY'], s['pZ']) for s in stair_segs]

    gnd_avg_Y = [mean([t[0][i] for t in gnd_norm]) for i in range(20)]
    gnd_avg_Z = [mean([t[1][i] for t in gnd_norm]) for i in range(20)]
    stair_avg_Y = [mean([t[0][i] for t in stair_norm]) for i in range(20)]
    stair_avg_Z = [mean([t[1][i] for t in stair_norm]) for i in range(20)]

    print(f"\n  [G] 평균 정규화 궤적 (20-point, 중심화+스케일링)")
    print(f"    시간  Ground(Y, Z)         Stair(Y, Z)")
    for i in range(20):
        t_pct = i / 19 * 100
        print(f"    {t_pct:5.1f}% ({gnd_avg_Y[i]:>+6.3f}, {gnd_avg_Z[i]:>+6.3f})    ({stair_avg_Y[i]:>+6.3f}, {stair_avg_Z[i]:>+6.3f})")

    # ASCII plot: 궤적을 11x21 그리드에 표시
    print(f"\n    Ground 평균 궤적 (Y→, Z↑):")
    _ascii_plot(gnd_avg_Y, gnd_avg_Z)
    print(f"    Stair 평균 궤적 (Y→, Z↑):")
    _ascii_plot(stair_avg_Y, stair_avg_Z)


# ══════════════════════════════════════════════════════════
#  COMBINED 분석
# ══════════════════════════════════════════════════════════
print(f"\n\n{'='*95}")
print(f"  COMBINED (3개 합산) — 궤적 형태 피처 분리력 순위")
print(f"{'='*95}")

gnd_all = [r for r in all_rows if r['gt']=='ground']
stair_all = [r for r in all_rows if r['gt'] in ('up','down','stair')]
print(f"  총 {len(gnd_all)} ground + {len(stair_all)} stair steps")

ALL_FEATS = [
    ('rangeY',          'lt',  '가로폭(Y range)'),
    ('rangeZ',          'gt',  '세로폭(Z range)'),
    ('aspect_ratio',    'gt',  '종횡비(Z/Y)'),
    ('bbox_area',       'gt',  'BBox 면적'),
    ('first_angle',     'gt',  '전반부 방향(°)'),
    ('first_dist',      'gt',  '전반부 이동거리'),
    ('second_angle',    'lt',  '후반부 방향(°)'),
    ('second_dist',     'gt',  '후반부 이동거리'),
    ('turn_angle',      'gt',  '꺾임각(°)'),
    ('net_angle',       'gt',  '순이동 방향(°)'),
    ('net_dist',        'lt',  '순이동 거리'),
    ('principal_angle', 'gt',  '주축 방향(°)'),
    ('elongation',      'gt',  '길쭉함'),
    ('path_symmetry',   'gt',  '경로 대칭'),
    ('z_energy_shift',  'gt',  'Z에너지 후/전 비'),
    ('complexity',      'gt',  '궤적 복잡도'),
    ('rise_fall_ratio', 'gt',  '상승/하강 비'),
    ('max_z_pos',       'gt',  'pZ피크 위치'),
    ('min_z_pos',       'gt',  'pZ최소 위치'),
    ('y_crossings',     'lt',  'Y축 교차수'),
    ('z_crossings',     'lt',  'Z축 교차수'),
]

results = []
for fk, direction, label in ALL_FEATS:
    g = [f[fk] for f in gnd_all]
    u = [f[fk] for f in stair_all]
    mg, mu = median_val(g), median_val(u)
    gap = mu - mg
    ovlp = iqr_overlap(g, u)
    p = mann_whitney_p(g, u)
    tag = "✅" if p < 0.01 and ovlp < 30 else ("⚠️" if p < 0.05 else "❌")
    results.append((fk, label, mg, mu, gap, ovlp, p, tag))

# IQR 겹침 기준 정렬
results.sort(key=lambda x: x[5])

print(f"\n  {'순위':>3} {'피처':<20} {'라벨':<18} {'Gnd_med':>8} {'Str_med':>8} {'Gap':>8} {'IQR겹침':>7} {'p':>10} {'판정'}")
print(f"  {'─'*100}")
for rank, (fk, label, mg, mu, gap, ovlp, p, tag) in enumerate(results, 1):
    print(f"  {rank:>3}  {fk:<20} {label:<18} {mg:>+8.3f} {mu:>+8.3f} {gap:>+8.3f} {ovlp:>6.1f}% {format_p(p):>10}  {tag}")


# ══════════════════════════════════════════════════════════
#  COMBINED 궤적 패턴 요약
# ══════════════════════════════════════════════════════════
print(f"\n\n{'='*95}")
print(f"  궤적 패턴 요약")
print(f"{'='*95}")

# 회전 방향
gnd_cw = sum(1 for f in gnd_all if f['rotation']=='CW')
st_cw = sum(1 for f in stair_all if f['rotation']=='CW')
print(f"\n  회전 방향:")
print(f"    Ground: CW {gnd_cw}/{len(gnd_all)} ({gnd_cw/len(gnd_all)*100:.0f}%)")
print(f"    Stair:  CW {st_cw}/{len(stair_all)} ({st_cw/len(stair_all)*100:.0f}%)")

# pZ 피크 순서
gnd_pf = sum(1 for f in gnd_all if f['z_peak_first'])
st_pf = sum(1 for f in stair_all if f['z_peak_first'])
print(f"\n  pZ 최대가 먼저 (상승→하강):")
print(f"    Ground: {gnd_pf}/{len(gnd_all)} ({gnd_pf/len(gnd_all)*100:.0f}%)")
print(f"    Stair:  {st_pf}/{len(stair_all)} ({st_pf/len(stair_all)*100:.0f}%)")

# 평균 Z 트렌드
print(f"\n  시간 3등분 pZ 평균 (전체 패턴):")
for label, feats in [("Ground", gnd_all), ("Stair", stair_all)]:
    ez = median_val([f['early_z'] for f in feats])
    mz = median_val([f['mid_z'] for f in feats])
    lz = median_val([f['late_z'] for f in feats])
    ey = median_val([f['early_y'] for f in feats])
    my = median_val([f['mid_y'] for f in feats])
    ly = median_val([f['late_y'] for f in feats])
    print(f"    {label}: pZ = {ez:>+6.2f} → {mz:>+6.2f} → {lz:>+6.2f}  |  pY = {ey:>+6.2f} → {my:>+6.2f} → {ly:>+6.2f}")

# COMBINED 평균 2D 궤적
print(f"\n  COMBINED 평균 정규화 2D 궤적:")
all_segs = []
for ds_name, ds_file in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_file)
    if os.path.exists(path):
        all_segs.extend(load_step_segments(path))

gnd_segs_all = [s for s in all_segs if s['gt'] == 'ground']
stair_segs_all = [s for s in all_segs if s['gt'] in ('up','down','stair')]

gnd_norm = [normalize_2d_trajectory(s['pY'], s['pZ']) for s in gnd_segs_all]
stair_norm = [normalize_2d_trajectory(s['pY'], s['pZ']) for s in stair_segs_all]

gnd_avg_Y = [mean([t[0][i] for t in gnd_norm]) for i in range(20)]
gnd_avg_Z = [mean([t[1][i] for t in gnd_norm]) for i in range(20)]
stair_avg_Y = [mean([t[0][i] for t in stair_norm]) for i in range(20)]
stair_avg_Z = [mean([t[1][i] for t in stair_norm]) for i in range(20)]

print(f"\n    Ground 평균 궤적:")
_ascii_plot(gnd_avg_Y, gnd_avg_Z)
print(f"    Stair 평균 궤적:")
_ascii_plot(stair_avg_Y, stair_avg_Z)

# 차이 분석
print(f"\n  Ground vs Stair 궤적 차이 (시간순):")
print(f"    {'시간':>6} {'Gnd_Y':>7} {'Str_Y':>7} {'ΔY':>7} {'Gnd_Z':>7} {'Str_Z':>7} {'ΔZ':>7}")
for i in range(20):
    t_pct = i / 19 * 100
    dy = stair_avg_Y[i] - gnd_avg_Y[i]
    dz = stair_avg_Z[i] - gnd_avg_Z[i]
    bar = '█' * int(abs(dz) * 20)
    sign = '+' if dz >= 0 else '-'
    print(f"    {t_pct:5.1f}% {gnd_avg_Y[i]:>+7.3f} {stair_avg_Y[i]:>+7.3f} {dy:>+7.3f} {gnd_avg_Z[i]:>+7.3f} {stair_avg_Z[i]:>+7.3f} {dz:>+7.3f} {sign}{bar}")


# TSV 저장
tsv_path = os.path.join(OUT_DIR, "trajectory_shape_features.tsv")
keys = ['dataset','step','gt','n',
        'rangeY','rangeZ','aspect_ratio','bbox_area',
        'first_angle','first_dist','second_angle','second_dist','turn_angle',
        'net_angle','net_dist','rotation','signed_area',
        'principal_angle','elongation','path_symmetry','z_energy_shift',
        'complexity','rise_fall_ratio',
        'max_z_pos','min_z_pos','z_peak_first',
        'y_crossings','z_crossings',
        'early_z','mid_z','late_z','early_y','mid_y','late_y']
with open(tsv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys, delimiter='\t', extrasaction='ignore')
    writer.writeheader()
    for r in all_rows:
        writer.writerow(r)
print(f"\n\n궤적 형태 피처 TSV 저장: {tsv_path}")
