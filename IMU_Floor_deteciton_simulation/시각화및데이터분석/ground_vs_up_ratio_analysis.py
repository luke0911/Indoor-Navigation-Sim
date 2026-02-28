#!/usr/bin/env python3
"""
Ground vs Up 궤적 변화 비율 분석

핵심 질문: "평지 보행 → 계단 올라감" 전환 시, 궤적 모양이
           어떤 방향으로, 얼마의 비율로 변하는가?

사람마다 절대값은 다르지만, 변화 비율(ratio)이 일정하면
State Machine으로 감지할 수 있다.

각 데이터셋(=각 보행 스타일)에서:
  ratio = median(up) / median(ground)
  → ratio가 3개 데이터셋 모두에서 같은 방향이면 robust feature
"""
import os, math, csv

DATA_DIR = "/Users/idohun/WorkSpace/KU-LAB/IMU_Floor_deteciton_simulation/data/output"

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


def load_step_segments(filepath):
    """batch TSV → 스텝별 (pureY, pureZ) 구간 추출. ground / up만."""
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
        gt = step_boundaries[si][2]

        # ground 와 up 만 (down 제외)
        if gt not in ('ground', 'up', 'stair'):
            continue

        label = 'ground' if gt == 'ground' else 'up'

        pY, pZ = [], []
        for j in range(start_idx, end_idx):
            try:
                pY.append(float(samples[j]['pureY']))
                pZ.append(float(samples[j]['pureZ']))
            except:
                pass
        if len(pY) < 5:
            continue
        segments.append({'gt': label, 'pY': pY, 'pZ': pZ, 'n': len(pY)})

    return segments


def compute_features(seg):
    """한 스텝의 궤적 shape 피처 계산"""
    pY, pZ = seg['pY'], seg['pZ']
    n = len(pY)

    cY, cZ = mean(pY), mean(pZ)
    cy = [y - cY for y in pY]
    cz = [z - cZ for z in pZ]

    f = {}

    # 바운딩 박스
    rangeY = max(pY) - min(pY)
    rangeZ = max(pZ) - min(pZ)
    f['rangeY'] = rangeY
    f['rangeZ'] = rangeZ
    f['aspect'] = rangeZ / (rangeY + 1e-9)

    # PCA elongation (eigenvalue ratio)
    varY = sum(y*y for y in cy) / n
    varZ = sum(z*z for z in cz) / n
    covYZ = sum(cy[i]*cz[i] for i in range(n)) / n
    trace = varY + varZ
    det = varY * varZ - covYZ**2
    disc = max(trace**2/4 - det, 0)
    lam1 = trace/2 + math.sqrt(disc)
    lam2 = max(trace/2 - math.sqrt(disc), 1e-12)
    f['elongation'] = lam1 / lam2

    # 전반부/후반부 이동거리
    half = n // 2
    f['first_dist'] = math.sqrt((pY[half]-pY[0])**2 + (pZ[half]-pZ[0])**2)
    f['second_dist'] = math.sqrt((pY[-1]-pY[half])**2 + (pZ[-1]-pZ[half])**2)

    # 전반부/후반부 방향각
    f['first_angle'] = math.degrees(math.atan2(pZ[half]-pZ[0], pY[half]-pY[0]))
    f['second_angle'] = math.degrees(math.atan2(pZ[-1]-pZ[half], pY[-1]-pY[half]))

    # Z에너지 비율 (후반/전반)
    first_z_e = sum(cz[i]**2 for i in range(half)) / max(half, 1)
    second_z_e = sum(cz[i]**2 for i in range(half, n)) / max(n-half, 1)
    f['z_shift'] = second_z_e / (first_z_e + 1e-9)

    # 궤적 복잡도 (path / diagonal)
    total_path = sum(math.sqrt((pY[i+1]-pY[i])**2+(pZ[i+1]-pZ[i])**2) for i in range(n-1))
    diag = math.sqrt(rangeY**2 + rangeZ**2) + 1e-9
    f['complexity'] = total_path / diag

    # varZ / (varY + varZ) — Z 에너지 비중
    f['varZ_frac'] = varZ / (varY + varZ + 1e-9)

    # 순이동 방향각 (절대값 — 수직에 가까울수록 계단)
    net_dY = pY[-1] - pY[0]
    net_dZ = pZ[-1] - pZ[0]
    f['net_angle_abs'] = abs(math.degrees(math.atan2(net_dZ, net_dY)))

    # 경로 길이 / n (속도 proxy)
    f['path_per_sample'] = total_path / n

    # Y축 에너지 (수평 흔들림 크기)
    f['varY'] = varY
    f['varZ'] = varZ

    # 전반부 Z 이동량 (상승 크기)
    f['first_dZ'] = abs(pZ[half] - pZ[0])
    f['first_dY'] = abs(pY[half] - pY[0])

    return f


# ─── 분석 시작 ───

print("=" * 100)
print("  Ground vs Up 궤적 변화 비율 분석")
print("  목표: 사람/보행스타일과 무관하게 일관적으로 변하는 피처 찾기")
print("=" * 100)

FEAT_NAMES = [
    'rangeY', 'rangeZ', 'aspect', 'elongation',
    'first_dist', 'second_dist', 'first_dZ', 'first_dY',
    'z_shift', 'complexity', 'varZ_frac',
    'path_per_sample', 'varY', 'varZ',
]

FEAT_LABELS = {
    'rangeY': '가로폭 (Y range)',
    'rangeZ': '세로폭 (Z range)',
    'aspect': '종횡비 (Z/Y)',
    'elongation': 'PCA 길쭉함',
    'first_dist': '전반부 이동거리',
    'second_dist': '후반부 이동거리',
    'first_dZ': '전반부 |ΔZ|',
    'first_dY': '전반부 |ΔY|',
    'z_shift': 'Z에너지 후/전',
    'complexity': '궤적 복잡도',
    'varZ_frac': 'Z분산 비중',
    'path_per_sample': '샘플당 경로길이',
    'varY': 'Y분산 (수평 에너지)',
    'varZ': 'Z분산 (수직 에너지)',
}

# 데이터셋별 ratio 수집
all_ratios = {}  # feat -> [ratio_일반, ratio_살살, ratio_거칠게]

for ds_name, ds_file in DATASETS.items():
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path):
        print(f"  {ds_name}: NOT FOUND"); continue

    segments = load_step_segments(path)
    gnd = [s for s in segments if s['gt'] == 'ground']
    up = [s for s in segments if s['gt'] == 'up']

    print(f"\n{'─'*100}")
    print(f"  {ds_name}: ground={len(gnd)}, up={len(up)}")
    print(f"{'─'*100}")

    gnd_feats = [compute_features(s) for s in gnd]
    up_feats = [compute_features(s) for s in up]

    print(f"\n  {'피처':<22} {'Ground med':>12} {'Up med':>12} {'ratio':>8} {'방향':>12} {'해석'}")
    print(f"  {'─'*90}")

    for fk in FEAT_NAMES:
        g_vals = [f[fk] for f in gnd_feats]
        u_vals = [f[fk] for f in up_feats]
        g_med = median_val(g_vals)
        u_med = median_val(u_vals)

        if abs(g_med) > 1e-9:
            ratio = u_med / g_med
        else:
            ratio = float('inf') if u_med > 0 else 1.0

        if fk not in all_ratios:
            all_ratios[fk] = []
        all_ratios[fk].append(ratio)

        direction = "UP ↑" if ratio > 1.05 else ("UP ↓" if ratio < 0.95 else "≈ 같음")

        # 해석
        interp = ""
        if fk == 'rangeY' and ratio < 1:
            interp = "수평 폭 줄어듦"
        elif fk == 'rangeZ' and ratio > 1:
            interp = "수직 폭 커짐"
        elif fk == 'aspect' and ratio > 1:
            interp = "더 세로로 길쭉"
        elif fk == 'elongation' and ratio > 1:
            interp = "궤적이 더 일직선"
        elif fk == 'varY' and ratio < 1:
            interp = "수평 흔들림 감소"
        elif fk == 'varZ' and ratio > 1:
            interp = "수직 흔들림 증가"

        print(f"  {FEAT_LABELS.get(fk,fk):<22} {g_med:>12.3f} {u_med:>12.3f} {ratio:>8.3f} {direction:>12} {interp}")


# ─── CROSS-DATASET 일관성 분석 ───

print(f"\n\n{'='*100}")
print(f"  CROSS-DATASET 일관성: ratio(up/ground)가 3개 데이터셋 모두 같은 방향?")
print(f"{'='*100}")

print(f"\n  {'피처':<22} {'일반보행':>10} {'살살걸음':>10} {'거칠게':>10} {'일관성':>8} {'평균ratio':>10} {'판정'}")
print(f"  {'─'*90}")

consistent_feats = []

for fk in FEAT_NAMES:
    ratios = all_ratios.get(fk, [])
    if len(ratios) < 3:
        continue

    r1, r2, r3 = ratios[0], ratios[1], ratios[2]

    # 일관성: 3개 모두 같은 방향 (>1 or <1)
    all_up = all(r > 1.0 for r in ratios)
    all_down = all(r < 1.0 for r in ratios)
    consistent = all_up or all_down

    avg_ratio = mean(ratios)
    min_ratio = min(ratios)
    max_ratio = max(ratios)

    # 방향이 일관되고, 변화 폭이 10% 이상이면 유용
    useful = consistent and (avg_ratio > 1.10 or avg_ratio < 0.90)
    # 변화 폭이 일관되면 더 좋음 (min/max 비율이 가까움)
    ratio_spread = max_ratio / (min_ratio + 1e-9) if consistent else 999

    tag = ""
    if useful and ratio_spread < 3.0:
        tag = "★★★ (강력)"
        consistent_feats.append((fk, avg_ratio, ratio_spread))
    elif useful:
        tag = "★★ (유용)"
        consistent_feats.append((fk, avg_ratio, ratio_spread))
    elif consistent:
        tag = "★ (약함)"
    else:
        tag = "✗ (불일치)"

    print(f"  {FEAT_LABELS.get(fk,fk):<22} {r1:>10.3f} {r2:>10.3f} {r3:>10.3f} "
          f"{'OK' if consistent else 'NO':>8} {avg_ratio:>10.3f} {tag}")


# ─── State Machine 설계를 위한 제안 ───

print(f"\n\n{'='*100}")
print(f"  State Machine 설계를 위한 제안")
print(f"{'='*100}")

print(f"""
  [원리] 스텝 디텍션과 같은 접근:
  - 절대 임계값 X → 자기 자신의 baseline 대비 변화 감지
  - 매 스텝마다 궤적 shape feature를 계산
  - "최근 N스텝의 feature" vs "자기 baseline" 비율로 판정

  [State Machine 구조]

  ┌──────────┐    ratio 변화 감지 (2+ 연속)    ┌──────────┐
  │  GROUND  │ ─────────────────────────────→ │   UP     │
  │ (평지)   │ ←───────────────────────────── │ (계단↑)  │
  └──────────┘    ratio 복귀 (2+ 연속)         └──────────┘

  1. GROUND 상태에서 baseline을 지속 업데이트 (EWMA)
  2. 매 스텝: current_feature / baseline → ratio 계산
  3. ratio가 임계 비율을 넘으면 → 상태 전환 후보
  4. 2+ 연속이면 확정 (hysteresis)

  [핵심] 임계값이 "절대값"이 아니라 "비율":
  - "aspect ratio > 2.0" (X) → 사람마다 다름
  - "aspect ratio가 baseline 대비 1.5배 이상" (O) → 보편적
""")

print(f"  [일관성 있는 피처들 (3개 데이터셋 모두 같은 방향)]")
for fk, avg_r, spread in sorted(consistent_feats, key=lambda x: abs(x[1]-1.0), reverse=True):
    direction = "상승" if avg_r > 1 else "하락"
    print(f"    {FEAT_LABELS.get(fk,fk):<22}: 평균 ratio={avg_r:.3f} ({direction})  spread={spread:.2f}")

print(f"""
  [추천 조합]
  - 궤적이 "세로로 길쭉해지는" 패턴: elongation ratio ↑ + aspect ratio ↑
  - 궤적이 "수평 움직임 줄어드는" 패턴: rangeY ratio ↓ + varY ratio ↓
  - 두 패턴 중 하나라도 baseline 대비 변화하면 → UP 후보
  - 2연속 확정 → 빠른 반응 (~1초)
""")
