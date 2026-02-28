#!/usr/bin/env python3
"""
스텝 시계열 분석 — 연속적 시간순 데이터로서의 분석

step_analysis TSV의 각 스텝을 시간순으로 읽으며:
1. 각 피처의 시계열 흐름을 그대로 출력
2. ground→stair, stair→ground 전환 구간 전후 변화 패턴
3. 전환 N스텝 전부터 피처가 어떻게 변하는지 (선행 지표)
4. 스텝별 pureY, pureZ 시작점 이동 궤적
"""
import os, csv, math

DATA_DIR = "/Users/idohun/WorkSpace/KU-LAB/IMU_Floor_deteciton_simulation/data/output"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    ("일반보행",   "step_analysis_Floor_sim_data_일반보행.tsv"),
    ("살살걸음",   "step_analysis_Floor_sim_data_살살걸음.tsv"),
    ("거칠게걸음", "step_analysis_Floor_sim_data_거칠게 걸음.tsv"),
]

def mean(v): return sum(v)/max(len(v),1)


def load_steps(filepath):
    steps = []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            d = {}
            for k, v in row.items():
                try: d[k] = float(v)
                except: d[k] = v
            steps.append(d)
    return steps


def find_transitions(steps):
    """ground→stair, stair→ground 전환점 찾기"""
    transitions = []
    for i in range(1, len(steps)):
        prev_gt = steps[i-1]['floorGT']
        curr_gt = steps[i]['floorGT']
        if prev_gt == 'ground' and curr_gt in ('up','down'):
            transitions.append((i, f'ground→{curr_gt}'))
        elif prev_gt in ('up','down') and curr_gt == 'ground':
            transitions.append((i, f'{prev_gt}→ground'))
    return transitions


# ══════════════════════════════════════════════════════════
#  메인 분석
# ══════════════════════════════════════════════════════════
print("=" * 100)
print("  스텝 시계열 분석 — step_analysis 데이터의 연속적 변화")
print("=" * 100)

# 분석할 주요 피처
KEY_FEATS = ['pureZ', 'pureY', 'zEnergyFrac', 'procRmsZ', 'stepVarH', 'energyFracZ']

for ds_name, ds_file in DATASETS:
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path):
        print(f"\n  {ds_name}: NOT FOUND"); continue

    steps = load_steps(path)
    transitions = find_transitions(steps)

    print(f"\n\n{'='*100}")
    print(f"  {ds_name}: {len(steps)} steps, {len(transitions)} transitions")
    print(f"{'='*100}")

    # ── 1. 전환점 목록 ──
    print(f"\n  전환점 목록:")
    for idx, ttype in transitions:
        s = steps[idx]
        print(f"    step {s['stepIdx']:>3.0f} (t={s['time_s']:>6.1f}s): {ttype}")

    # ── 2. 전체 시계열 출력 (주요 피처) ──
    print(f"\n  전체 스텝 시계열:")
    print(f"  {'step':>4} {'time':>6} {'GT':>6} {'pureZ':>8} {'pureY':>8} {'zEF':>7} {'rmsZ':>7} {'stepVH':>7} {'eFracZ':>7} {'dmag':>7}")
    print(f"  {'─'*80}")

    for i, s in enumerate(steps):
        gt = s['floorGT']
        # 전환점 표시
        marker = ""
        for tidx, ttype in transitions:
            if tidx == i:
                marker = f"  ◀── {ttype}"
                break

        gt_short = gt[:3] if isinstance(gt, str) else gt
        print(f"  {s['stepIdx']:>4.0f} {s['time_s']:>6.1f} {gt_short:>6}"
              f" {s.get('pureZ',0):>+8.3f}"
              f" {s.get('pureY',0):>+8.3f}"
              f" {s.get('zEnergyFrac',0):>7.3f}"
              f" {s.get('procRmsZ',0):>7.3f}"
              f" {s.get('stepVarH',0):>7.3f}"
              f" {s.get('energyFracZ',0):>7.3f}"
              f" {s.get('dmag',0):>7.3f}"
              f"{marker}")

    # ── 3. 전환 구간 상세 분석 (전후 5스텝) ──
    WINDOW = 5
    print(f"\n\n  전환 구간 상세 (전후 {WINDOW}스텝):")
    print(f"  {'='*90}")

    for tidx, ttype in transitions:
        start = max(0, tidx - WINDOW)
        end = min(len(steps), tidx + WINDOW + 1)

        print(f"\n  ▶ {ttype} (step {steps[tidx]['stepIdx']:.0f}, t={steps[tidx]['time_s']:.1f}s)")
        print(f"  {'':>4} {'GT':>6} {'pureZ':>8} {'Δ_pZ':>7} {'pureY':>8} {'Δ_pY':>7} {'zEF':>7} {'Δ_zEF':>7} {'rmsZ':>7} {'Δ_rmsZ':>7}")
        print(f"  {'─'*85}")

        prev_pZ, prev_pY, prev_zEF, prev_rmsZ = None, None, None, None
        for i in range(start, end):
            s = steps[i]
            pZ = s.get('pureZ', 0)
            pY = s.get('pureY', 0)
            zEF = s.get('zEnergyFrac', 0)
            rmsZ = s.get('procRmsZ', 0)

            dpZ = f"{pZ - prev_pZ:>+7.3f}" if prev_pZ is not None else f"{'':>7}"
            dpY = f"{pY - prev_pY:>+7.3f}" if prev_pY is not None else f"{'':>7}"
            dzEF = f"{zEF - prev_zEF:>+7.3f}" if prev_zEF is not None else f"{'':>7}"
            drmsZ = f"{rmsZ - prev_rmsZ:>+7.3f}" if prev_rmsZ is not None else f"{'':>7}"

            arrow = " ◀── 전환" if i == tidx else ""
            gt_short = s['floorGT'][:3] if isinstance(s['floorGT'], str) else s['floorGT']
            print(f"  {s['stepIdx']:>4.0f} {gt_short:>6}"
                  f" {pZ:>+8.3f} {dpZ}"
                  f" {pY:>+8.3f} {dpY}"
                  f" {zEF:>7.3f} {dzEF}"
                  f" {rmsZ:>7.3f} {drmsZ}"
                  f"{arrow}")

            prev_pZ, prev_pY, prev_zEF, prev_rmsZ = pZ, pY, zEF, rmsZ

    # ── 4. pureZ, pureY 시작점 연속 이동 궤적 ──
    print(f"\n\n  pureY-pureZ 시작점 연속 이동 궤적:")
    print(f"  {'step':>4} {'GT':>6} {'pureY':>8} {'pureZ':>8} {'ΔpY':>7} {'ΔpZ':>7} {'이동거리':>8} {'이동방향°':>8}")
    print(f"  {'─'*65}")

    for i, s in enumerate(steps):
        pY = s.get('pureY', 0)
        pZ = s.get('pureZ', 0)
        gt_short = s['floorGT'][:3] if isinstance(s['floorGT'], str) else s['floorGT']

        if i > 0:
            dpY = pY - steps[i-1].get('pureY', 0)
            dpZ = pZ - steps[i-1].get('pureZ', 0)
            dist = math.sqrt(dpY**2 + dpZ**2)
            angle = math.degrees(math.atan2(dpZ, dpY))

            marker = ""
            for tidx, ttype in transitions:
                if tidx == i:
                    marker = f"  ◀── {ttype}"
                    break

            print(f"  {s['stepIdx']:>4.0f} {gt_short:>6}"
                  f" {pY:>+8.3f} {pZ:>+8.3f}"
                  f" {dpY:>+7.3f} {dpZ:>+7.3f}"
                  f" {dist:>8.3f} {angle:>+8.1f}"
                  f"{marker}")
        else:
            print(f"  {s['stepIdx']:>4.0f} {gt_short:>6}"
                  f" {pY:>+8.3f} {pZ:>+8.3f}"
                  f" {'':>7} {'':>7}"
                  f" {'':>8} {'':>8}")

    # ── 5. 연속 스텝 간 변화량의 패턴 ──
    print(f"\n\n  구간별 피처 변화 요약:")

    # GT가 같은 연속 구간 찾기
    segments = []
    seg_start = 0
    for i in range(1, len(steps)):
        curr_gt = steps[i]['floorGT']
        prev_gt = steps[i-1]['floorGT']
        if curr_gt != prev_gt:
            segments.append((seg_start, i-1, prev_gt))
            seg_start = i
    segments.append((seg_start, len(steps)-1, steps[-1]['floorGT']))

    print(f"\n  연속 구간:")
    print(f"  {'구간':>10} {'GT':>8} {'N':>4} {'pZ_평균':>8} {'pZ_std':>8} {'pY_평균':>8} {'pY_std':>8} "
          f"{'zEF_평균':>8} {'zEF_std':>8} {'rmsZ_평균':>8} {'rmsZ_std':>8}")
    print(f"  {'─'*105}")

    for s_start, s_end, gt in segments:
        n = s_end - s_start + 1
        pZ_vals = [steps[i].get('pureZ',0) for i in range(s_start, s_end+1)]
        pY_vals = [steps[i].get('pureY',0) for i in range(s_start, s_end+1)]
        zEF_vals = [steps[i].get('zEnergyFrac',0) for i in range(s_start, s_end+1)]
        rmsZ_vals = [steps[i].get('procRmsZ',0) for i in range(s_start, s_end+1)]

        pZ_m = mean(pZ_vals)
        pZ_s = math.sqrt(sum((x-pZ_m)**2 for x in pZ_vals)/max(n,1)) if n > 1 else 0
        pY_m = mean(pY_vals)
        pY_s = math.sqrt(sum((x-pY_m)**2 for x in pY_vals)/max(n,1)) if n > 1 else 0
        zEF_m = mean(zEF_vals)
        zEF_s = math.sqrt(sum((x-zEF_m)**2 for x in zEF_vals)/max(n,1)) if n > 1 else 0
        rmsZ_m = mean(rmsZ_vals)
        rmsZ_s = math.sqrt(sum((x-rmsZ_m)**2 for x in rmsZ_vals)/max(n,1)) if n > 1 else 0

        idx_range = f"{steps[s_start]['stepIdx']:.0f}-{steps[s_end]['stepIdx']:.0f}"
        gt_short = gt[:5] if isinstance(gt, str) else gt
        print(f"  {idx_range:>10} {gt_short:>8} {n:>4}"
              f" {pZ_m:>+8.3f} {pZ_s:>8.3f}"
              f" {pY_m:>+8.3f} {pY_s:>8.3f}"
              f" {zEF_m:>8.3f} {zEF_s:>8.3f}"
              f" {rmsZ_m:>8.3f} {rmsZ_s:>8.3f}")

    # ── 6. 전환점 전후 피처 변화 추세 (선행 지표 탐색) ──
    print(f"\n\n  전환 선행 지표 분석 (전환점 기준 -5 ~ +5 스텝):")

    for tidx, ttype in transitions:
        start = max(0, tidx - WINDOW)
        end = min(len(steps), tidx + WINDOW + 1)

        print(f"\n  ▶ {ttype} (step {steps[tidx]['stepIdx']:.0f})")

        # 전환 전 5스텝의 추세
        before = list(range(max(0, tidx-WINDOW), tidx))
        after = list(range(tidx, min(len(steps), tidx+WINDOW)))

        if len(before) >= 2:
            # 추세 = 마지막 - 처음
            for feat in ['pureZ', 'pureY', 'zEnergyFrac', 'procRmsZ', 'stepVarH']:
                bvals = [steps[i].get(feat, 0) for i in before]
                avals = [steps[i].get(feat, 0) for i in after]

                # 전환 전 추세 (선형)
                b_trend = (bvals[-1] - bvals[0]) / max(len(bvals)-1, 1)
                # 전환 후 추세
                a_trend = (avals[-1] - avals[0]) / max(len(avals)-1, 1) if len(avals) >= 2 else 0
                # 전환 시점 점프
                jump = avals[0] - bvals[-1] if avals else 0

                print(f"    {feat:<15}: 전 추세={b_trend:>+7.3f}/step  점프={jump:>+7.3f}  후 추세={a_trend:>+7.3f}/step")


# ══════════════════════════════════════════════════════════
#  전체 요약: 전환 패턴 공통점 찾기
# ══════════════════════════════════════════════════════════
print(f"\n\n{'='*100}")
print(f"  전환 패턴 공통점 분석 (3개 데이터셋 합산)")
print(f"{'='*100}")

# 모든 전환 수집
all_transitions_data = {}  # ttype → list of (before_feats, after_feats, jump)

for ds_name, ds_file in DATASETS:
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path): continue

    steps = load_steps(path)
    transitions = find_transitions(steps)

    for tidx, ttype in transitions:
        base_type = 'ground→stair' if '→up' in ttype or '→down' in ttype or '→sta' in ttype else 'stair→ground'

        if base_type not in all_transitions_data:
            all_transitions_data[base_type] = {f: {'jumps': [], 'before_trends': [], 'after_trends': []}
                                                for f in KEY_FEATS}

        before = list(range(max(0, tidx-5), tidx))
        after = list(range(tidx, min(len(steps), tidx+5)))

        for feat in KEY_FEATS:
            bvals = [steps[i].get(feat, 0) for i in before]
            avals = [steps[i].get(feat, 0) for i in after]

            if len(bvals) >= 2:
                b_trend = (bvals[-1] - bvals[0]) / max(len(bvals)-1, 1)
                all_transitions_data[base_type][feat]['before_trends'].append(b_trend)

            if avals and bvals:
                jump = avals[0] - bvals[-1]
                all_transitions_data[base_type][feat]['jumps'].append(jump)

            if len(avals) >= 2:
                a_trend = (avals[-1] - avals[0]) / max(len(avals)-1, 1)
                all_transitions_data[base_type][feat]['after_trends'].append(a_trend)

for ttype, feats_data in sorted(all_transitions_data.items()):
    n_trans = len(feats_data[KEY_FEATS[0]]['jumps'])
    print(f"\n  {ttype} ({n_trans}회):")
    print(f"  {'피처':<15} {'전 추세(med)':>12} {'점프(med)':>12} {'후 추세(med)':>12} {'점프 방향 일관성':>16}")
    print(f"  {'─'*70}")

    for feat in KEY_FEATS:
        fd = feats_data[feat]
        jumps = fd['jumps']
        bt = fd['before_trends']
        at = fd['after_trends']

        if not jumps: continue

        j_med = sorted(jumps)[len(jumps)//2]
        bt_med = sorted(bt)[len(bt)//2] if bt else 0
        at_med = sorted(at)[len(at)//2] if at else 0

        # 점프 방향 일관성: 같은 부호인 비율
        pos = sum(1 for j in jumps if j > 0)
        neg = sum(1 for j in jumps if j < 0)
        consistency = max(pos, neg) / len(jumps) * 100

        print(f"  {feat:<15} {bt_med:>+12.4f} {j_med:>+12.4f} {at_med:>+12.4f} {consistency:>13.0f}% ({'일관' if consistency >= 70 else '불일관'})")


# ── 결과 저장 ──
out_path = os.path.join(OUT_DIR, "timeseries_analysis_result.txt")
print(f"\n\n분석 완료. 전환 패턴을 시계열로 확인하세요.")
