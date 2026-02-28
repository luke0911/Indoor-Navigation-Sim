#!/usr/bin/env python3
"""
연속 스텝 궤적 추적 — analysis 창에서 보이는 것처럼

batch TSV (매 샘플)를 사용하여:
1. 스텝별 (pureY, pureZ) 궤적을 시간순으로 보여줌
2. 연속 스텝의 시작점(pY, pZ) 이동 경로
3. ground→stair 전환 전후에 궤적/시작점이 어떻게 바뀌는지
4. ASCII 2D plot으로 시각화
"""
import os, csv, math

DATA_DIR = "/Users/idohun/WorkSpace/KU-LAB/IMU_Floor_deteciton_simulation/data/output"

DATASETS = [
    ("일반보행",   "batch_Floor_sim_data_일반보행.tsv"),
    ("살살걸음",   "batch_Floor_sim_data_살살걸음.tsv"),
    ("거칠게걸음", "batch_Floor_sim_data_거칠게 걸음.tsv"),
]


def load_batch_segments(filepath):
    """batch TSV → 스텝별 (pY, pZ) 세그먼트"""
    samples = []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            samples.append(row)

    step_bounds = []
    for i, s in enumerate(samples):
        if s.get('isStep') == '1':
            step_bounds.append((i, int(s['stepIdx']), s['floorGT']))

    segments = []
    for si in range(len(step_bounds) - 1):
        s_idx = step_bounds[si][0]
        e_idx = step_bounds[si+1][0]
        step_num = step_bounds[si][1]
        gt = step_bounds[si][2]

        pY, pZ = [], []
        for j in range(s_idx, e_idx):
            try:
                pY.append(float(samples[j]['pureY']))
                pZ.append(float(samples[j]['pureZ']))
            except:
                pass

        if len(pY) >= 3:
            segments.append({
                'step': step_num, 'gt': gt,
                'pY': pY, 'pZ': pZ,
                'start_pY': pY[0], 'start_pZ': pZ[0],
                'end_pY': pY[-1], 'end_pZ': pZ[-1],
                'n': len(pY),
            })
    return segments


def ascii_2d(points, labels=None, w=60, h=20, title=""):
    """2D ASCII scatter plot. points = list of (y, z), labels = list of str"""
    if not points:
        return
    ys = [p[0] for p in points]
    zs = [p[1] for p in points]
    minY, maxY = min(ys), max(ys)
    minZ, maxZ = min(zs), max(zs)
    rY = maxY - minY + 1e-9
    rZ = maxZ - minZ + 1e-9

    grid = [[' '] * w for _ in range(h)]

    for i, (y, z) in enumerate(points):
        col = int((y - minY) / rY * (w - 1))
        row = h - 1 - int((z - minZ) / rZ * (h - 1))
        col = max(0, min(w-1, col))
        row = max(0, min(h-1, row))
        if labels:
            ch = labels[i]
        else:
            ch = str(i % 10)
        if grid[row][col] == ' ':
            grid[row][col] = ch

    print(f"    {title}")
    print(f"    pZ↑  Y:[{minY:.1f} ~ {maxY:.1f}]  Z:[{minZ:.1f} ~ {maxZ:.1f}]")
    for row in grid:
        print(f"    |{''.join(row)}|")
    print(f"    +{'-' * w}+ →pY")


def find_transitions(segments):
    trans = []
    for i in range(1, len(segments)):
        pg = segments[i-1]['gt']
        cg = segments[i]['gt']
        if pg == 'ground' and cg in ('up', 'down'):
            trans.append((i, f"ground→{cg}"))
        elif pg in ('up', 'down') and cg == 'ground':
            trans.append((i, f"{pg}→ground"))
    return trans


# ══════════════════════════════════════════════════════════
print("=" * 80)
print("  연속 스텝 궤적 추적 (pureY-pureZ)")
print("=" * 80)

for ds_name, ds_file in DATASETS:
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path):
        print(f"\n  {ds_name}: NOT FOUND"); continue

    segments = load_batch_segments(path)
    transitions = find_transitions(segments)

    print(f"\n\n{'='*80}")
    print(f"  {ds_name}: {len(segments)} steps, {len(transitions)} transitions")
    print(f"{'='*80}")

    # ── 1. 전체 시작점 이동 경로 ──
    print(f"\n  [1] 전체 시작점 이동 경로:")
    print(f"  {'step':>4} {'GT':>6} {'시작pY':>8} {'시작pZ':>8} {'→ΔpY':>7} {'→ΔpZ':>7} {'끝pY':>8} {'끝pZ':>8} {'궤적내ΔpY':>9} {'궤적내ΔpZ':>9}")
    print(f"  {'─'*85}")

    for i, seg in enumerate(segments):
        gt = seg['gt'][:3]
        spY, spZ = seg['start_pY'], seg['start_pZ']
        epY, epZ = seg['end_pY'], seg['end_pZ']

        # 스텝 내 이동 (시작→끝)
        in_dY = epY - spY
        in_dZ = epZ - spZ

        # 이전 스텝 대비 시작점 이동
        if i > 0:
            dpY = spY - segments[i-1]['start_pY']
            dpZ = spZ - segments[i-1]['start_pZ']
            dpY_s = f"{dpY:>+7.3f}"
            dpZ_s = f"{dpZ:>+7.3f}"
        else:
            dpY_s = f"{'':>7}"
            dpZ_s = f"{'':>7}"

        marker = ""
        for tidx, ttype in transitions:
            if tidx == i:
                marker = f"  ◀── {ttype}"

        print(f"  {seg['step']:>4} {gt:>6}"
              f" {spY:>+8.3f} {spZ:>+8.3f}"
              f" {dpY_s} {dpZ_s}"
              f" {epY:>+8.3f} {epZ:>+8.3f}"
              f" {in_dY:>+9.3f} {in_dZ:>+9.3f}"
              f"{marker}")

    # ── 2. 전환 구간별 궤적 상세 + ASCII plot ──
    WINDOW = 4  # 전환 전후 4스텝

    for tidx, ttype in transitions:
        start = max(0, tidx - WINDOW)
        end = min(len(segments), tidx + WINDOW)

        print(f"\n\n  [2] 전환 상세: {ttype} (step {segments[tidx]['step']})")
        print(f"  {'='*75}")

        # 전환 전후 각 스텝의 궤적을 순서대로 표시
        for si in range(start, end):
            seg = segments[si]
            gt = seg['gt']
            is_trans = (si == tidx)

            pY, pZ = seg['pY'], seg['pZ']
            n = len(pY)

            # 궤적 요약: 시작→1/4→1/2→3/4→끝
            pts = []
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                idx = min(int(frac * (n-1)), n-1)
                pts.append((pY[idx], pZ[idx]))

            rangeY = max(pY) - min(pY)
            rangeZ = max(pZ) - min(pZ)

            arrow = "◀ 전환점" if is_trans else ""
            print(f"\n    step {seg['step']:>3} [{gt:>6}] ({n} samples) {arrow}")
            print(f"      시작:  pY={pY[0]:>+7.3f}  pZ={pZ[0]:>+7.3f}")
            print(f"      1/4:   pY={pts[1][0]:>+7.3f}  pZ={pts[1][1]:>+7.3f}")
            print(f"      1/2:   pY={pts[2][0]:>+7.3f}  pZ={pts[2][1]:>+7.3f}")
            print(f"      3/4:   pY={pts[3][0]:>+7.3f}  pZ={pts[3][1]:>+7.3f}")
            print(f"      끝:    pY={pY[-1]:>+7.3f}  pZ={pZ[-1]:>+7.3f}")
            print(f"      범위:  rangeY={rangeY:.3f}  rangeZ={rangeZ:.3f}  aspect={rangeZ/(rangeY+1e-9):.2f}")

        # 전환 전후 시작점들을 하나의 2D plot에
        plot_pts = []
        plot_labels = []
        for si in range(start, end):
            seg = segments[si]
            plot_pts.append((seg['start_pY'], seg['start_pZ']))
            gt_ch = 'G' if seg['gt'] == 'ground' else ('U' if seg['gt'] == 'up' else 'D')
            if si == tidx:
                plot_labels.append('*')  # 전환점
            else:
                plot_labels.append(gt_ch)

        print(f"\n    시작점 이동 경로 (G=ground, U/D=stair, *=전환점):")
        ascii_2d(plot_pts, plot_labels, w=50, h=12,
                 title=f"{ttype} 전후 시작점")

        # 전환 전후 궤적을 겹쳐 표시
        # 전환 전 마지막 ground 스텝과 전환 후 첫 stair 스텝
        before_seg = segments[tidx - 1] if tidx > 0 else None
        trans_seg = segments[tidx]

        if before_seg:
            print(f"\n    궤적 비교: 전환 직전 step {before_seg['step']}({before_seg['gt']}) vs 전환 step {trans_seg['step']}({trans_seg['gt']})")
            # 두 궤적을 하나의 plot에
            all_pts = []
            all_labels = []

            # before 궤적 (소문자)
            bpY, bpZ = before_seg['pY'], before_seg['pZ']
            step_n = len(bpY)
            for k in range(0, step_n, max(1, step_n // 8)):
                all_pts.append((bpY[k], bpZ[k]))
                all_labels.append('.')

            # transition 궤적 (대문자)
            tpY, tpZ = trans_seg['pY'], trans_seg['pZ']
            step_n = len(tpY)
            for k in range(0, step_n, max(1, step_n // 8)):
                all_pts.append((tpY[k], tpZ[k]))
                all_labels.append('#')

            ascii_2d(all_pts, all_labels, w=50, h=14,
                     title=f".=step{before_seg['step']}({before_seg['gt']})  #=step{trans_seg['step']}({trans_seg['gt']})")

    # ── 3. ground 연속구간 vs stair 연속구간 — 시작점 이동 패턴 ──
    print(f"\n\n  [3] 연속 구간별 시작점 이동 패턴:")

    # 연속 구간 분리
    segs_grouped = []
    g_start = 0
    for i in range(1, len(segments)):
        if segments[i]['gt'] != segments[i-1]['gt']:
            segs_grouped.append((g_start, i-1, segments[g_start]['gt']))
            g_start = i
    segs_grouped.append((g_start, len(segments)-1, segments[g_start]['gt']))

    for g_s, g_e, gt in segs_grouped:
        n = g_e - g_s + 1
        if n < 2:
            continue

        print(f"\n    구간 step {segments[g_s]['step']}-{segments[g_e]['step']} [{gt}] ({n}스텝):")

        # 시작점 이동 벡터
        dYs, dZs, dists = [], [], []
        for i in range(g_s + 1, g_e + 1):
            dy = segments[i]['start_pY'] - segments[i-1]['start_pY']
            dz = segments[i]['start_pZ'] - segments[i-1]['start_pZ']
            dYs.append(dy)
            dZs.append(dz)
            dists.append(math.sqrt(dy*dy + dz*dz))

        mean_dY = sum(dYs) / len(dYs)
        mean_dZ = sum(dZs) / len(dZs)
        mean_dist = sum(dists) / len(dists)

        # 시작점 범위
        spy_list = [segments[i]['start_pY'] for i in range(g_s, g_e+1)]
        spz_list = [segments[i]['start_pZ'] for i in range(g_s, g_e+1)]
        pY_range = max(spy_list) - min(spy_list)
        pZ_range = max(spz_list) - min(spz_list)

        print(f"      시작점 범위:  pY [{min(spy_list):>+6.2f} ~ {max(spy_list):>+6.2f}] (range={pY_range:.2f})")
        print(f"                    pZ [{min(spz_list):>+6.2f} ~ {max(spz_list):>+6.2f}] (range={pZ_range:.2f})")
        print(f"      스텝간 이동:  평균ΔpY={mean_dY:>+6.3f}  평균ΔpZ={mean_dZ:>+6.3f}  평균거리={mean_dist:.3f}")

        # 시작점들 ASCII plot
        pts = [(segments[i]['start_pY'], segments[i]['start_pZ']) for i in range(g_s, g_e+1)]
        lbls = [str(i % 10) for i in range(n)]
        ascii_2d(pts, lbls, w=45, h=10,
                 title=f"step {segments[g_s]['step']}-{segments[g_e]['step']} [{gt}] 시작점 순서(0→{n-1})")

        # 궤적 방향 일관성: 연속 ΔpZ가 같은 부호인 비율
        if len(dZs) >= 2:
            same_sign_z = sum(1 for i in range(len(dZs)-1) if dZs[i]*dZs[i+1] > 0)
            print(f"      ΔpZ 부호 연속성: {same_sign_z}/{len(dZs)-1} ({same_sign_z/(len(dZs)-1)*100:.0f}%) — {'일정한 방향' if same_sign_z/(len(dZs)-1) > 0.6 else '번갈아 변화'}")

            same_sign_y = sum(1 for i in range(len(dYs)-1) if dYs[i]*dYs[i+1] > 0)
            print(f"      ΔpY 부호 연속성: {same_sign_y}/{len(dYs)-1} ({same_sign_y/(len(dYs)-1)*100:.0f}%) — {'일정한 방향' if same_sign_y/(len(dYs)-1) > 0.6 else '번갈아 변화'}")

    # ── 4. 스텝 내 궤적 형태 연속 비교 ──
    print(f"\n\n  [4] 스텝 내 궤적 형태 — 연속 스텝 비교:")

    for tidx, ttype in transitions:
        start = max(0, tidx - 3)
        end = min(len(segments), tidx + 4)

        print(f"\n    {ttype} (step {segments[tidx]['step']}) 전후 궤적 형태:")
        print(f"    {'step':>4} {'GT':>6} {'시작(pY,pZ)':>18} {'→끝(pY,pZ)':>18} {'rY':>6} {'rZ':>6} {'종횡비':>6} {'궤적형태':>12}")

        for si in range(start, end):
            seg = segments[si]
            pY, pZ = seg['pY'], seg['pZ']
            n = len(pY)
            rY = max(pY) - min(pY)
            rZ = max(pZ) - min(pZ)
            asp = rZ / (rY + 1e-9)

            # 궤적 형태 판단
            if asp > 3:
                shape = "↕ Z축길쭉"
            elif asp > 1.5:
                shape = "↕ Z우세"
            elif asp > 0.67:
                shape = "◇ 정방"
            elif asp > 0.33:
                shape = "↔ Y우세"
            else:
                shape = "↔ Y축길쭉"

            arrow = " ◀" if si == tidx else ""
            gt = seg['gt'][:3]
            print(f"    {seg['step']:>4} {gt:>6}"
                  f" ({seg['start_pY']:>+6.2f},{seg['start_pZ']:>+6.2f})"
                  f" ({seg['end_pY']:>+6.2f},{seg['end_pZ']:>+6.2f})"
                  f" {rY:>6.2f} {rZ:>6.2f} {asp:>6.2f}"
                  f" {shape}{arrow}")


print(f"\n\n분석 완료.")
