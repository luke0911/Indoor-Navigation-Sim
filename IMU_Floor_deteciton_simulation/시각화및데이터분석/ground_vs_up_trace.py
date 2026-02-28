#!/usr/bin/env python3
"""
Ground vs Up 궤적 상세 비교 (down 제외)

batch TSV에서 ground/up 구간만 추출하여:
1. 스텝 내 궤적 이동 방향 (시작→끝 벡터)
2. 연속 스텝 시작점 군집 정도
3. 스텝간 시작점 이동 벡터 방향
4. 궤적 내 pZ 피크 방향, pY 이동 방향
"""
import os, csv, math

DATA_DIR = "/Users/idohun/WorkSpace/KU-LAB/IMU_Floor_deteciton_simulation/data/output"

DATASETS = [
    ("일반보행",   "batch_Floor_sim_data_일반보행.tsv"),
    ("살살걸음",   "batch_Floor_sim_data_살살걸음.tsv"),
    ("거칠게걸음", "batch_Floor_sim_data_거칠게 걸음.tsv"),
]

def mean(v): return sum(v)/max(len(v),1)
def std_val(v):
    m = mean(v)
    return math.sqrt(sum((x-m)**2 for x in v)/max(len(v),1))


def load_segments(filepath):
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
        gt = step_bounds[si][2]
        pY, pZ = [], []
        for j in range(s_idx, e_idx):
            try:
                pY.append(float(samples[j]['pureY']))
                pZ.append(float(samples[j]['pureZ']))
            except: pass
        if len(pY) >= 5:
            segments.append({
                'step': step_bounds[si][1], 'gt': gt,
                'pY': pY, 'pZ': pZ, 'n': len(pY),
            })
    return segments


def ascii_2d(points, labels, w=55, h=16, title=""):
    if not points: return
    ys = [p[0] for p in points]
    zs = [p[1] for p in points]
    minY, maxY = min(ys), max(ys)
    minZ, maxZ = min(zs), max(zs)
    rY = maxY - minY + 1e-9
    rZ = maxZ - minZ + 1e-9
    grid = [[' ']*w for _ in range(h)]
    for i, (y, z) in enumerate(points):
        col = int((y - minY) / rY * (w - 1))
        row = h - 1 - int((z - minZ) / rZ * (h - 1))
        col = max(0, min(w-1, col))
        row = max(0, min(h-1, row))
        ch = labels[i] if i < len(labels) else '.'
        if grid[row][col] == ' ':
            grid[row][col] = ch
    if title:
        print(f"      {title}")
    print(f"      pZ↑ Y:[{minY:.2f}~{maxY:.2f}] Z:[{minZ:.2f}~{maxZ:.2f}]")
    for row in grid:
        print(f"      |{''.join(row)}|")
    print(f"      +{'-'*w}+ →pY")


print("=" * 85)
print("  Ground vs Up 궤적 상세 비교 (down 제외)")
print("=" * 85)

for ds_name, ds_file in DATASETS:
    path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(path): continue

    segments = load_segments(path)

    # ground/up만, down 제외
    segments = [s for s in segments if s['gt'] in ('ground', 'up')]
    if not segments: continue

    # 연속 구간 분리
    runs = []
    rs = 0
    for i in range(1, len(segments)):
        if segments[i]['gt'] != segments[i-1]['gt']:
            runs.append((rs, i-1, segments[rs]['gt']))
            rs = i
    runs.append((rs, len(segments)-1, segments[rs]['gt']))

    # ground/up 구간만
    gnd_runs = [(s,e,gt) for s,e,gt in runs if gt == 'ground']
    up_runs = [(s,e,gt) for s,e,gt in runs if gt == 'up']

    print(f"\n\n{'='*85}")
    print(f"  {ds_name}: ground 구간 {len(gnd_runs)}개, up 구간 {len(up_runs)}개")
    print(f"{'='*85}")

    # ══════════════════════════════════════════
    # A. 각 up 구간과 그 직전 ground 구간 비교
    # ══════════════════════════════════════════
    for rs, re, gt in up_runs:
        up_segs = segments[rs:re+1]
        # 직전 ground 구간 찾기
        prev_gnd = None
        for gs, ge, ggt in gnd_runs:
            if ge < rs:
                prev_gnd = (gs, ge)
        if prev_gnd is None: continue

        gs, ge = prev_gnd
        gnd_segs = segments[gs:ge+1]

        # 비교할 구간: ground 마지막 5스텝 vs up 전체
        gnd_last = gnd_segs[-min(5, len(gnd_segs)):]
        up_all = up_segs

        print(f"\n\n  ────────────────────────────────────────────────────────")
        print(f"  직전 Ground (step {gnd_segs[0]['step']}-{gnd_segs[-1]['step']}, {len(gnd_segs)}스텝)")
        print(f"  → Up 전환 (step {up_segs[0]['step']}-{up_segs[-1]['step']}, {len(up_segs)}스텝)")
        print(f"  ────────────────────────────────────────────────────────")

        # ── 1. 시작점 비교 ──
        print(f"\n  [시작점 비교]")
        print(f"  {'step':>4} {'GT':>5} {'시작pY':>8} {'시작pZ':>8} {'끝pY':>8} {'끝pZ':>8}"
              f" {'궤적ΔpY':>8} {'궤적ΔpZ':>8} {'궤적방향°':>9}")

        compare_segs = gnd_last + up_all
        for seg in compare_segs:
            pY, pZ = seg['pY'], seg['pZ']
            dY = pY[-1] - pY[0]
            dZ = pZ[-1] - pZ[0]
            angle = math.degrees(math.atan2(dZ, dY))
            gt_s = seg['gt'][:3]
            marker = " ◀ 전환" if seg == up_all[0] else ""
            print(f"  {seg['step']:>4} {gt_s:>5}"
                  f" {pY[0]:>+8.3f} {pZ[0]:>+8.3f}"
                  f" {pY[-1]:>+8.3f} {pZ[-1]:>+8.3f}"
                  f" {dY:>+8.3f} {dZ:>+8.3f}"
                  f" {angle:>+9.1f}°"
                  f"{marker}")

        # ── 2. 시작점 ASCII plot ──
        pts_gnd = [(s['pY'][0], s['pZ'][0]) for s in gnd_last]
        pts_up = [(s['pY'][0], s['pZ'][0]) for s in up_all]
        all_pts = pts_gnd + pts_up
        all_labels = ['G'] * len(pts_gnd) + ['U'] * len(pts_up)
        # 끝점도 표시
        ends_gnd = [(s['pY'][-1], s['pZ'][-1]) for s in gnd_last]
        ends_up = [(s['pY'][-1], s['pZ'][-1]) for s in up_all]
        all_pts += ends_gnd + ends_up
        all_labels += ['g'] * len(ends_gnd) + ['u'] * len(ends_up)

        print(f"\n  시작점(대문자) + 끝점(소문자):")
        ascii_2d(all_pts, all_labels, title="G/g=ground  U/u=up")

        # ── 3. 시작점 군집도 수치 ──
        gnd_spy = [s['pY'][0] for s in gnd_last]
        gnd_spz = [s['pZ'][0] for s in gnd_last]
        up_spy = [s['pY'][0] for s in up_all]
        up_spz = [s['pZ'][0] for s in up_all]

        print(f"\n  시작점 군집도:")
        print(f"    Ground(마지막{len(gnd_last)}): pY std={std_val(gnd_spy):.3f} pZ std={std_val(gnd_spz):.3f}")
        print(f"    Up({len(up_all)}):            pY std={std_val(up_spy):.3f} pZ std={std_val(up_spz):.3f}")
        if std_val(up_spy) < std_val(gnd_spy):
            print(f"    → Up의 pY 시작점이 더 모여있음 ✓")
        if std_val(up_spz) < std_val(gnd_spz):
            print(f"    → Up의 pZ 시작점이 더 모여있음 ✓")

        # ── 4. 궤적 이동 방향 비교 ──
        print(f"\n  [궤적 이동 방향 분석]")

        # 각 스텝의 (시작→1/4, 1/4→1/2, 1/2→3/4, 3/4→끝) 방향
        for label, segs_group in [("Ground 마지막", gnd_last), ("Up", up_all)]:
            angles_first_half = []  # 시작→중간 방향
            angles_second_half = []  # 중간→끝 방향
            angles_net = []  # 순이동 방향
            pz_peaks = []  # pZ 최대 시점

            for seg in segs_group:
                pY, pZ = seg['pY'], seg['pZ']
                n = len(pY)
                mid = n // 2

                # 전반부 이동 방향
                dy1 = pY[mid] - pY[0]
                dz1 = pZ[mid] - pZ[0]
                angles_first_half.append(math.degrees(math.atan2(dz1, dy1)))

                # 후반부 이동 방향
                dy2 = pY[-1] - pY[mid]
                dz2 = pZ[-1] - pZ[mid]
                angles_second_half.append(math.degrees(math.atan2(dz2, dy2)))

                # 순이동
                dy = pY[-1] - pY[0]
                dz = pZ[-1] - pZ[0]
                angles_net.append(math.degrees(math.atan2(dz, dy)))

                # pZ 피크 위치
                peak_pos = pZ.index(max(pZ)) / (n-1)
                pz_peaks.append(peak_pos)

            print(f"\n    {label} ({len(segs_group)}스텝):")
            print(f"      전반부 방향(°): 각 스텝 = [{', '.join(f'{a:+.0f}' for a in angles_first_half)}]")
            print(f"        → 평균: {mean(angles_first_half):+.1f}°  std: {std_val(angles_first_half):.1f}°")
            print(f"      후반부 방향(°): 각 스텝 = [{', '.join(f'{a:+.0f}' for a in angles_second_half)}]")
            print(f"        → 평균: {mean(angles_second_half):+.1f}°  std: {std_val(angles_second_half):.1f}°")
            print(f"      순이동 방향(°): 각 스텝 = [{', '.join(f'{a:+.0f}' for a in angles_net)}]")
            print(f"        → 평균: {mean(angles_net):+.1f}°  std: {std_val(angles_net):.1f}°")
            print(f"      pZ피크 위치:  각 스텝 = [{', '.join(f'{p:.2f}' for p in pz_peaks)}]")
            print(f"        → 평균: {mean(pz_peaks):.2f}  std: {std_val(pz_peaks):.2f}")

        # ── 5. 스텝간 시작점 이동 벡터 ──
        print(f"\n  [스텝간 시작점 이동 벡터]")
        for label, segs_group in [("Ground 마지막", gnd_last), ("Up", up_all)]:
            if len(segs_group) < 2: continue
            move_dY, move_dZ, move_angles, move_dists = [], [], [], []
            for i in range(1, len(segs_group)):
                dy = segs_group[i]['pY'][0] - segs_group[i-1]['pY'][0]
                dz = segs_group[i]['pZ'][0] - segs_group[i-1]['pZ'][0]
                move_dY.append(dy)
                move_dZ.append(dz)
                move_angles.append(math.degrees(math.atan2(dz, dy)))
                move_dists.append(math.sqrt(dy*dy + dz*dz))

            print(f"\n    {label}:")
            print(f"      이동 ΔpY: [{', '.join(f'{d:+.2f}' for d in move_dY)}]")
            print(f"      이동 ΔpZ: [{', '.join(f'{d:+.2f}' for d in move_dZ)}]")
            print(f"      이동 방향°: [{', '.join(f'{a:+.0f}' for a in move_angles)}]")
            print(f"      이동 거리: [{', '.join(f'{d:.2f}' for d in move_dists)}]")
            print(f"      평균: ΔpY={mean(move_dY):+.3f} ΔpZ={mean(move_dZ):+.3f} 거리={mean(move_dists):.3f}")

            # pY 부호 번갈아 변화 정도
            py_sign_changes = sum(1 for i in range(len(move_dY)-1) if move_dY[i]*move_dY[i+1] < 0)
            pz_sign_changes = sum(1 for i in range(len(move_dZ)-1) if move_dZ[i]*move_dZ[i+1] < 0)
            if len(move_dY) >= 2:
                print(f"      ΔpY 부호 교번: {py_sign_changes}/{len(move_dY)-1}")
                print(f"      ΔpZ 부호 교번: {pz_sign_changes}/{len(move_dZ)-1}")

        # ── 6. 궤적 형태: 정규화 후 비교 ──
        print(f"\n  [정규화 궤적 형태 비교 (10-point)]")
        for label, segs_group in [("Ground 마지막", gnd_last), ("Up", up_all)]:
            all_norm_pY, all_norm_pZ = [], []
            for seg in segs_group:
                pY, pZ = seg['pY'], seg['pZ']
                n = len(pY)
                # 10-point resample
                rY, rZ = [], []
                for k in range(10):
                    t = k / 9 * (n-1)
                    idx = int(t); frac = t - idx
                    if idx >= n-1:
                        rY.append(pY[-1]); rZ.append(pZ[-1])
                    else:
                        rY.append(pY[idx]*(1-frac)+pY[idx+1]*frac)
                        rZ.append(pZ[idx]*(1-frac)+pZ[idx+1]*frac)
                # 시작점 기준 정규화 (시작=0)
                rY = [y - rY[0] for y in rY]
                rZ = [z - rZ[0] for z in rZ]
                all_norm_pY.append(rY)
                all_norm_pZ.append(rZ)

            avg_pY = [mean([t[i] for t in all_norm_pY]) for i in range(10)]
            avg_pZ = [mean([t[i] for t in all_norm_pZ]) for i in range(10)]

            print(f"\n    {label} 평균 궤적 (시작점=0,0 기준):")
            print(f"    시간%  평균pY    평균pZ")
            for i in range(10):
                pct = i / 9 * 100
                print(f"    {pct:5.1f}  {avg_pY[i]:>+7.3f}  {avg_pZ[i]:>+7.3f}")

            # ASCII plot
            pts = [(avg_pY[i], avg_pZ[i]) for i in range(10)]
            lbls = [str(i) for i in range(10)]
            ascii_2d(pts, lbls, w=45, h=12, title=f"{label} 평균 궤적 (시작=0,0)")


print(f"\n\n분석 완료.")
