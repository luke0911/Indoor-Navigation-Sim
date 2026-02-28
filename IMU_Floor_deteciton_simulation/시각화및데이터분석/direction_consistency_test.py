#!/usr/bin/env python3
"""
궤적 방향 일관성(Direction Consistency) 피처 + zEnergyFrac 결합 테스트

피처 정의:
- 각 스텝의 전반부 방향: atan2(pZ_mid - pZ_start, pY_mid - pY_start)
- 최근 N스텝의 방향 일관성 = Mean Resultant Length (R)
  R = sqrt(mean(cos)² + mean(sin)²),  R∈[0,1]
  R≈1 → 방향 일관적 (계단), R≈0 → 방향 랜덤 (평지)
- zEnergyFrac = var(pZ) / (var(pY) + var(pZ))

결합 규칙: dirConsistency > thr_R AND zEnergyFrac > thr_Z → stair
"""

import os, math, statistics

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output")
FILES = [
    ("일반보행", os.path.join(DATA_DIR, "batch_Floor_sim_data_일반보행.tsv")),
    ("살살걸음", os.path.join(DATA_DIR, "batch_Floor_sim_data_살살걸음.tsv")),
    ("거칠게걸음", os.path.join(DATA_DIR, "batch_Floor_sim_data_거칠게 걸음.tsv")),
]

WINDOW_SIZES = [3, 5, 7]


# ── pure-python helpers ──

def _mean(v):
    return sum(v) / len(v) if v else 0.0

def _std(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))

def _median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def _var(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return sum((x - m) ** 2 for x in v) / len(v)


def load_batch(path):
    """배치 TSV → per-step 데이터 추출
    stepIdx는 isStep=1인 행에서만 설정, 중간은 -1.
    → isStep=1 ~ 다음 isStep=1 사이의 모든 샘플을 한 스텝으로 수집."""
    rows = []
    with open(path, encoding="utf-8") as f:
        hdr = f.readline().strip().split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < len(hdr):
                continue
            rows.append(cols)

    steps = []
    cur_step_idx = -1
    cur_samples = []
    cur_gt = ""
    collecting = False

    for cols in rows:
        is_step = int(cols[ci["isStep"]])
        step_idx = int(cols[ci["stepIdx"]])
        py = float(cols[ci["pureY"]])
        pz = float(cols[ci["pureZ"]])
        gt = cols[ci["floorGT"]]

        if is_step and step_idx >= 0:
            # 이전 스텝 저장
            if collecting and cur_samples:
                steps.append({
                    "idx": cur_step_idx,
                    "gt": cur_gt,
                    "samples": cur_samples[:],
                })
            # 새 스텝 시작
            cur_step_idx = step_idx
            cur_gt = gt
            cur_samples = [(py, pz)]
            collecting = True
        elif collecting:
            # 스텝 사이 샘플 계속 수집
            cur_samples.append((py, pz))

    # 마지막 스텝
    if collecting and cur_samples:
        steps.append({
            "idx": cur_step_idx,
            "gt": cur_gt,
            "samples": cur_samples[:],
        })

    return steps


def compute_first_half_angle(samples):
    if len(samples) < 4:
        return None
    mid = len(samples) // 2
    py0, pz0 = samples[0]
    py_m, pz_m = samples[mid]
    dy = py_m - py0
    dz = pz_m - pz0
    if abs(dy) < 1e-9 and abs(dz) < 1e-9:
        return None
    return math.atan2(dz, dy)


def compute_second_half_angle(samples):
    if len(samples) < 4:
        return None
    mid = len(samples) // 2
    py_m, pz_m = samples[mid]
    py_e, pz_e = samples[-1]
    dy = py_e - py_m
    dz = pz_e - pz_m
    if abs(dy) < 1e-9 and abs(dz) < 1e-9:
        return None
    return math.atan2(dz, dy)


def mean_resultant_length(angles):
    """Mean Resultant Length: 방향 일관성 [0,1]"""
    if not angles:
        return 0.0
    n = len(angles)
    cs = sum(math.cos(a) for a in angles) / n
    sn = sum(math.sin(a) for a in angles) / n
    return math.sqrt(cs * cs + sn * sn)


def z_energy_frac_variance(samples):
    """var(pZ) / (var(pY) + var(pZ))"""
    if len(samples) < 3:
        return 0.5
    pys = [s[0] for s in samples]
    pzs = [s[1] for s in samples]
    vy = _var(pys)
    vz = _var(pzs)
    denom = vy + vz
    if denom < 1e-9:
        return 0.5
    return vz / denom


def evaluate(labels, preds):
    tp = fp = fn = tn = 0
    for gt, pr in zip(labels, preds):
        is_stair = (gt != "ground")
        pred_stair = (pr == "stair")
        if is_stair and pred_stair:
            tp += 1
        elif not is_stair and pred_stair:
            fp += 1
        elif is_stair and not pred_stair:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "prec": prec, "rec": rec, "f1": f1}


def _frange(start, stop, step):
    v = start
    while v < stop - 1e-9:
        yield round(v, 4)
        v += step


def main():
    print("=" * 80)
    print("  궤적 방향 일관성 + zEnergyFrac 결합 테스트")
    print("=" * 80)

    all_steps_data = []

    for name, path in FILES:
        if not os.path.exists(path):
            print(f"  !! {name}: 파일 없음")
            continue

        steps = load_batch(path)
        steps = [s for s in steps if s["gt"] != "down"]
        print(f"\n  {name}: {len(steps)} steps (ground+up only)")

        for s in steps:
            s["angle1"] = compute_first_half_angle(s["samples"])
            s["angle2"] = compute_second_half_angle(s["samples"])
            s["zef_var"] = z_energy_frac_variance(s["samples"])

        all_steps_data.append((name, steps))

    # ──────────────────────────────────────────
    # 1. 단독 피처 분포 확인
    # ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  1. 단독 피처 분포 (ground vs up)")
    print("=" * 80)

    for name, steps in all_steps_data:
        print(f"\n  [{name}]")
        for win in WINDOW_SIZES:
            gnd_r1, up_r1 = [], []
            gnd_r2, up_r2 = [], []

            for i, s in enumerate(steps):
                if i < win:
                    continue
                angles1 = [steps[j]["angle1"] for j in range(i - win + 1, i + 1)
                           if steps[j]["angle1"] is not None]
                angles2 = [steps[j]["angle2"] for j in range(i - win + 1, i + 1)
                           if steps[j]["angle2"] is not None]

                if len(angles1) >= win - 1:
                    r1 = mean_resultant_length(angles1)
                    (up_r1 if s["gt"] != "ground" else gnd_r1).append(r1)

                if len(angles2) >= win - 1:
                    r2 = mean_resultant_length(angles2)
                    (up_r2 if s["gt"] != "ground" else gnd_r2).append(r2)

            if gnd_r1 and up_r1:
                print(f"    Window={win} 전반부 R:")
                print(f"      Ground: mean={_mean(gnd_r1):.3f} std={_std(gnd_r1):.3f} "
                      f"med={_median(gnd_r1):.3f} [{min(gnd_r1):.3f}~{max(gnd_r1):.3f}]")
                print(f"      Up:     mean={_mean(up_r1):.3f} std={_std(up_r1):.3f} "
                      f"med={_median(up_r1):.3f} [{min(up_r1):.3f}~{max(up_r1):.3f}]")

            if gnd_r2 and up_r2:
                print(f"    Window={win} 후반부 R:")
                print(f"      Ground: mean={_mean(gnd_r2):.3f} std={_std(gnd_r2):.3f} "
                      f"med={_median(gnd_r2):.3f} [{min(gnd_r2):.3f}~{max(gnd_r2):.3f}]")
                print(f"      Up:     mean={_mean(up_r2):.3f} std={_std(up_r2):.3f} "
                      f"med={_median(up_r2):.3f} [{min(up_r2):.3f}~{max(up_r2):.3f}]")

    # ──────────────────────────────────────────
    # 2. zEnergyFrac(variance 기반) 분포
    # ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  2. zEnergyFrac (variance 기반) 분포")
    print("=" * 80)
    for name, steps in all_steps_data:
        gnd_zef = [s["zef_var"] for s in steps if s["gt"] == "ground"]
        up_zef = [s["zef_var"] for s in steps if s["gt"] != "ground"]
        print(f"\n  [{name}]")
        if gnd_zef:
            print(f"    Ground: mean={_mean(gnd_zef):.3f} std={_std(gnd_zef):.3f} "
                  f"med={_median(gnd_zef):.3f} [{min(gnd_zef):.3f}~{max(gnd_zef):.3f}]")
        if up_zef:
            print(f"    Up:     mean={_mean(up_zef):.3f} std={_std(up_zef):.3f} "
                  f"med={_median(up_zef):.3f} [{min(up_zef):.3f}~{max(up_zef):.3f}]")

    # ──────────────────────────────────────────
    # 3. 결합 테스트: 그리드 서치
    # ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  3. 결합 피처 그리드 서치 (모든 데이터셋 통합)")
    print("=" * 80)

    for win in WINDOW_SIZES:
        all_labels, all_r1, all_r2, all_zef = [], [], [], []

        for name, steps in all_steps_data:
            for i, s in enumerate(steps):
                if i < win:
                    continue
                a1 = [steps[j]["angle1"] for j in range(i - win + 1, i + 1)
                      if steps[j]["angle1"] is not None]
                a2 = [steps[j]["angle2"] for j in range(i - win + 1, i + 1)
                      if steps[j]["angle2"] is not None]

                r1 = mean_resultant_length(a1) if len(a1) >= win - 1 else 0.5
                r2 = mean_resultant_length(a2) if len(a2) >= win - 1 else 0.5

                all_labels.append(s["gt"])
                all_r1.append(r1)
                all_r2.append(r2)
                all_zef.append(s["zef_var"])

        n_gnd = sum(1 for l in all_labels if l == "ground")
        n_up = len(all_labels) - n_gnd
        print(f"\n  Window={win}: ground={n_gnd} up={n_up}")

        # --- 3a. dirR1 단독 ---
        best_r1_f1, best_r1_thr, best_r1_res = 0, 0, {}
        for thr in _frange(0.30, 0.98, 0.02):
            preds = ["stair" if r > thr else "ground" for r in all_r1]
            res = evaluate(all_labels, preds)
            if res["f1"] > best_r1_f1:
                best_r1_f1, best_r1_thr, best_r1_res = res["f1"], thr, res

        print(f"    dirR1 단독 best: thr={best_r1_thr:.2f} -> "
              f"F1={best_r1_res['f1']:.3f} P={best_r1_res['prec']:.3f} R={best_r1_res['rec']:.3f}")

        # --- 3b. dirR2 단독 ---
        best_r2_f1, best_r2_thr, best_r2_res = 0, 0, {}
        for thr in _frange(0.30, 0.98, 0.02):
            preds = ["stair" if r > thr else "ground" for r in all_r2]
            res = evaluate(all_labels, preds)
            if res["f1"] > best_r2_f1:
                best_r2_f1, best_r2_thr, best_r2_res = res["f1"], thr, res

        print(f"    dirR2 단독 best: thr={best_r2_thr:.2f} -> "
              f"F1={best_r2_res['f1']:.3f} P={best_r2_res['prec']:.3f} R={best_r2_res['rec']:.3f}")

        # --- 3c. zEF 단독 ---
        best_zef_f1, best_zef_thr, best_zef_res = 0, 0, {}
        for thr in _frange(0.40, 0.98, 0.02):
            preds = ["stair" if z > thr else "ground" for z in all_zef]
            res = evaluate(all_labels, preds)
            if res["f1"] > best_zef_f1:
                best_zef_f1, best_zef_thr, best_zef_res = res["f1"], thr, res

        print(f"    zEF_var 단독 best: thr={best_zef_thr:.2f} -> "
              f"F1={best_zef_res['f1']:.3f} P={best_zef_res['prec']:.3f} R={best_zef_res['rec']:.3f}")

        # --- 3d. dirR1 AND zEF ---
        best_combo_f1, best_combo = 0, None
        for thr_r in _frange(0.30, 0.96, 0.02):
            for thr_z in _frange(0.40, 0.96, 0.02):
                preds = ["stair" if (all_r1[i] > thr_r and all_zef[i] > thr_z) else "ground"
                         for i in range(len(all_labels))]
                res = evaluate(all_labels, preds)
                if res["f1"] > best_combo_f1:
                    best_combo_f1 = res["f1"]
                    best_combo = (thr_r, thr_z, res)

        if best_combo:
            r, z, res = best_combo
            print(f"    dirR1 AND zEF best: R>{r:.2f} & zEF>{z:.2f} -> "
                  f"F1={res['f1']:.3f} P={res['prec']:.3f} R={res['rec']:.3f} "
                  f"(TP={res['tp']} FP={res['fp']} FN={res['fn']})")

        # --- 3e. dirR2 AND zEF ---
        best_combo2_f1, best_combo2 = 0, None
        for thr_r in _frange(0.30, 0.96, 0.02):
            for thr_z in _frange(0.40, 0.96, 0.02):
                preds = ["stair" if (all_r2[i] > thr_r and all_zef[i] > thr_z) else "ground"
                         for i in range(len(all_labels))]
                res = evaluate(all_labels, preds)
                if res["f1"] > best_combo2_f1:
                    best_combo2_f1 = res["f1"]
                    best_combo2 = (thr_r, thr_z, res)

        if best_combo2:
            r, z, res = best_combo2
            print(f"    dirR2 AND zEF best: R>{r:.2f} & zEF>{z:.2f} -> "
                  f"F1={res['f1']:.3f} P={res['prec']:.3f} R={res['rec']:.3f} "
                  f"(TP={res['tp']} FP={res['fp']} FN={res['fn']})")

        # --- 3f. max(R1,R2) AND zEF ---
        best_combo3_f1, best_combo3 = 0, None
        all_r_max = [max(all_r1[i], all_r2[i]) for i in range(len(all_labels))]
        for thr_r in _frange(0.30, 0.96, 0.02):
            for thr_z in _frange(0.40, 0.96, 0.02):
                preds = ["stair" if (all_r_max[i] > thr_r and all_zef[i] > thr_z) else "ground"
                         for i in range(len(all_labels))]
                res = evaluate(all_labels, preds)
                if res["f1"] > best_combo3_f1:
                    best_combo3_f1 = res["f1"]
                    best_combo3 = (thr_r, thr_z, res)

        if best_combo3:
            r, z, res = best_combo3
            print(f"    max(R1,R2) AND zEF: R>{r:.2f} & zEF>{z:.2f} -> "
                  f"F1={res['f1']:.3f} P={res['prec']:.3f} R={res['rec']:.3f} "
                  f"(TP={res['tp']} FP={res['fp']} FN={res['fn']})")

        # --- 3g. R1*zEF 곱 스코어 ---
        all_score = [all_r1[i] * all_zef[i] for i in range(len(all_labels))]
        best_score_f1, best_score = 0, None
        for thr in _frange(0.20, 0.80, 0.01):
            preds = ["stair" if s > thr else "ground" for s in all_score]
            res = evaluate(all_labels, preds)
            if res["f1"] > best_score_f1:
                best_score_f1 = res["f1"]
                best_score = (thr, res)

        if best_score:
            t, res = best_score
            print(f"    R1*zEF score: thr>{t:.2f} -> "
                  f"F1={res['f1']:.3f} P={res['prec']:.3f} R={res['rec']:.3f}")

        # --- 3h. dirR1 OR zEF (둘 중 하나만 충족) ---
        best_or_f1, best_or = 0, None
        for thr_r in _frange(0.60, 0.96, 0.02):
            for thr_z in _frange(0.70, 0.96, 0.02):
                preds = ["stair" if (all_r1[i] > thr_r or all_zef[i] > thr_z) else "ground"
                         for i in range(len(all_labels))]
                res = evaluate(all_labels, preds)
                if res["f1"] > best_or_f1:
                    best_or_f1 = res["f1"]
                    best_or = (thr_r, thr_z, res)

        if best_or:
            r, z, res = best_or
            print(f"    dirR1 OR zEF best: R>{r:.2f} | zEF>{z:.2f} -> "
                  f"F1={res['f1']:.3f} P={res['prec']:.3f} R={res['rec']:.3f} "
                  f"(TP={res['tp']} FP={res['fp']} FN={res['fn']})")

    # ──────────────────────────────────────────
    # 4. 데이터셋별 성능 (window=5)
    # ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  4. 데이터셋별 성능 (window=5, 통합 best 임계값)")
    print("=" * 80)

    win = 5
    all_labels_all, all_r1_all, all_zef_all = [], [], []
    per_dataset = {}

    for name, steps in all_steps_data:
        labels, r1s, zefs = [], [], []
        for i, s in enumerate(steps):
            if i < win:
                continue
            a1 = [steps[j]["angle1"] for j in range(i - win + 1, i + 1)
                  if steps[j]["angle1"] is not None]
            r1 = mean_resultant_length(a1) if len(a1) >= win - 1 else 0.5
            labels.append(s["gt"])
            r1s.append(r1)
            zefs.append(s["zef_var"])

        per_dataset[name] = (labels, r1s, zefs)
        all_labels_all.extend(labels)
        all_r1_all.extend(r1s)
        all_zef_all.extend(zefs)

    # 통합 best 찾기 (AND)
    best_f1, best_params = 0, (0, 0)
    for thr_r in _frange(0.30, 0.96, 0.02):
        for thr_z in _frange(0.40, 0.96, 0.02):
            preds = ["stair" if (all_r1_all[i] > thr_r and all_zef_all[i] > thr_z) else "ground"
                     for i in range(len(all_labels_all))]
            res = evaluate(all_labels_all, preds)
            if res["f1"] > best_f1:
                best_f1 = res["f1"]
                best_params = (thr_r, thr_z)

    thr_r, thr_z = best_params
    print(f"\n  통합 best AND 임계값: dirR1>{thr_r:.2f} & zEF>{thr_z:.2f} (F1={best_f1:.3f})")

    for name in per_dataset:
        labels, r1s, zefs = per_dataset[name]
        preds = ["stair" if (r1s[i] > thr_r and zefs[i] > thr_z) else "ground"
                 for i in range(len(labels))]
        res = evaluate(labels, preds)
        n_gnd = sum(1 for l in labels if l == "ground")
        n_up = len(labels) - n_gnd
        print(f"    {name}: F1={res['f1']:.3f} P={res['prec']:.3f} R={res['rec']:.3f} "
              f"(TP={res['tp']} FP={res['fp']} FN={res['fn']}) [gnd={n_gnd} up={n_up}]")

    # 통합 best 찾기 (OR)
    best_f1_or, best_params_or = 0, (0, 0)
    for thr_r in _frange(0.60, 0.96, 0.02):
        for thr_z in _frange(0.70, 0.96, 0.02):
            preds = ["stair" if (all_r1_all[i] > thr_r or all_zef_all[i] > thr_z) else "ground"
                     for i in range(len(all_labels_all))]
            res = evaluate(all_labels_all, preds)
            if res["f1"] > best_f1_or:
                best_f1_or = res["f1"]
                best_params_or = (thr_r, thr_z)

    thr_r_or, thr_z_or = best_params_or
    print(f"\n  통합 best OR 임계값: dirR1>{thr_r_or:.2f} | zEF>{thr_z_or:.2f} (F1={best_f1_or:.3f})")

    for name in per_dataset:
        labels, r1s, zefs = per_dataset[name]
        preds = ["stair" if (r1s[i] > thr_r_or or zefs[i] > thr_z_or) else "ground"
                 for i in range(len(labels))]
        res = evaluate(labels, preds)
        n_gnd = sum(1 for l in labels if l == "ground")
        n_up = len(labels) - n_gnd
        print(f"    {name}: F1={res['f1']:.3f} P={res['prec']:.3f} R={res['rec']:.3f} "
              f"(TP={res['tp']} FP={res['fp']} FN={res['fn']}) [gnd={n_gnd} up={n_up}]")

    # ──────────────────────────────────────────
    # 5. 시계열 전환구간 상세 (best AND 임계값)
    # ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  5. 시계열 전환 구간 상세 (win=5, R>{thr_r:.2f} & zEF>{thr_z:.2f})")
    print("=" * 80)

    for name in per_dataset:
        print(f"\n  [{name}]")
        labels, r1s, zefs = per_dataset[name]

        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                lo = max(0, i - 5)
                hi = min(len(labels), i + 6)
                print(f"\n    전환: step {lo+win}~{hi+win-1} ({labels[i-1]}->{labels[i]})")
                print(f"    {'step':>6} {'GT':>6} {'dirR1':>7} {'zEF':>7} {'pred':>6}")
                for j in range(lo, hi):
                    pred = "stair" if (r1s[j] > thr_r and zefs[j] > thr_z) else "ground"
                    marker = " <-" if j == i else ""
                    ok = "O" if (pred == "stair") == (labels[j] != "ground") else "X"
                    print(f"    {j+win:6d} {labels[j]:>6} {r1s[j]:7.3f} {zefs[j]:7.3f} {pred:>6} {ok}{marker}")


if __name__ == "__main__":
    main()
