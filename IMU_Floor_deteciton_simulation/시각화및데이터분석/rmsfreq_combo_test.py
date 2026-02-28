#!/usr/bin/env python3
"""
rmsZ/freq + dirR1 + zEF 결합 테스트

기존 batch TSV에서 per-step 피처를 뽑아서:
1. rmsZdivFreq 단독
2. procRmsZ 단독
3. rmsZdivFreq + dirR1 결합
4. rmsZdivFreq + zEF 결합
5. rmsZdivFreq + dirR1 + zEF 3피처 결합
각각의 F1 성능 비교.
"""

import os, math

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output")
FILES = [
    ("일반보행", os.path.join(DATA_DIR, "batch_Floor_sim_data_일반보행.tsv")),
    ("살살걸음", os.path.join(DATA_DIR, "batch_Floor_sim_data_살살걸음.tsv")),
    ("거칠게걸음", os.path.join(DATA_DIR, "batch_Floor_sim_data_거칠게 걸음.tsv")),
]

DIR_WINDOW = 5


def _mean(v): return sum(v) / len(v) if v else 0.0
def _std(v):
    if len(v) < 2: return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
def _var(v):
    if len(v) < 2: return 0.0
    m = _mean(v)
    return sum((x - m) ** 2 for x in v) / len(v)
def _frange(start, stop, step):
    v = start
    while v < stop - 1e-9:
        yield round(v, 4)
        v += step


def load_steps(path):
    """배치 TSV → per-step 피처 추출 (isStep=1 행에서 피처, 사이 샘플로 궤적)"""
    with open(path, encoding="utf-8") as f:
        hdr = f.readline().strip().split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        rows = []
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) >= len(hdr):
                rows.append(cols)

    steps = []
    cur_idx = -1
    cur_traj = []
    cur_feat = {}

    for cols in rows:
        is_step = int(cols[ci["isStep"]])
        step_idx = int(cols[ci["stepIdx"]])
        py = float(cols[ci["pureY"]])
        pz = float(cols[ci["pureZ"]])

        if is_step and step_idx >= 0:
            # 이전 스텝 저장
            if cur_traj and cur_idx >= 0:
                cur_feat["traj"] = cur_traj[:]
                steps.append(cur_feat)

            # 새 스텝 시작 — isStep=1 행에서 피처 읽기
            cur_idx = step_idx
            cur_feat = {
                "idx": step_idx,
                "gt": cols[ci["floorGT"]],
                "rmsZdivFreq": float(cols[ci["rmsZdivFreq"]]),
                "procRmsZ": float(cols[ci["procRmsZ"]]),
                "stepFreqHz": float(cols[ci["stepFreqHz"]]),
            }
            cur_traj = [(py, pz)]
        elif cur_idx >= 0:
            cur_traj.append((py, pz))

    if cur_traj and cur_idx >= 0:
        cur_feat["traj"] = cur_traj[:]
        steps.append(cur_feat)

    return steps


def first_half_angle(traj):
    if len(traj) < 4: return None
    mid = len(traj) // 2
    dy = traj[mid][0] - traj[0][0]
    dz = traj[mid][1] - traj[0][1]
    if abs(dy) < 1e-9 and abs(dz) < 1e-9: return None
    return math.atan2(dz, dy)


def mean_resultant_length(angles):
    if not angles: return 0.0
    n = len(angles)
    cs = sum(math.cos(a) for a in angles) / n
    sn = sum(math.sin(a) for a in angles) / n
    return math.sqrt(cs * cs + sn * sn)


def z_energy_frac(traj):
    if len(traj) < 3: return 0.5
    pys = [t[0] for t in traj]
    pzs = [t[1] for t in traj]
    vy, vz = _var(pys), _var(pzs)
    d = vy + vz
    return vz / d if d > 1e-9 else 0.5


def evaluate(labels, preds):
    tp = fp = fn = tn = 0
    for gt, pr in zip(labels, preds):
        s = (gt != "ground")
        p = (pr == "stair")
        if s and p: tp += 1
        elif not s and p: fp += 1
        elif s and not p: fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "prec": prec, "rec": rec, "f1": f1}


def main():
    print("=" * 85)
    print("  rmsZ/freq + dirR1 + zEF 결합 테스트 (ground vs up only)")
    print("=" * 85)

    all_steps_data = []
    for name, path in FILES:
        if not os.path.exists(path): continue
        steps = load_steps(path)
        steps = [s for s in steps if s["gt"] != "down"]
        for s in steps:
            s["angle1"] = first_half_angle(s["traj"])
            s["zef"] = z_energy_frac(s["traj"])
        all_steps_data.append((name, steps))
        print(f"  {name}: {len(steps)} steps")

    # ── 1. 피처 분포 ──
    print("\n" + "=" * 85)
    print("  1. 피처 분포 (ground vs up)")
    print("=" * 85)

    for name, steps in all_steps_data:
        gnd = [s for s in steps if s["gt"] == "ground"]
        up = [s for s in steps if s["gt"] != "ground"]
        print(f"\n  [{name}] ground={len(gnd)} up={len(up)}")

        for feat, label in [("rmsZdivFreq", "rmsZ/f"),
                            ("procRmsZ", "rmsZ"),
                            ("stepFreqHz", "freq"),
                            ("zef", "zEF")]:
            gv = [s[feat] for s in gnd]
            uv = [s[feat] for s in up]
            diff = _mean(uv) - _mean(gv)
            rev = " *** REV" if diff < 0 and feat in ("rmsZdivFreq", "procRmsZ", "zef") else ""
            print(f"    {label:>8}: Gnd={_mean(gv):.3f}({_std(gv):.3f})  "
                  f"Up={_mean(uv):.3f}({_std(uv):.3f})  diff={diff:+.3f}{rev}")

    # ── 2. 통합 그리드 서치 ──
    print("\n" + "=" * 85)
    print("  2. 통합 그리드 서치")
    print("=" * 85)

    # rolling dirR1 계산 & 통합 배열
    all_l, all_rzf, all_rmsz, all_freq, all_r1, all_zef = [], [], [], [], [], []
    per_ds = {}

    for name, steps in all_steps_data:
        l, rzf, rmsz, freq, r1, zef = [], [], [], [], [], []
        for i, s in enumerate(steps):
            if i < DIR_WINDOW: continue
            angles = [steps[j]["angle1"] for j in range(i - DIR_WINDOW + 1, i + 1)
                      if steps[j]["angle1"] is not None]
            rv = mean_resultant_length(angles) if len(angles) >= DIR_WINDOW - 1 else 0.5
            l.append(s["gt"])
            rzf.append(s["rmsZdivFreq"])
            rmsz.append(s["procRmsZ"])
            freq.append(s["stepFreqHz"])
            r1.append(rv)
            zef.append(s["zef"])
        per_ds[name] = (l, rzf, rmsz, freq, r1, zef)
        all_l.extend(l); all_rzf.extend(rzf); all_rmsz.extend(rmsz)
        all_freq.extend(freq); all_r1.extend(r1); all_zef.extend(zef)

    N = len(all_l)
    n_gnd = sum(1 for x in all_l if x == "ground")
    n_up = N - n_gnd
    print(f"\n  총 스텝: ground={n_gnd} up={n_up}")

    # --- 2a. rmsZ/freq 단독 ---
    best_f1, best = 0, None
    for thr in _frange(0.3, 3.0, 0.05):
        preds = ["stair" if all_rzf[i] > thr else "ground" for i in range(N)]
        res = evaluate(all_l, preds)
        if res["f1"] > best_f1: best_f1, best = res["f1"], (thr, res)
    t, r = best
    print(f"\n    rmsZ/freq 단독: thr>{t:.2f} -> F1={r['f1']:.3f} P={r['prec']:.3f} R={r['rec']:.3f}")

    # --- 2b. procRmsZ 단독 ---
    best_f1, best = 0, None
    for thr in _frange(0.5, 5.0, 0.1):
        preds = ["stair" if all_rmsz[i] > thr else "ground" for i in range(N)]
        res = evaluate(all_l, preds)
        if res["f1"] > best_f1: best_f1, best = res["f1"], (thr, res)
    t, r = best
    print(f"    procRmsZ 단독:  thr>{t:.2f} -> F1={r['f1']:.3f} P={r['prec']:.3f} R={r['rec']:.3f}")

    # --- 2c. dirR1 단독 ---
    best_f1, best = 0, None
    for thr in _frange(0.3, 0.98, 0.02):
        preds = ["stair" if all_r1[i] > thr else "ground" for i in range(N)]
        res = evaluate(all_l, preds)
        if res["f1"] > best_f1: best_f1, best = res["f1"], (thr, res)
    t, r = best
    print(f"    dirR1 단독:     thr>{t:.2f} -> F1={r['f1']:.3f} P={r['prec']:.3f} R={r['rec']:.3f}")

    # --- 2d. zEF 단독 ---
    best_f1, best = 0, None
    for thr in _frange(0.4, 0.98, 0.02):
        preds = ["stair" if all_zef[i] > thr else "ground" for i in range(N)]
        res = evaluate(all_l, preds)
        if res["f1"] > best_f1: best_f1, best = res["f1"], (thr, res)
    t, r = best
    print(f"    zEF 단독:       thr>{t:.2f} -> F1={r['f1']:.3f} P={r['prec']:.3f} R={r['rec']:.3f}")

    # --- 2e. rmsZ/freq AND dirR1 ---
    best_f1, best = 0, None
    for t1 in _frange(0.3, 3.0, 0.1):
        for t2 in _frange(0.3, 0.96, 0.02):
            preds = ["stair" if (all_rzf[i] > t1 and all_r1[i] > t2) else "ground" for i in range(N)]
            res = evaluate(all_l, preds)
            if res["f1"] > best_f1: best_f1, best = res["f1"], (t1, t2, res)
    t1, t2, r = best
    print(f"\n    rmsZ/f AND dirR1: rzf>{t1:.2f} & R>{t2:.2f} -> F1={r['f1']:.3f} P={r['prec']:.3f} R={r['rec']:.3f} "
          f"(TP={r['tp']} FP={r['fp']} FN={r['fn']})")

    # --- 2f. rmsZ/freq AND zEF ---
    best_f1, best = 0, None
    for t1 in _frange(0.3, 3.0, 0.1):
        for t2 in _frange(0.4, 0.96, 0.02):
            preds = ["stair" if (all_rzf[i] > t1 and all_zef[i] > t2) else "ground" for i in range(N)]
            res = evaluate(all_l, preds)
            if res["f1"] > best_f1: best_f1, best = res["f1"], (t1, t2, res)
    t1, t2, r = best
    print(f"    rmsZ/f AND zEF:   rzf>{t1:.2f} & zEF>{t2:.2f} -> F1={r['f1']:.3f} P={r['prec']:.3f} R={r['rec']:.3f} "
          f"(TP={r['tp']} FP={r['fp']} FN={r['fn']})")

    # --- 2g. rmsZ/freq AND dirR1 AND zEF ---
    best_f1, best = 0, None
    for t1 in _frange(0.3, 3.0, 0.1):
        for t2 in _frange(0.3, 0.96, 0.04):
            for t3 in _frange(0.4, 0.96, 0.04):
                preds = ["stair" if (all_rzf[i] > t1 and all_r1[i] > t2 and all_zef[i] > t3)
                         else "ground" for i in range(N)]
                res = evaluate(all_l, preds)
                if res["f1"] > best_f1: best_f1, best = res["f1"], (t1, t2, t3, res)
    t1, t2, t3, r = best
    print(f"    3피처 AND:        rzf>{t1:.2f} & R>{t2:.2f} & zEF>{t3:.2f} -> F1={r['f1']:.3f} "
          f"P={r['prec']:.3f} R={r['rec']:.3f} (TP={r['tp']} FP={r['fp']} FN={r['fn']})")

    # --- 2h. rmsZ/freq OR dirR1 (high thresholds) ---
    best_f1, best = 0, None
    for t1 in _frange(0.8, 3.0, 0.1):
        for t2 in _frange(0.80, 0.98, 0.02):
            preds = ["stair" if (all_rzf[i] > t1 or all_r1[i] > t2) else "ground" for i in range(N)]
            res = evaluate(all_l, preds)
            if res["f1"] > best_f1: best_f1, best = res["f1"], (t1, t2, res)
    t1, t2, r = best
    print(f"    rmsZ/f OR dirR1:  rzf>{t1:.2f} | R>{t2:.2f} -> F1={r['f1']:.3f} P={r['prec']:.3f} R={r['rec']:.3f} "
          f"(TP={r['tp']} FP={r['fp']} FN={r['fn']})")

    # ── 3. 데이터셋별 best combo 성능 ──
    print("\n" + "=" * 85)
    print("  3. 데이터셋별 성능")
    print("=" * 85)

    # 주요 combo들 정리
    combos = []

    # rmsZ/f AND dirR1
    best_f1, best = 0, None
    for t1 in _frange(0.3, 3.0, 0.1):
        for t2 in _frange(0.3, 0.96, 0.02):
            preds = ["stair" if (all_rzf[i] > t1 and all_r1[i] > t2) else "ground" for i in range(N)]
            res = evaluate(all_l, preds)
            if res["f1"] > best_f1: best_f1, best = res["f1"], (t1, t2)
    combos.append(("rmsZ/f & dirR1", lambda rzf, r1, zef, t=best: rzf > t[0] and r1 > t[1],
                    f"rzf>{best[0]:.2f} & R>{best[1]:.2f}"))

    # rmsZ/f AND zEF
    best_f1, best = 0, None
    for t1 in _frange(0.3, 3.0, 0.1):
        for t2 in _frange(0.4, 0.96, 0.02):
            preds = ["stair" if (all_rzf[i] > t1 and all_zef[i] > t2) else "ground" for i in range(N)]
            res = evaluate(all_l, preds)
            if res["f1"] > best_f1: best_f1, best = res["f1"], (t1, t2)
    combos.append(("rmsZ/f & zEF", lambda rzf, r1, zef, t=best: rzf > t[0] and zef > t[1],
                    f"rzf>{best[0]:.2f} & zEF>{best[1]:.2f}"))

    # dirR1 AND zEF (기존)
    best_f1, best = 0, None
    for t1 in _frange(0.3, 0.96, 0.02):
        for t2 in _frange(0.4, 0.96, 0.02):
            preds = ["stair" if (all_r1[i] > t1 and all_zef[i] > t2) else "ground" for i in range(N)]
            res = evaluate(all_l, preds)
            if res["f1"] > best_f1: best_f1, best = res["f1"], (t1, t2)
    combos.append(("dirR1 & zEF", lambda rzf, r1, zef, t=best: r1 > t[0] and zef > t[1],
                    f"R>{best[0]:.2f} & zEF>{best[1]:.2f}"))

    for combo_name, pred_fn, desc in combos:
        print(f"\n  {combo_name} ({desc}):")
        for name in per_ds:
            l, rzf, rmsz, freq, r1, zef = per_ds[name]
            preds = ["stair" if pred_fn(rzf[i], r1[i], zef[i]) else "ground" for i in range(len(l))]
            res = evaluate(l, preds)
            ng = sum(1 for x in l if x == "ground")
            nu = len(l) - ng
            print(f"    {name}: F1={res['f1']:.3f} P={res['prec']:.3f} R={res['rec']:.3f} "
                  f"(TP={res['tp']} FP={res['fp']} FN={res['fn']}) [gnd={ng} up={nu}]")

    # ── 4. 시계열 전환구간 (best combo) ──
    print("\n" + "=" * 85)
    print("  4. 시계열 전환 구간 상세 (rmsZ/f AND dirR1)")
    print("=" * 85)

    # best rmsZ/f AND dirR1 thresholds
    best_f1, best_t = 0, (0, 0)
    for t1 in _frange(0.3, 3.0, 0.1):
        for t2 in _frange(0.3, 0.96, 0.02):
            preds = ["stair" if (all_rzf[i] > t1 and all_r1[i] > t2) else "ground" for i in range(N)]
            res = evaluate(all_l, preds)
            if res["f1"] > best_f1: best_f1, best_t = res["f1"], (t1, t2)
    rzf_t, r1_t = best_t
    print(f"  임계값: rmsZ/f>{rzf_t:.2f} & dirR1>{r1_t:.2f}")

    for name in per_ds:
        print(f"\n  [{name}]")
        l, rzf, rmsz, freq, r1, zef = per_ds[name]
        for i in range(1, len(l)):
            if l[i] != l[i - 1]:
                lo = max(0, i - 5)
                hi = min(len(l), i + 6)
                print(f"\n    전환: step {lo+DIR_WINDOW}~{hi+DIR_WINDOW-1} ({l[i-1]}->{l[i]})")
                print(f"    {'step':>6} {'GT':>6} {'rmsZ/f':>7} {'dirR1':>7} {'zEF':>7} {'pred':>6}")
                for j in range(lo, hi):
                    pred = "stair" if (rzf[j] > rzf_t and r1[j] > r1_t) else "ground"
                    marker = " <-" if j == i else ""
                    ok = "O" if (pred == "stair") == (l[j] != "ground") else "X"
                    print(f"    {j+DIR_WINDOW:6d} {l[j]:>6} {rzf[j]:7.3f} {r1[j]:7.3f} "
                          f"{zef[j]:7.3f} {pred:>6} {ok}{marker}")


if __name__ == "__main__":
    main()
