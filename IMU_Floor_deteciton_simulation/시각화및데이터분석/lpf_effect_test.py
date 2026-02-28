#!/usr/bin/env python3
"""
LPF 적용 전/후 피처 비교 테스트

기존 batch TSV의 pureY/pureZ에 1차 IIR LPF를 적용하여
궤적 방향 일관성 + zEnergyFrac이 개선되는지 검증.

LPF: smoothed[n] = alpha * raw[n] + (1-alpha) * smoothed[n-1]
  alpha 크면 → raw에 가까움 (노이즈 많음)
  alpha 작으면 → 스무딩 강함 (지연 큼)

테스트 alpha: 0.1 (강), 0.2 (중), 0.3 (약), 1.0 (원본=LPF없음)
"""

import os, math

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output")
FILES = [
    ("일반보행", os.path.join(DATA_DIR, "batch_Floor_sim_data_일반보행.tsv")),
    ("살살걸음", os.path.join(DATA_DIR, "batch_Floor_sim_data_살살걸음.tsv")),
    ("거칠게걸음", os.path.join(DATA_DIR, "batch_Floor_sim_data_거칠게 걸음.tsv")),
]

LPF_ALPHAS = [1.0, 0.3, 0.2, 0.15, 0.1, 0.05]
DIR_WINDOW = 5


def _mean(v):
    return sum(v) / len(v) if v else 0.0

def _var(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return sum((x - m) ** 2 for x in v) / len(v)

def _std(v):
    return math.sqrt(_var(v))


def load_raw_batch(path):
    """배치 TSV → 전체 샘플 (time, pureY, pureZ, isStep, floorGT)"""
    samples = []
    with open(path, encoding="utf-8") as f:
        hdr = f.readline().strip().split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < len(hdr):
                continue
            samples.append({
                "py": float(cols[ci["pureY"]]),
                "pz": float(cols[ci["pureZ"]]),
                "isStep": int(cols[ci["isStep"]]),
                "stepIdx": int(cols[ci["stepIdx"]]),
                "gt": cols[ci["floorGT"]],
            })
    return samples


def apply_lpf(samples, alpha):
    """1차 IIR LPF를 pureY, pureZ에 적용"""
    if alpha >= 1.0:
        return [(s["py"], s["pz"]) for s in samples]

    result = []
    sy, sz = samples[0]["py"], samples[0]["pz"]
    for s in samples:
        sy = alpha * s["py"] + (1.0 - alpha) * sy
        sz = alpha * s["pz"] + (1.0 - alpha) * sz
        result.append((sy, sz))
    return result


def extract_steps(samples, smoothed):
    """isStep=1 기준으로 스텝별 궤적 추출"""
    steps = []
    cur_samples = []
    cur_gt = ""
    cur_idx = -1
    collecting = False

    for i, s in enumerate(samples):
        if s["isStep"] == 1 and s["stepIdx"] >= 0:
            if collecting and cur_samples:
                steps.append({"idx": cur_idx, "gt": cur_gt, "traj": cur_samples[:]})
            cur_idx = s["stepIdx"]
            cur_gt = s["gt"]
            cur_samples = [smoothed[i]]
            collecting = True
        elif collecting:
            cur_samples.append(smoothed[i])

    if collecting and cur_samples:
        steps.append({"idx": cur_idx, "gt": cur_gt, "traj": cur_samples[:]})

    return steps


def first_half_angle(traj):
    if len(traj) < 4:
        return None
    mid = len(traj) // 2
    dy = traj[mid][0] - traj[0][0]
    dz = traj[mid][1] - traj[0][1]
    if abs(dy) < 1e-9 and abs(dz) < 1e-9:
        return None
    return math.atan2(dz, dy)


def mean_resultant_length(angles):
    if not angles:
        return 0.0
    n = len(angles)
    cs = sum(math.cos(a) for a in angles) / n
    sn = sum(math.sin(a) for a in angles) / n
    return math.sqrt(cs * cs + sn * sn)


def z_energy_frac(traj):
    if len(traj) < 3:
        return 0.5
    pys = [t[0] for t in traj]
    pzs = [t[1] for t in traj]
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
    print("=" * 85)
    print("  LPF 적용 전/후 피처 비교 (ground vs up only)")
    print("=" * 85)

    # 데이터셋 로드
    all_raw = {}
    for name, path in FILES:
        if not os.path.exists(path):
            continue
        all_raw[name] = load_raw_batch(path)
        print(f"  {name}: {len(all_raw[name])} samples")

    print()

    for alpha in LPF_ALPHAS:
        alpha_label = f"alpha={alpha:.2f}" if alpha < 1.0 else "원본(LPF없음)"
        print("=" * 85)
        print(f"  LPF {alpha_label}")
        print("=" * 85)

        # 모든 데이터셋 통합 피처
        all_labels, all_r1, all_zef = [], [], []
        per_dataset = {}

        for name in all_raw:
            samples = all_raw[name]
            smoothed = apply_lpf(samples, alpha)
            steps = extract_steps(samples, smoothed)
            # down 제외
            steps = [s for s in steps if s["gt"] != "down"]

            # 피처 계산
            for s in steps:
                s["angle1"] = first_half_angle(s["traj"])
                s["zef"] = z_energy_frac(s["traj"])

            # rolling direction consistency
            labels, r1s, zefs = [], [], []
            for i, s in enumerate(steps):
                if i < DIR_WINDOW:
                    continue
                angles = [steps[j]["angle1"] for j in range(i - DIR_WINDOW + 1, i + 1)
                          if steps[j]["angle1"] is not None]
                r1 = mean_resultant_length(angles) if len(angles) >= DIR_WINDOW - 1 else 0.5
                labels.append(s["gt"])
                r1s.append(r1)
                zefs.append(s["zef"])

            per_dataset[name] = (labels, r1s, zefs)
            all_labels.extend(labels)
            all_r1.extend(r1s)
            all_zef.extend(zefs)

        n_gnd = sum(1 for l in all_labels if l == "ground")
        n_up = len(all_labels) - n_gnd

        # ── 피처 분포 ──
        for name in per_dataset:
            labels, r1s, zefs = per_dataset[name]
            gnd_r1 = [r1s[i] for i in range(len(labels)) if labels[i] == "ground"]
            up_r1 = [r1s[i] for i in range(len(labels)) if labels[i] != "ground"]
            gnd_zef = [zefs[i] for i in range(len(labels)) if labels[i] == "ground"]
            up_zef = [zefs[i] for i in range(len(labels)) if labels[i] != "ground"]

            if gnd_r1 and up_r1:
                diff_r = _mean(up_r1) - _mean(gnd_r1)
                print(f"  [{name}]")
                print(f"    dirR1: Gnd={_mean(gnd_r1):.3f}({_std(gnd_r1):.3f}) "
                      f"Up={_mean(up_r1):.3f}({_std(up_r1):.3f}) diff={diff_r:+.3f}")
            if gnd_zef and up_zef:
                diff_z = _mean(up_zef) - _mean(gnd_zef)
                rev = " *** REVERSED" if diff_z < 0 else ""
                print(f"    zEF:   Gnd={_mean(gnd_zef):.3f}({_std(gnd_zef):.3f}) "
                      f"Up={_mean(up_zef):.3f}({_std(up_zef):.3f}) diff={diff_z:+.3f}{rev}")

        # ── 결합 그리드 서치 ──
        # dirR1 단독
        best_r1_f1 = 0
        for thr in _frange(0.30, 0.98, 0.02):
            preds = ["stair" if r > thr else "ground" for r in all_r1]
            res = evaluate(all_labels, preds)
            if res["f1"] > best_r1_f1:
                best_r1_f1 = res["f1"]
                best_r1_res = res
                best_r1_thr = thr

        # zEF 단독
        best_zef_f1 = 0
        for thr in _frange(0.40, 0.98, 0.02):
            preds = ["stair" if z > thr else "ground" for z in all_zef]
            res = evaluate(all_labels, preds)
            if res["f1"] > best_zef_f1:
                best_zef_f1 = res["f1"]
                best_zef_res = res
                best_zef_thr = thr

        # AND 결합
        best_and_f1, best_and = 0, None
        for thr_r in _frange(0.30, 0.96, 0.02):
            for thr_z in _frange(0.40, 0.96, 0.02):
                preds = ["stair" if (all_r1[i] > thr_r and all_zef[i] > thr_z) else "ground"
                         for i in range(len(all_labels))]
                res = evaluate(all_labels, preds)
                if res["f1"] > best_and_f1:
                    best_and_f1 = res["f1"]
                    best_and = (thr_r, thr_z, res)

        print(f"\n  [통합 성능] ground={n_gnd} up={n_up}")
        print(f"    dirR1 단독: thr={best_r1_thr:.2f} -> F1={best_r1_res['f1']:.3f} "
              f"P={best_r1_res['prec']:.3f} R={best_r1_res['rec']:.3f}")
        print(f"    zEF 단독:   thr={best_zef_thr:.2f} -> F1={best_zef_res['f1']:.3f} "
              f"P={best_zef_res['prec']:.3f} R={best_zef_res['rec']:.3f}")
        if best_and:
            r, z, res = best_and
            print(f"    R1 AND zEF: R>{r:.2f} & zEF>{z:.2f} -> F1={res['f1']:.3f} "
                  f"P={res['prec']:.3f} R={res['rec']:.3f} (TP={res['tp']} FP={res['fp']} FN={res['fn']})")

        # ── 데이터셋별 성능 ──
        if best_and:
            r_thr, z_thr, _ = best_and
            print(f"\n  [데이터셋별] R>{r_thr:.2f} & zEF>{z_thr:.2f}")
            for name in per_dataset:
                labels, r1s, zefs = per_dataset[name]
                preds = ["stair" if (r1s[i] > r_thr and zefs[i] > z_thr) else "ground"
                         for i in range(len(labels))]
                res = evaluate(labels, preds)
                n_g = sum(1 for l in labels if l == "ground")
                n_u = len(labels) - n_g
                print(f"    {name}: F1={res['f1']:.3f} P={res['prec']:.3f} R={res['rec']:.3f} "
                      f"(TP={res['tp']} FP={res['fp']} FN={res['fn']}) [gnd={n_g} up={n_u}]")

        print()


if __name__ == "__main__":
    main()
