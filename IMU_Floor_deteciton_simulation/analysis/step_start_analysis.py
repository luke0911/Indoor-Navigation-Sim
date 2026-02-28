#!/usr/bin/env python3.9
"""
Step Start Point 심층 분석
- ground vs up vs down 스텝 시작점의 pureY/pureZ 분포 차이
- 추가 파생 피처: 시작점 벡터 크기, 각도, 이전 스텝 대비 변화량
- 통계 요약 + scatter + histogram + boxplot
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUT_DIR), 'data', 'output')

# ── 1. 데이터 로드 ──────────────────────────────────────────
def load_batch(path):
    df = pd.read_csv(path, sep='\t')
    return df

def extract_step_starts(df):
    """isStep==1 인 행만 추출 + 파생 피처 계산"""
    steps = df[df['isStep'] == 1].copy().reset_index(drop=True)
    if steps.empty:
        return steps

    # 기본 피처
    steps['magYZ'] = np.sqrt(steps['pureY']**2 + steps['pureZ']**2)
    steps['angleYZ'] = np.degrees(np.arctan2(steps['pureZ'], steps['pureY']))

    # 이전 스텝 대비 변화량 (delta)
    steps['dpureY'] = steps['pureY'].diff().fillna(0)
    steps['dpureZ'] = steps['pureZ'].diff().fillna(0)
    steps['dmag'] = np.sqrt(steps['dpureY']**2 + steps['dpureZ']**2)

    # 이전 스텝과의 시간 간격
    steps['dt'] = steps['time_s'].diff().fillna(0)

    # 이전 스텝 대비 각도 변화
    steps['dangle'] = steps['angleYZ'].diff().fillna(0)

    # pureX도 포함 (3D 분석)
    steps['magXYZ'] = np.sqrt(steps['pureX']**2 + steps['pureY']**2 + steps['pureZ']**2)

    # 스텝 시작시 globalZ (중력 방향 raw)
    steps['startGlobalZ'] = steps['globalZ'].values

    return steps

# ── 2. 통계 요약 ────────────────────────────────────────────
def print_stats(steps, dataset_name):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Total steps: {len(steps)}")
    print(f"{'='*60}")

    for label in ['ground', 'up', 'down']:
        s = steps[steps['floorGT'] == label]
        if s.empty:
            continue
        print(f"\n  [{label}] n={len(s)}")
        for feat in ['pureY', 'pureZ', 'magYZ', 'angleYZ', 'dmag', 'dangle',
                      'rmsZdivFreq', 'stepVarH', 'stepFreqHz', 'procRmsZ']:
            if feat in s.columns:
                vals = s[feat].values
                print(f"    {feat:15s}: mean={np.mean(vals):7.3f}  std={np.std(vals):7.3f}  "
                      f"med={np.median(vals):7.3f}  [min={np.min(vals):7.3f}, max={np.max(vals):7.3f}]")

# ── 3. 시각화 ───────────────────────────────────────────────
COLORS = {'ground': '#22cc44', 'up': '#dd2222', 'down': '#2266dd'}

def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """2D 데이터에 대한 n_std 신뢰 타원 그리기"""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ell)

def plot_scatter_with_ellipse(steps, title, filename, x_feat='pureY', y_feat='pureZ'):
    """2D scatter + 신뢰 타원"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    for label in ['ground', 'up', 'down']:
        s = steps[steps['floorGT'] == label]
        if s.empty:
            continue
        ax.scatter(s[x_feat], s[y_feat], c=COLORS[label], label=f'{label} (n={len(s)})',
                   alpha=0.7, s=40, edgecolors='white', linewidth=0.5, zorder=3)
        confidence_ellipse(s[x_feat].values, s[y_feat].values, ax, n_std=2.0,
                          facecolor=COLORS[label], alpha=0.15, edgecolor=COLORS[label],
                          linewidth=2, linestyle='--', zorder=2)
        # 평균점 표시
        ax.scatter(s[x_feat].mean(), s[y_feat].mean(), c=COLORS[label],
                   s=200, marker='*', edgecolors='black', linewidth=1, zorder=4)

    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel(x_feat, fontsize=12)
    ax.set_ylabel(y_feat, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_multi_feature_scatter(steps, title_prefix, filename_prefix):
    """여러 피처 조합으로 scatter 그리기"""
    pairs = [
        ('pureY', 'pureZ', 'Step Start: pureY vs pureZ'),
        ('pureY', 'pureX', 'Step Start: pureY vs pureX'),
        ('magYZ', 'angleYZ', 'Step Start: magnitude vs angle (YZ plane)'),
        ('dmag', 'dangle', 'Step-to-step: delta magnitude vs delta angle'),
        ('rmsZdivFreq', 'stepVarH', 'Features: rmsZ/freq vs stepVarH'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i, (xf, yf, subtitle) in enumerate(pairs):
        ax = axes[i]
        for label in ['ground', 'up', 'down']:
            s = steps[steps['floorGT'] == label]
            if s.empty or xf not in s.columns or yf not in s.columns:
                continue
            ax.scatter(s[xf], s[yf], c=COLORS[label], label=f'{label} ({len(s)})',
                       alpha=0.6, s=25, edgecolors='none')
            if len(s) >= 3:
                confidence_ellipse(s[xf].values, s[yf].values, ax, n_std=2.0,
                                  facecolor=COLORS[label], alpha=0.1,
                                  edgecolor=COLORS[label], linewidth=1.5, linestyle='--')
        ax.set_xlabel(xf, fontsize=9)
        ax.set_ylabel(yf, fontsize=9)
        ax.set_title(subtitle, fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

    # 6번째 플롯: pureZ vs stepFreqHz
    ax = axes[5]
    for label in ['ground', 'up', 'down']:
        s = steps[steps['floorGT'] == label]
        if s.empty:
            continue
        ax.scatter(s['stepFreqHz'], s['pureZ'], c=COLORS[label],
                   label=f'{label} ({len(s)})', alpha=0.6, s=25, edgecolors='none')
    ax.set_xlabel('stepFreqHz', fontsize=9)
    ax.set_ylabel('pureZ', fontsize=9)
    ax.set_title('Step freq vs pureZ at start', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f'{title_prefix} — Multi-Feature Step Start Analysis', fontsize=14)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'{filename_prefix}_multi_scatter.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_boxplots(steps, title_prefix, filename_prefix):
    """GT label별 피처 boxplot 비교"""
    features = ['pureY', 'pureZ', 'magYZ', 'angleYZ', 'dmag', 'dangle',
                'rmsZdivFreq', 'stepVarH', 'procRmsZ', 'stepFreqHz']
    features = [f for f in features if f in steps.columns]

    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    labels_present = [l for l in ['ground', 'up', 'down'] if l in steps['floorGT'].values]

    for i, feat in enumerate(features):
        ax = axes[i]
        data_by_label = [steps[steps['floorGT'] == l][feat].dropna().values for l in labels_present]
        bp = ax.boxplot(data_by_label, labels=labels_present, patch_artist=True, notch=True)
        for j, l in enumerate(labels_present):
            bp['boxes'][j].set_facecolor(COLORS[l])
            bp['boxes'][j].set_alpha(0.5)
        ax.set_title(feat, fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')

    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{title_prefix} — Feature Distribution by GT Label', fontsize=14)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'{filename_prefix}_boxplots.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_temporal_scatter(steps, title_prefix, filename_prefix):
    """시간 순서대로 step 시작점 변화 — 연속적인 구간 특성 파악"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # pureY over time
    for label in ['ground', 'up', 'down']:
        s = steps[steps['floorGT'] == label]
        if s.empty:
            continue
        axes[0].scatter(s['time_s'], s['pureY'], c=COLORS[label], s=20, alpha=0.7, label=label)
    axes[0].set_ylabel('pureY at step start')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_title(f'{title_prefix} — Step Start pureY over time')

    # pureZ over time
    for label in ['ground', 'up', 'down']:
        s = steps[steps['floorGT'] == label]
        if s.empty:
            continue
        axes[1].scatter(s['time_s'], s['pureZ'], c=COLORS[label], s=20, alpha=0.7, label=label)
    axes[1].set_ylabel('pureZ at step start')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    # magYZ over time
    for label in ['ground', 'up', 'down']:
        s = steps[steps['floorGT'] == label]
        if s.empty:
            continue
        axes[2].scatter(s['time_s'], s['magYZ'], c=COLORS[label], s=20, alpha=0.7, label=label)
    axes[2].set_ylabel('magYZ at step start')
    axes[2].set_xlabel('time (s)')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'{filename_prefix}_temporal.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_consecutive_step_pattern(steps, title_prefix, filename_prefix):
    """연속 스텝간 시작점 이동 패턴: 화살표로 표시"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 전체 연속 화살표 (흐리게)
    for i in range(1, len(steps)):
        y0, z0 = steps.iloc[i-1]['pureY'], steps.iloc[i-1]['pureZ']
        y1, z1 = steps.iloc[i]['pureY'], steps.iloc[i]['pureZ']
        label = steps.iloc[i]['floorGT']
        col = COLORS.get(label, 'gray')
        ax.annotate('', xy=(y1, z1), xytext=(y0, z0),
                    arrowprops=dict(arrowstyle='->', color=col, alpha=0.3, lw=0.8))

    # 점 표시 (GT 색상)
    for label in ['ground', 'up', 'down']:
        s = steps[steps['floorGT'] == label]
        if s.empty:
            continue
        ax.scatter(s['pureY'], s['pureZ'], c=COLORS[label], label=f'{label} ({len(s)})',
                   s=40, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=5)

    ax.set_xlabel('pureY', fontsize=12)
    ax.set_ylabel('pureZ', fontsize=12)
    ax.set_title(f'{title_prefix} — Step-to-Step Transition Arrows', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'{filename_prefix}_transitions.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_variance_analysis(steps, title_prefix, filename_prefix):
    """GT label별 스텝 시작점 분산 비교 — 핵심 인사이트"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. pureY/pureZ 분산 비교
    ax = axes[0]
    labels_present = [l for l in ['ground', 'up', 'down'] if l in steps['floorGT'].values]
    x_pos = range(len(labels_present))
    varY_vals = [steps[steps['floorGT']==l]['pureY'].var() for l in labels_present]
    varZ_vals = [steps[steps['floorGT']==l]['pureZ'].var() for l in labels_present]
    w = 0.35
    ax.bar([x - w/2 for x in x_pos], varY_vals, w, label='var(pureY)', color='#ff9944')
    ax.bar([x + w/2 for x in x_pos], varZ_vals, w, label='var(pureZ)', color='#4488ff')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels_present)
    ax.set_ylabel('Variance')
    ax.set_title('Start point variance by label')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # 2. 연속 스텝 간 이동 거리 (dmag) 분포
    ax = axes[1]
    for label in labels_present:
        s = steps[steps['floorGT'] == label]
        if s.empty or 'dmag' not in s.columns:
            continue
        vals = s['dmag'].dropna().values
        if len(vals) > 0:
            ax.hist(vals, bins=20, alpha=0.5, color=COLORS[label], label=f'{label} (med={np.median(vals):.2f})',
                    density=True, edgecolor='white')
    ax.set_xlabel('dmag (step-to-step distance)')
    ax.set_ylabel('Density')
    ax.set_title('Step-to-step movement distance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # 3. 각도 분산 비교
    ax = axes[2]
    for label in labels_present:
        s = steps[steps['floorGT'] == label]
        if s.empty:
            continue
        vals = s['angleYZ'].dropna().values
        if len(vals) > 0:
            ax.hist(vals, bins=20, alpha=0.5, color=COLORS[label],
                    label=f'{label} (std={np.std(vals):.1f}°)', density=True, edgecolor='white')
    ax.set_xlabel('angleYZ (degrees)')
    ax.set_ylabel('Density')
    ax.set_title('Start point angle distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f'{title_prefix} — Variance & Distribution Analysis', fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'{filename_prefix}_variance.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_separability_summary(steps, title_prefix, filename_prefix):
    """Ground vs Up 분리 가능성 정량 평가"""
    gnd = steps[steps['floorGT'] == 'ground']
    up = steps[steps['floorGT'] == 'up']
    if gnd.empty or up.empty:
        print("  (skip separability — need both ground and up)")
        return

    features = ['pureY', 'pureZ', 'magYZ', 'angleYZ', 'dmag',
                'rmsZdivFreq', 'stepVarH', 'procRmsZ', 'stepFreqHz']
    features = [f for f in features if f in steps.columns]

    # Cohen's d for each feature
    d_values = {}
    for feat in features:
        g_vals = gnd[feat].dropna().values
        u_vals = up[feat].dropna().values
        if len(g_vals) < 2 or len(u_vals) < 2:
            continue
        pooled_std = np.sqrt((np.var(g_vals) + np.var(u_vals)) / 2)
        if pooled_std < 1e-9:
            continue
        d = abs(np.mean(g_vals) - np.mean(u_vals)) / pooled_std
        d_values[feat] = d

    # KS test p-value
    ks_values = {}
    for feat in features:
        g_vals = gnd[feat].dropna().values
        u_vals = up[feat].dropna().values
        if len(g_vals) < 2 or len(u_vals) < 2:
            continue
        stat, pval = stats.ks_2samp(g_vals, u_vals)
        ks_values[feat] = (stat, pval)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Cohen's d bar chart
    ax = axes[0]
    sorted_feats = sorted(d_values.keys(), key=lambda x: d_values[x], reverse=True)
    vals = [d_values[f] for f in sorted_feats]
    bars = ax.barh(sorted_feats, vals, color=['#ff6644' if v > 0.8 else '#ffaa44' if v > 0.5 else '#aaaaaa' for v in vals])
    ax.axvline(0.8, color='red', ls='--', lw=1, label='Large effect (0.8)')
    ax.axvline(0.5, color='orange', ls='--', lw=1, label='Medium effect (0.5)')
    ax.set_xlabel("Cohen's d (ground vs up)")
    ax.set_title("Feature Separability: Cohen's d")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='x')

    # KS statistic
    ax = axes[1]
    ks_feats = sorted(ks_values.keys(), key=lambda x: ks_values[x][0], reverse=True)
    ks_stats = [ks_values[f][0] for f in ks_feats]
    ks_pvals = [ks_values[f][1] for f in ks_feats]
    colors_ks = ['#ff6644' if p < 0.01 else '#ffaa44' if p < 0.05 else '#aaaaaa' for p in ks_pvals]
    ax.barh(ks_feats, ks_stats, color=colors_ks)
    ax.set_xlabel('KS Statistic (ground vs up)')
    ax.set_title('KS Test: Distribution Difference')
    # 유의성 표시
    for i, (f, p) in enumerate(zip(ks_feats, ks_pvals)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(ks_stats[i] + 0.01, i, f'p={p:.4f} {sig}', va='center', fontsize=8)
    ax.grid(True, alpha=0.2, axis='x')

    fig.suptitle(f'{title_prefix} — Ground vs Up Separability', fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'{filename_prefix}_separability.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # Print summary
    print(f"\n  {'Feature':20s} {'Cohen_d':>8s}  {'KS_stat':>8s}  {'KS_p':>10s}")
    print(f"  {'-'*50}")
    for f in sorted_feats:
        d = d_values.get(f, 0)
        ks = ks_values.get(f, (0, 1))
        sig = '***' if ks[1] < 0.001 else '**' if ks[1] < 0.01 else '*' if ks[1] < 0.05 else ''
        print(f"  {f:20s} {d:8.3f}  {ks[0]:8.3f}  {ks[1]:10.6f} {sig}")

# ── Main ─────────────────────────────────────────────────────
def analyze_file(tsv_path, name):
    print(f"\n{'#'*60}")
    print(f"  Analyzing: {name}")
    print(f"  File: {tsv_path}")
    print(f"{'#'*60}")

    df = load_batch(tsv_path)
    steps = extract_step_starts(df)
    print(f"  Total rows: {len(df)}, Steps: {len(steps)}")
    label_counts = steps['floorGT'].value_counts().to_dict()
    print(f"  Step counts by GT: {label_counts}")

    if len(steps) < 5:
        print("  (too few steps, skipping)")
        return steps

    prefix = name.replace('.tsv', '')
    print_stats(steps, name)
    plot_scatter_with_ellipse(steps, f'{name}: Step Start (pureY vs pureZ)', f'{prefix}_scatter_yz.png')
    plot_multi_feature_scatter(steps, name, prefix)
    plot_boxplots(steps, name, prefix)
    plot_temporal_scatter(steps, name, prefix)
    plot_consecutive_step_pattern(steps, name, prefix)
    plot_variance_analysis(steps, name, prefix)
    plot_separability_summary(steps, name, prefix)
    return steps

def main():
    tsv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.tsv')])
    if not tsv_files:
        print(f"No TSV files found in {DATA_DIR}")
        return

    print(f"Found {len(tsv_files)} TSV files in {DATA_DIR}")

    all_steps = []
    for f in tsv_files:
        path = os.path.join(DATA_DIR, f)
        steps = analyze_file(path, f)
        if steps is not None and len(steps) > 0:
            steps['dataset'] = f
            all_steps.append(steps)

    # 전체 통합 분석 (모든 데이터셋 합산)
    if len(all_steps) > 1:
        combined = pd.concat(all_steps, ignore_index=True)
        print(f"\n{'#'*60}")
        print(f"  COMBINED ANALYSIS ({len(combined)} steps from {len(all_steps)} datasets)")
        print(f"{'#'*60}")
        label_counts = combined['floorGT'].value_counts().to_dict()
        print(f"  Combined step counts: {label_counts}")
        print_stats(combined, 'COMBINED')
        plot_scatter_with_ellipse(combined, 'COMBINED: Step Start (pureY vs pureZ)',
                                  'COMBINED_scatter_yz.png')
        plot_multi_feature_scatter(combined, 'COMBINED', 'COMBINED')
        plot_boxplots(combined, 'COMBINED', 'COMBINED')
        plot_variance_analysis(combined, 'COMBINED', 'COMBINED')
        plot_separability_summary(combined, 'COMBINED', 'COMBINED')

    print(f"\n{'='*60}")
    print(f"  Analysis complete. Images saved to: {OUT_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
