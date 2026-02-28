import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import filedialog
import os

# ==========================================
# 1. 중력 제거 함수 (High-Pass Filter)
# ==========================================
def apply_hpf(df, alpha=0.995):
    """
    전체 가속도 데이터에서 중력을 분리하여 순수 동적 가속도(pureX, Y, Z)를 추출합니다.
    """
    print("-> Applying High-Pass Filter...")
    pureX = np.zeros(len(df))
    pureY = np.zeros(len(df))
    pureZ = np.zeros(len(df))
    
    bX, bY, bZ = df['globalX'].iloc[0], df['globalY'].iloc[0], df['globalZ'].iloc[0]

    for i in range(1, len(df)):
        bX = alpha * bX + (1 - alpha) * df['globalX'].iloc[i]
        bY = alpha * bY + (1 - alpha) * df['globalY'].iloc[i]
        bZ = alpha * bZ + (1 - alpha) * df['globalZ'].iloc[i]
        
        pureX[i] = df['globalX'].iloc[i] - bX
        pureY[i] = df['globalY'].iloc[i] - bY
        pureZ[i] = df['globalZ'].iloc[i] - bZ

    df['pureX'] = pureX
    df['pureY'] = pureY
    df['pureZ'] = pureZ
    return df

# ==========================================
# 2. 스텝 단위 피처 추출 및 FSM 판별 함수 (3스텝 지연 & 주파수 적용)
# ==========================================
def process_step_by_step(df, z_threshold=1.0, step_window=2):
    print(f"-> Processing Step-by-Step Features & FSM (Window: {step_window} steps)...")
    dt = df['time(s)'].diff().median()
    if pd.isna(dt) or dt == 0: dt = 0.02
    
    step_indices = df[df['stepCount'] > 0].index.tolist()
    
    df['step_rmsZ'] = 0.0
    df['step_energyRatio'] = 0.0
    df['step_freq'] = 0.0
    df['pred_label'] = 'ground'  # 초기 상태

    df['arm_spin'] = 0  # 팔을 과도하게 돌리는 구간(오염) 감지 플래그

    # 팔 회전/흔들림으로 XY 에너지가 폭증할 때 energyRatio가 붕괴하는 문제를 막기 위한 적응형 기준
    ewma_varH = None
    VARH_MULT = 6.0          # ewma 대비 몇 배면 '팔 과도 회전'으로 볼지
    MIN_VARH_CAP = 0.25      # 너무 작은 값으로 캡이 잡히는 것 방지
    
    current_state = 'ground'
    cand_state = 'ground'
    streak = 0

    for i in range(1, len(step_indices)):
        current_start = step_indices[i-1]
        current_end = step_indices[i]
        
        if current_end - current_start < 5: 
            df.loc[current_start:current_end, 'pred_label'] = current_state
            continue
        
        lookback_i = max(0, i - step_window)
        window_start_idx = step_indices[lookback_i]
        
        window = df.iloc[window_start_idx:current_end]
        actual_steps = i - lookback_i 
        
        wZ = window['pureZ'].values
        wY = window['pureY'].values
        wX = window['pureX'].values
        
        # --------------------------------------------------
        # [피처 계산]
        # --------------------------------------------------
        rmsZ = np.sqrt(np.mean(wZ**2))

        # XY(수평) 에너지와 Z(수직) 에너지 분리
        varX = np.var(wX)
        varY = np.var(wY)
        varZ = np.var(wZ)
        varH = varX + varY + 1e-6
        varAll = varH + varZ

        # 기존 energyRatio(varZ/varH)는 팔 흔들림 시 분모 폭증으로 붕괴 → Z 비중(0~1)으로 변경
        energyFracZ = varZ / (varAll + 1e-6)

        # ★ Step Frequency는 '샘플수*dt'가 아니라 실제 step 이벤트 시간차로 계산(실시간에서도 안정)
        t_start = df['time(s)'].iloc[window_start_idx]
        t_end = df['time(s)'].iloc[current_end]
        time_taken = float(t_end - t_start)
        step_freq = actual_steps / time_taken if time_taken > 1e-6 else 0.0

        # 팔 과도 회전(오염) 감지: varH가 평상시(ewma) 대비 과도하게 크면 판정 업데이트를 동결
        if ewma_varH is None:
            ewma_varH = varH
        varH_cap = max(ewma_varH * VARH_MULT, MIN_VARH_CAP)
        is_arm_spin = varH > varH_cap
        
        df.loc[current_start:current_end, 'step_rmsZ'] = rmsZ
        df.loc[current_start:current_end, 'step_energyRatio'] = energyFracZ  # 컬럼명 유지(호환), 값은 Z 비중
        df.loc[current_start:current_end, 'step_freq'] = step_freq
        df.loc[current_start:current_end, 'arm_spin'] = 1 if is_arm_spin else 0
        
        # --------------------------------------------------
        # [FSM 상태 판별 로직 (단순화: ground vs stair)]
        # --------------------------------------------------
        raw_state = 'ground'
        step_count_val = df['stepCount'].iloc[current_end]
        
        if step_count_val <= 5:
            raw_state = 'ground'  # 센서 안정화

        # 팔을 과도하게 돌리는 구간은 계단/평지 판정을 오염시킴 → 상태머신 업데이트를 '동결'
        elif is_arm_spin:
            # pred_label은 아래에서 current_state로 유지하고, streak/candidate도 건드리지 않음
            df.loc[current_start:current_end, 'pred_label'] = current_state
            # ground 상태에서 정상 구간일 때만 ewma를 업데이트 (오염 구간은 학습 금지)
            continue

        elif step_freq > 3.2:
            raw_state = 'ground'  # 사람이 계단을 오르기엔 매우 어려운 주파수(제자리 흔들기 등) 컷을 조금 여유있게

        else:
            # 계단(stair) 판별 조건:
            # - Z 비중이 충분히 크거나(팔 흔들림/수평 노이즈에 덜 민감)
            # - 수직 충격(rmsZ)이 충분히 크면 계단으로 간주
            if energyFracZ > 0.55 or rmsZ > 3.8:
                raw_state = 'stair'
            else:
                raw_state = 'ground'

        # ewma_varH는 '오염이 아니고, 현재 상태가 ground'일 때만 천천히 업데이트
        if current_state == 'ground':
            ewma_varH = 0.95 * ewma_varH + 0.05 * varH
                        
        # --------------------------------------------------
        # [연속성 검증 (3스텝 이상 유지 시 전환)]
        # --------------------------------------------------
        if raw_state == cand_state:
            streak += 1
        else:
            cand_state = raw_state
            streak = 1
            
        # ★ 핵심: 똑같은 상태가 3스텝 연속 감지될 때만 진짜 상태를 변경함!
        if streak >= 3:
            current_state = cand_state
            
        df.loc[current_start:current_end, 'pred_label'] = current_state

    if len(step_indices) > 0:
        df.loc[step_indices[-1]:, 'pred_label'] = current_state
        
    return df, step_indices

# ==========================================
# 3. 시각화 함수
# ==========================================
def plot_analysis(df, step_indices, file_name):
    print("-> Plotting results...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # 색상 딕셔너리 업데이트 (평지: 초록, 계단: 빨강)
    color_dict = {'ground': 'limegreen', 'stair': 'red'}

    def draw_bg(ax, col_name, alpha_val):
        if col_name not in df.columns: return
        start_t = df['time(s)'].iloc[0]
        curr_l = df[col_name].iloc[0]
        
        # 정답 라벨에 up/down이 섞여 있을 수 있으므로 시각화용 통합
        if curr_l in ['up', 'down']: curr_l = 'stair'
            
        for i in range(1, len(df)):
            l = df[col_name].iloc[i]
            if l in ['up', 'down']: l = 'stair' # 정답 라벨 통합 처리
            
            if l != curr_l or i == len(df)-1:
                end_t = df['time(s)'].iloc[i]
                if curr_l in color_dict:
                    ax.axvspan(start_t, end_t, color=color_dict[curr_l], alpha=alpha_val, lw=0)
                start_t = end_t
                curr_l = l

    # [창 1] 실제 정답 & 원시 데이터 파형
    draw_bg(axes[0], 'label', 0.2)
    axes[0].plot(df['time(s)'], df['pureZ'], color='black', alpha=0.8, label='pureZ (Vertical)')
    axes[0].plot(df['time(s)'], -df['pureY'], color='magenta', alpha=0.5, label='-pureY (Forward)')
    axes[0].set_title(f'[{file_name}] 1. Ground Truth & Raw Accel', fontsize=13)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # [창 2] 스텝 단위 피처 값 (주파수 포함)
    axes[1].plot(df['time(s)'], np.clip(df['step_energyRatio'], 0, 1.0), color='orange', linewidth=2, label='Energy Fraction (Z / (X+Y+Z))')
    axes[1].plot(df['time(s)'], df['step_rmsZ'], color='blue', linewidth=2, label='RMS Z (Impact)')
    axes[1].plot(df['time(s)'], df['step_freq'] * 2, color='purple', linewidth=2, linestyle='--', label='Step Freq Hz (x2 for scale)')
    if 'arm_spin' in df.columns:
        axes[1].plot(df['time(s)'], df['arm_spin'] * 5, color='black', linewidth=1.5, linestyle=':', label='Arm Spin Flag (x5)')
    
    axes[1].axhline(0.55, color='orange', linestyle='--', alpha=0.5, label='Stair Condition (Z-Frac > 0.55)')
    axes[1].set_title('2. Step-by-Step Features (Freq added)', fontsize=13)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # [창 3] FSM 예측 결과 & 스텝 경계선 (3스텝 Hysteresis)
    draw_bg(axes[2], 'pred_label', 0.4)
    axes[2].plot(df['time(s)'], df['pureZ'], color='black', alpha=0.3)
    
    step_times = df['time(s)'].iloc[step_indices].values
    for st in step_times:
        axes[2].axvline(st, color='black', linestyle=':', alpha=0.4)

    axes[2].set_title('3. FSM Prediction Result (Requires 3 Consecutive Steps to Change State)', fontsize=13)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    custom_lines = [Line2D([0], [0], color=color_dict['ground'], lw=6),
                    Line2D([0], [0], color=color_dict['stair'], lw=6)]
    axes[2].legend(custom_lines, ['Pred: GROUND', 'Pred: STAIR'], loc='upper right')

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. 파일 실행 및 메인 블록
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="분석할 센서 데이터 파일을 선택하세요", 
        filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if file_path:
        file_name = os.path.basename(file_path)
        print(f"Loading data: {file_name}")
        
        df = pd.read_csv(file_path, sep='\t')
        df = apply_hpf(df)
        
        # 3스텝 지연 로직이 포함된 함수 실행
        df, step_indices = process_step_by_step(df, z_threshold=1.0, step_window=2)
        
        plot_analysis(df, step_indices, file_name)
    else:
        print("파일 선택이 취소되었습니다.")