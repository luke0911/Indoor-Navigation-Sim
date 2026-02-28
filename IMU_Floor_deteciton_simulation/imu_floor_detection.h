#ifndef IMU_FLOOR_DETECTION_H
#define IMU_FLOOR_DETECTION_H

// =========================================================
// IMU Floor Detection - 층 이동 감지 알고리즘
//
// Android 프로젝트에 그대로 복붙 가능하도록 설계.
// 외부 의존성: <string>, <cstdint>, <cmath>, <vector>, <deque> 만 사용.
// =========================================================

#include <string>
#include <cstdint>
#include <cmath>
#include <vector>
#include <deque>

struct FloorDetectionInput {
    double timeSec;
    double globalX, globalY, globalZ;
    double magX, magY, magZ;
    double gyroAngleDeg;
    double pitch;
    bool   isStep;              // PDR 스텝 디텍션 결과
    int    stepCount;            // 누적 스텝 수
    double stepPeakToPeakMs;     // 스텝 peak-to-peak 시간 (ms), 0이면 없음
};

struct FloorDetectionResult {
    std::string predictedLabel;  // "ground" or "stair"
    double confidence;           // 0.0 ~ 1.0

    // HPF 순수 가속도
    double pureX, pureY, pureZ;

    // Local 보정 가속도 (gyroAngle 회전 변환)
    double localForwardY;   // 내 배꼽 앞방향 전진 가속도
    double localLateralX;   // 내 배꼽 기준 좌우 가속도

    // ── 핵심 피처 (스텝 단위, hold) ──
    double stepRmsZ;       // sqrt(mean(pureZ^2)) per step
    double stepFreqHz;     // onHand peak-to-peak 기반 주파수
    double rmsZdivFreq;    // stepRmsZ / stepFreqHz — 기본 계단 지표
    double stepVarH;       // varX + varY per step — 수평 에너지

    // ── 스텝 시작점 피처 (분석 기반) ──
    double dmag;           // 이전 스텝 시작점과의 거리 (YZ 평면)
    double recentStepVarYZ; // 최근 5스텝 시작점 var(pY)+var(pZ)

    // ── 자기정규화 비율 (Self-normalizing ratios) ──
    //   각 피처를 자기 자신의 장기 baseline 대비 비율로 표현
    //   → 데이터셋/보행자 간 절대값 차이 제거
    double varYZRatio;     // recentStepVarYZ / baseline (< 1 = 집중 = stair)
    double dmagRatio;      // dmag / baseline (< 1 = 일정 = stair)
    double zfRatio;        // rmsZdivFreq / baseline (> 1 = Z에너지 상승 = stair)
    double freqRatio;      // stepFreqHz / baseline (< 1 = 느림 = stair)
    double stairConfidence; // 감지 신뢰도 [0~1]
    double totalStairScore; // 가중치 점수 합계 (0~100)

    // ── 궤적 방향 일관성 ──
    double dirAngle1;      // 스텝 전반부 방향각 (rad)
    double dirR1;          // 최근 5스텝 방향 일관성 [0,1]

    // ── 보조 피처 ──
    double energyFracZ;    // varZ / (varAll + eps)  [0~1]
    double zPeakYRatio;    // Z-peak ±0.1s 구간 Y 에너지 비율

    // Envelope (실시간 참고용)
    double envZ, envH, stairScore;

    // FSM
    std::string fsmLabel;  // "ground" or "stair"
    bool armSpinFlag;

    // legacy
    double rmsX, rmsY, rmsZ;
    double procStepFreq;
};

class IMUFloorDetection {
public:
    IMUFloorDetection();
    void reset();
    FloorDetectionResult update(const FloorDetectionInput& input);

private:
    int sampleCount_;

    // HPF state
    double baseX_, baseY_, baseZ_;
    bool hpfInit_;

    // 스텝 윈도우 버퍼
    struct PureSample { double t; double pX, pY, pZ; };
    std::vector<PureSample> stepBuf_;
    double lastProcStepTime_;
    int    procStepCount_;

    // 스텝 단위 피처 holds
    double lastProcRmsZ_, lastProcStepFreq_;
    double lastRmsZdivFreq_;
    double lastStepVarH_;
    double lastEnergyFracZ_;

    // Arm-spin detection
    double ewmaVarH_;
    bool ewmaInit_;
    bool lastArmSpinFlag_;

    // Envelope LPF state
    double envZ_, envH_;

    // FSM (3-step hysteresis)
    std::string fsmState_, fsmCandState_;
    int fsmStreak_;

    // Z-peak Y analysis buffer
    std::deque<PureSample> zPeakBuf_;
    double lastZPeakYEnergy_, lastZPeakYRatio_;

    // 스텝 구간 RMS 누적 (legacy)
    double sumSqX_, sumSqY_, sumSqZ_;
    int    segCount_;
    double lastRmsX_, lastRmsY_, lastRmsZ_;

    // 스텝 주파수 (peak-to-peak 기반, onHand)
    double lastStepFreqHz_;

    // 스텝 시작점 추적 (dmag, recentStepVarYZ)
    struct StepStartPt { double pY; double pZ; };
    std::deque<StepStartPt> recentStepStarts_;  // 최근 5개
    double lastDmag_;
    double lastRecentStepVarYZ_;

    // Self-normalizing baselines (slow EWMA)
    //   ground 상태에서 빠르게(α=0.10), stair 상태에서 느리게(α=0.02) 업데이트
    //   → baseline이 ground 보행 특성을 주로 반영
    double ewmaBaseVarYZ_;
    double ewmaBaseDmag_;
    double ewmaBaseZdivFreq_;
    double ewmaBaseFreqHz_;
    bool   baselineInit_;
    int    baselineStepCount_;

    // Computed ratios (instantaneous)
    double lastVarYZRatio_;
    double lastDmagRatio_;
    double lastZfRatio_;
    double lastFreqRatio_;

    // Smoothed ratios (fast EWMA α=0.35 for temporal stability)
    double smoothVarYZRatio_;
    double smoothZfRatio_;

    double lastStairConfidence_;
    double lastTotalStairScore_;

    // 궤적 방향 일관성 (Direction Consistency)
    std::deque<double> recentDirAngles_;  // 최근 5개 전반부 방향각
    double lastDirAngle1_;
    double lastDirR1_;
    double smoothDirR1_;
};

#endif // IMU_FLOOR_DETECTION_H
