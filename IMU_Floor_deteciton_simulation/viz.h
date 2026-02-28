#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>

// Floor label 색상 (BGR)
inline cv::Scalar floorLabelColor(const std::string& label) {
    if (label == "up" || label == "stair") return cv::Scalar(0, 0, 255);     // 빨강
    if (label == "down") return cv::Scalar(255, 100, 0);    // 파랑
    return cv::Scalar(0, 200, 0);                            // 초록 (ground)
}

// Floor label 배경 tint (어두운 배경 위 은은한 색조)
inline cv::Scalar floorLabelTint(const std::string& label) {
    if (label == "up" || label == "stair") return cv::Scalar(30, 25, 55);
    if (label == "down") return cv::Scalar(55, 35, 25);
    return cv::Scalar(25, 48, 25);
}

// label -> 수치: ground=0, up/stair=1, down=-1
inline double floorLabelToValue(const std::string& label) {
    if (label == "up" || label == "stair") return  1.0;
    if (label == "down") return -1.0;
    return 0.0;
}

// 매 샘플 trace 데이터
struct FloorTraceData {
    double time_s;
    double globalX, globalY, globalZ;
    double magX, magY, magZ;
    double gyroAngleDeg;
    double pitch;
    double filteredZ;

    std::string floorLabel;      // ground truth
    std::string predictedLabel;  // 알고리즘 예측
    double confidence;

    // 수치형 floor 값 (그래프용): ground=0, up=1, down=-1
    double floorGT;
    double floorPred;

    // HPF 순수 가속도
    double pureX, pureY, pureZ;

    // Local 보정 가속도 (gyroAngle 회전 변환)
    double localForwardY;   // 내 배꼽 앞방향 전진 가속도
    double localLateralX;   // 내 배꼽 기준 좌우 가속도

    // Envelope 피처 (매 샘플 실시간)
    double envZ;           // LPF(|pureZ|)
    double envH;           // LPF(magH)
    double stairScore;     // envZ - envH

    // 핵심 피처 (스텝 단위, hold)
    double procRmsZ;       // sqrt(mean(pureZ^2)) in step window
    double stepFreqHz;     // onHand peak-to-peak 기반 주파수
    double rmsZdivFreq;    // procRmsZ / stepFreqHz — 기본 계단 지표
    double stepVarH;       // varX + varY per step — 수평 에너지

    // 스텝 시작점 피처
    double dmag;           // 이전 스텝 시작점과의 YZ 거리
    double recentStepVarYZ; // 최근 5스텝 시작점 분산

    // 자기정규화 비율 (Self-normalizing ratios)
    double varYZRatio;     // recentStepVarYZ / baseline (< 1 = stair)
    double dmagRatio;      // dmag / baseline (< 1 = stair)
    double zfRatio;        // rmsZdivFreq / baseline (> 1 = stair)
    double freqRatio;      // stepFreqHz / baseline (< 1 = stair)
    double stairConfidence; // 감지 신뢰도 [0~1]
    double totalStairScore; // 가중치 점수 합계 (0~100)

    // 궤적 방향 일관성
    double dirR1;          // 최근 5스텝 방향 일관성 [0,1]

    // 보조 피처
    double energyFracZ;    // varZ / (varAll + eps)  [0~1]
    double procStepFreq;   // steps per second (legacy)

    // FSM
    std::string fsmLabel;  // "ground" or "stair"
    bool armSpinFlag;

    // Z-peak aligned Y analysis
    double zPeakYRatio;

    // 스텝 구간 가속도 RMS (각 축 개별, legacy)
    double rmsX, rmsY, rmsZ;

    bool isStep;
    int  stepIdx;                // -1 = not a step
    int  totalSteps;
};

// Window 1: Main (파형 + FSM + 스트립 + info)
// gtLabels: 전체 데이터의 GT label (미니맵용, 비어있으면 trace에서 읽음)
void showMainWindow(const std::vector<FloorTraceData>& trace, int sampleIdx,
                    const std::string& dataFileName, bool isLiveMode, bool isRecording,
                    int totalSamples = 0,
                    const std::vector<std::string>& gtLabels = {});

// Window 2: Analysis (2D scatter + Z-peak Y-phase)
void showAnalysisWindow(const std::vector<FloorTraceData>& trace, int sampleIdx);

// Window 3: Step Energy (Z-energy fraction + pureZ level, step-indexed)
void showStepEnergyWindow(const std::vector<FloorTraceData>& trace, int sampleIdx);
