#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp> // cv::Mat, cv::Point

// 지도 뷰 상태 관리 구조체
struct MapViewState {
    int rotate90 = 0;       // 지도 회전: 0: 0도, 1: 90도, 2: 180도, 3: 270도
    bool flipX = false;     // 지도 좌우 반전
    bool flipY = false;     // 지도 상하 반전
    int moveX = 0;          // 시작점 X 이동량
    int moveY = 0;          // 시작점 Y 이동량
    double pdrRotateDeg = 0.0; // PDR 경로 회전 각도
    bool showNewPath = true;   // New 경로 표시 여부
    bool showOldPath = true;   // Old 경로 표시 여부
};

// Waveform click state (for inspect overlay)
struct WaveformClick {
    bool active = false;
    int sampleIdx = -1;
    double value = 0.0;          // globalZ at clicked sample
    double filteredValue = 0.0;  // filteredZ at clicked sample
    double time_s = 0.0;
};

// Per-step result
struct StepResult {
    int    stepNum;
    double time_s;
    double stepLen;
    double rawStepLen;
    double directionDeg;
    double ampZ;
    double freqHz;
    double peak;
    double valley;
    double x, y;
    int    sampleIdx;
    // Step length pipeline intermediate values
    double fourthRoot = 0.0;     // ampZ^0.25
    double correctedRoot = 0.0;  // threshold correction applied
    double durationMs = 0.0;     // time between steps (ms)
    double kf = 0.0;             // k(f) frequency coefficient
    double kf_walk = 0.0;        // walking sigmoid component
    double kf_run = 0.0;         // running sigmoid component
    double L0 = 0.0;             // kf * correctedRoot * scale
    double Lfinal = 0.0;         // after clamp (before smoothing)
};

// Per-sample raw trace
struct SampleData {
    double time_s;
    double globalZ;
    double filteredZ;
    bool   isStep;
    int    stepIdx;

    bool   isOldStep;
    int    oldStepIdx;

    // Dynamic thresholds from step detection (per-sample)
    double dynamicZDiff      = 0.0;
    double dynamicUpThresh   = 0.0;
    double dynamicDownThresh = 0.0;
};

// 함수 선언
void showAtSample(const std::vector<StepResult>& steps,
                  const std::vector<StepResult>& oldSteps,
                  const std::vector<SampleData>& trace,
                  int sampleIdx,
                  int stride,
                  const std::string& dataFileName,
                  const cv::Mat& mapImage,
                  cv::Point initialStartPos,
                  double mapScalePxPerM,
                  const MapViewState& viewState,
                  const WaveformClick* wfClick = nullptr);

// Controls 요약 창 (한 번만 생성, 매 프레임 갱신)
void showControlsWindow(const MapViewState& viewState, int stride);

// Analysis 창 (설계 의도 검증용 5개 차트)
void showAnalysisWindow(const std::vector<StepResult>& steps,
                        const std::vector<StepResult>& oldSteps,
                        const std::vector<SampleData>& trace,
                        int sampleIdx);

// Step Analysis 창 (보폭 계산 파이프라인 + 걸음별 테이블)
void showStepAnalysisWindow(const std::vector<StepResult>& steps,
                            const std::vector<StepResult>& oldSteps,
                            int sampleIdx,
                            bool isFemale = false);

// World Map 캔버스 오프셋 (마우스 좌표 보정용)
cv::Point getWorldMapCanvasOffset();