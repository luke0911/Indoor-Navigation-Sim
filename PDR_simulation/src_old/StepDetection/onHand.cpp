#include "StepDetection/onHand.h"
#include "Sensor/CoordinateTransform.h"
#include "Sensor/SensorManager.h"
#include <android/log.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <numeric>
#include <algorithm>
#include "PDRresult.h"

#define LOG_TAG "OnHandStepDetection"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace {
    static inline int64_t now_ms() {
        using namespace std::chrono;
        return duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
} // anonymous

extern SensorManager* getSensorManager();

// =================================================================================
// 생성자: [개선됨] 과검출 방지 튜닝 값 적용 + [복구됨] YawFilter 초기화
// =================================================================================
OnHandStepDetection::OnHandStepDetection()
        : coordinateTransform(new CoordinateTransform()),
          movingAvgAccZ(5),
          movingAvgLinAccZ(5),
          totalStepCount(0),
          lastStepTime(0),
        // [중요] 따닥 방지를 위해 간격 450ms로 설정
          minStepIntervalMs(400),
          stepLength(0.65),
          isUpPeak(false),
          isDownPeak(false),
          isStepFinished(false),
          maxAccZ(0.0),
          minAccZ(0.0),
          upPeakTime(0.0),
          downPeakTime(0.0),
          previousUpPeakTime(0.0),
          currentUpPeakToUpPeakTime(0.0),
          isFirstStep(true),
        // ▼▼▼ 튜닝된 값 (회전 시 오작동 방지) ▼▼▼
          UP_PEAK_THRESHOLD(0.8),
          DOWN_PEAK_THRESHOLD(-0.8),
          MIN_Z_DIFF_THRESHOLD(1.5),   // 1.8 -> 2.2
        // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
          MIN_PEAK2PEAK_MS(200),
          MAX_PEAK2PEAK_MS(2000),
        // [복구] Yaw 필터 초기화
          yawFilter(0.07f, 0.8f, 20.0f),
          filteredYaw(0.0f),
          state(0)
{}

OnHandStepDetection::OnHandStepDetection(float alphaSlow, float alphaFast, float thresholdDeg)
        : coordinateTransform(new CoordinateTransform()),
          movingAvgAccZ(5),
          movingAvgLinAccZ(5),
          totalStepCount(0),
          lastStepTime(0),
          minStepIntervalMs(400), // 여기도 450ms 적용
          stepLength(0.60),
          isUpPeak(false),
          isDownPeak(false),
          isStepFinished(false),
          maxAccZ(0.0),
          minAccZ(0.0),
          upPeakTime(0.0),
          downPeakTime(0.0),
          previousUpPeakTime(0.0),
          currentUpPeakToUpPeakTime(0.0),
          isFirstStep(true),
          UP_PEAK_THRESHOLD(0.8),
          DOWN_PEAK_THRESHOLD(-0.8),
          MIN_Z_DIFF_THRESHOLD(1.5),
          MIN_PEAK2PEAK_MS(200),
          MAX_PEAK2PEAK_MS(2000),
        // [복구] Yaw 필터 초기화
          yawFilter(alphaSlow, alphaFast, thresholdDeg),
          filteredYaw(0.0f),
          state(0)
{}

OnHandStepDetection::~OnHandStepDetection() {
    delete coordinateTransform;
}
bool OnHandStepDetection::isStep(const std::array<float,3>& rotangle,
                                 const std::deque<float>& stepQueue,
                                 int64_t currentTimeMillis,
                                 int statetmp)
{
    if (currentTimeMillis < 0) currentTimeMillis = now_ms();

    filteredYaw = yawFilter.update(rotangle[2]);
    state = statetmp;

    if (!stepQueue.empty()) {
        stepLength = static_cast<double>(stepQueue.front());
    }

    double currentTime = static_cast<double>(currentTimeMillis);

    SensorManager* sm = getSensorManager();
    if (!sm) return false;

    if (sm->isRotationVectorReady()) {
        coordinateTransform->updateFromSensorManager();
    }

    // 1. 센서 데이터 가져오기
    double transformedZ = 0.0;
    if (sm->isLinearAccelerometerReady()) {
        auto linAccData = sm->getLatestLinearAccelerometer();
        TransformedAcceleration globalAcc = coordinateTransform->transformToGlobal(
                linAccData.x, linAccData.y, linAccData.z, currentTime
        );
        lastRawGlobalZ_ = globalAcc.z;
        movingAvgLinAccZ.newData(static_cast<float>(globalAcc.z));
        transformedZ = movingAvgLinAccZ.getAvg();
    } else if (sm->isAccelerometerReady()) {
        auto accData = sm->getLatestAccelerometer();
        TransformedAcceleration globalAcc = coordinateTransform->transformToGlobal(
                accData.x, accData.y, accData.z - 9.8, currentTime
        );
        lastRawGlobalZ_ = globalAcc.z;
        movingAvgAccZ.newData(static_cast<float>(globalAcc.z));
        transformedZ = movingAvgAccZ.getAvg();
    }

    // =========================================================
    // [적응형 스레스홀드] 로직
    // =========================================================

    // 1. 윈도우 업데이트
    ampWindow.push_back(transformedZ);
    if (ampWindow.size() > WINDOW_SIZE) {
        ampWindow.pop_front();
    }

    // 2. 정지 상태 감지 (표준편차/분산 체크) - [핵심 수정 사항]
    // 데이터가 충분히 모였을 때, 변동폭이 너무 적으면 아예 로직을 태우지 않음
    if (ampWindow.size() >= 10) {
        double sum = std::accumulate(ampWindow.begin(), ampWindow.end(), 0.0);
        double mean = sum / ampWindow.size();
        double sq_sum = 0.0;
        for(double v : ampWindow) {
            sq_sum += (v - mean) * (v - mean);
        }
        double std_dev = std::sqrt(sq_sum / ampWindow.size());

        // 표준편차가 0.3 미만이면(거의 정지 상태) 리셋하고 리턴
        // 일반적인 걸음은 표준편차가 최소 1.0 이상 나옴
        if (std_dev < 0.3) {
            if (isUpPeak) { // 진행 중이던 스텝이 있었다면 취소
                reset();
            }
            return false;
        }
    }

    // 3. 동적 Range 계산
    double longRange = 0.0;
    if (ampWindow.size() >= 5) {
        auto result = std::minmax_element(ampWindow.begin(), ampWindow.end());
        longRange = *result.second - *result.first;
    }

    double shortRange = longRange;
    if (ampWindow.size() > 10) {
        auto startIt = ampWindow.end() - 10;
        auto shortResult = std::minmax_element(startIt, ampWindow.end());
        shortRange = *shortResult.second - *shortResult.first;
    }

    double effectiveRange = std::min(longRange, shortRange);

    // 4. 동적 임계값 계산
    // [수정] 최소값을 0.75 -> 1.2로 상향 조정 (작은 노이즈 무시)
    const double ABSOLUTE_MIN_DIFF = 1.2;
    const double RATIO = 0.45;

    double dynamicZDiff = std::max(ABSOLUTE_MIN_DIFF, effectiveRange * RATIO);

    // Up/Down Peak 기준 재설정
    double dynamicUpThresh = dynamicZDiff * 0.4;
    double dynamicDownThresh = -(dynamicZDiff * 0.4);

    // 타임아웃 리셋
    if (isUpPeak && (currentTime - upPeakTime > 2500.0)) {
        reset();
    }

    // =========================================================
    // 스텝 검출 상태 머신
    // =========================================================

    // 1) Up Peak 감지
    if (!isUpPeak && !isDownPeak && !isStepFinished && transformedZ > dynamicUpThresh) {
        if (transformedZ < maxAccZ) {
            isUpPeak = true;
            upPeakTime = currentTime;
        } else {
            maxAccZ = transformedZ;
        }
    }
        // 2) Down Peak 감지
    else if (isUpPeak && !isDownPeak && !isStepFinished) {
        if (transformedZ > maxAccZ) {
            maxAccZ = transformedZ;
            upPeakTime = currentTime;
        } else if (transformedZ < dynamicDownThresh) {
            if (transformedZ > minAccZ) {
                isDownPeak = true;
                downPeakTime = currentTime;
            } else {
                minAccZ = transformedZ;
            }
        }
    }
        // 3) Step 완료 감지
    else if (isUpPeak && isDownPeak && !isStepFinished) {
        if (transformedZ < minAccZ) {
            minAccZ = transformedZ;
            downPeakTime = currentTime;
        }
        else if (transformedZ >= (dynamicUpThresh * 0.5)) {
            isStepFinished = true;
        }
    }

    // 4) Step 판정 및 유효성 검사
    if (isUpPeak && isDownPeak && isStepFinished) {
        bool validStep = false;

        const int64_t timeSinceLastStep = currentTimeMillis - lastStepTime;
        bool isPauseAndGo = (lastStepTime > 0 && timeSinceLastStep > 2000);

        double timePeak2Peak;
        bool isRestartStep = false;

        if (isFirstStep || isPauseAndGo || previousUpPeakTime <= 0.0) {
            timePeak2Peak = downPeakTime - upPeakTime;
            isRestartStep = true;
            if (isFirstStep) isFirstStep = false;
        } else {
            timePeak2Peak = upPeakTime - previousUpPeakTime;
        }

        double zAccDifference = maxAccZ - minAccZ;

        // 유효성 검사 1: Peak-to-Peak 시간
        bool isValidPeakToPeak;
        if (isRestartStep) {
            isValidPeakToPeak = (timePeak2Peak >= 100.0 && timePeak2Peak <= 1500.0);
        } else {
            isValidPeakToPeak = (timePeak2Peak >= MIN_PEAK2PEAK_MS && timePeak2Peak <= MAX_PEAK2PEAK_MS);
        }

        // 유효성 검사 2: 진폭 크기 (동적 임계값 사용)
        bool isValidZDiff = (zAccDifference >= dynamicZDiff);

        // 유효성 검사 3: 스텝 간격
        bool isValidInterval = isRestartStep || (timeSinceLastStep >= minStepIntervalMs);

        // 유효성 검사 4: Peak-Valley 시간 (너무 급격한 노이즈 방지)
        double peakToValleyDuration = downPeakTime - upPeakTime;
        bool isValidDuration = (peakToValleyDuration >= 80.0); // 100 -> 80으로 살짝 완화

        if (isValidPeakToPeak && isValidZDiff && isValidInterval && isValidDuration) {
            validStep = true;
            ++totalStepCount;
            lastStepTime = currentTimeMillis;

            // 유효 스텝의 진폭을 윈도우에 반영하여 적응성 유지
            double detectedAmp = maxAccZ - minAccZ;
            for(int i=0; i<3; i++) { // 5번 -> 3번으로 줄임 (과도한 편향 방지)
                ampWindow.push_back(detectedAmp);
                if (ampWindow.size() > WINDOW_SIZE) {
                    ampWindow.pop_front();
                }
            }

            previousUpPeakTime = upPeakTime;
            lastPeakZ_ = maxAccZ;
            lastValleyZ_ = minAccZ;
        }

        reset();
        return validStep;
    }

    return false;
}


PDR OnHandStepDetection::getStatus() const {
    PDR out{};

    // [복구] 방향 값 설정: CoordinateTransform 우선, 없으면 FilteredYaw 사용
    bool rvReady = (coordinateTransform && coordinateTransform->isReady());

    if (state == 22) { // 특정 상태(예: 주머니) 처리
        double dir = std::fmod(-static_cast<double>(filteredYaw) + 360.0, 360.0);
        out.direction = dir;
    } else if (rvReady) {
        double yawDegrees = coordinateTransform->getYawDegrees();
        while (yawDegrees < 0) yawDegrees += 360.0;
        while (yawDegrees >= 360.0) yawDegrees -= 360.0;
        out.direction = yawDegrees;
    } else {
        // Fallback: 필터링된 Yaw 값 사용
        double dir = std::fmod(-static_cast<double>(filteredYaw) + 360.0, 360.0);
        while (dir < 0) dir += 360.0;
        out.direction = dir;
    }

    out.totalStepCount = totalStepCount;
    out.stepLength = stepLength;
    return out;
}

void OnHandStepDetection::reset() {
    isUpPeak = false;
    isDownPeak = false;
    isStepFinished = false;
    maxAccZ = 0.0;
    minAccZ = 0.0;
}

// DetectionState 구조체가 헤더에 정의되어 있다면 아래 메서드 유지
/*
OnHandStepDetection::DetectionState OnHandStepDetection::getCurrentState() const {
    return { isUpPeak, isDownPeak, isStepFinished, maxAccZ, minAccZ, currentUpPeakToUpPeakTime };
}
*/