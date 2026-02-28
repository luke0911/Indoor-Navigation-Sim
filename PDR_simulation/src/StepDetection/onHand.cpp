#include "StepDetection/onHand.h"
#include "Sensor/CoordinateTransform.h"
#include "Sensor/SensorManager.h"
#include <android/log.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <algorithm> // minmax_element
#include <numeric>   // accumulate (평균 계산용)
#include "PDRresult.h"
#include "SimDataLogger.h"

#define LOG_TAG "OnHandStepDetection"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace {
    inline int64_t now_ms() {
        using namespace std::chrono;
        return duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
}

extern SensorManager* getSensorManager();

#ifdef PDR_SIMULATOR
// --- Cached per-sample dynamic threshold values (for visualization) ---
static double s_lastDynamicZDiff      = 0.0;
static double s_lastDynamicUpThresh   = 0.0;
static double s_lastDynamicDownThresh = 0.0;

double OnHand_getDynamicZDiff()      { return s_lastDynamicZDiff; }
double OnHand_getDynamicUpThresh()   { return s_lastDynamicUpThresh; }
double OnHand_getDynamicDownThresh() { return s_lastDynamicDownThresh; }
#endif

// 생성자 1
OnHandStepDetection::OnHandStepDetection()
        : coordinateTransform(new CoordinateTransform()),
          movingAvgAccZ(5),
          movingAvgLinAccZ(5),
          totalStepCount(0),
          lastStepTime(0),
          minStepIntervalMs(250),
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
          // [수정] 정지 감지를 위해 최소 임계값 상향 조정 (0.75 -> 1.2)
          UP_PEAK_THRESHOLD(0.8),
          DOWN_PEAK_THRESHOLD(-0.8),
          MIN_Z_DIFF_THRESHOLD(1.5),
          MIN_PEAK2PEAK_MS(200),
          MAX_PEAK2PEAK_MS(2000),
          yawFilter(0.07f, 0.8f, 20.0f),
          filteredYaw(0.0f),
          state(0)
{}

// 생성자 2
OnHandStepDetection::OnHandStepDetection(float alphaSlow, float alphaFast, float thresholdDeg)
        : coordinateTransform(new CoordinateTransform()),
          movingAvgAccZ(5),
          movingAvgLinAccZ(5),
          totalStepCount(0),
          lastStepTime(0),
          minStepIntervalMs(250),
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
    if (!sm || !coordinateTransform) return false;

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

    // --- Sim Data 기록 (SimDataLogger 모듈이 자동 처리) ---
    {
        double gyroAngleDeg = 0.0;
        if (coordinateTransform && coordinateTransform->isReady()) {
            gyroAngleDeg = coordinateTransform->getYawDegrees();
            while (gyroAngleDeg < 0) gyroAngleDeg += 360.0;
            while (gyroAngleDeg >= 360.0) gyroAngleDeg -= 360.0;
        }
        SimDataLog_write(currentTimeMillis, lastRawGlobalZ_, gyroAngleDeg);
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
        if (std_dev < 0.25) {
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


    double dynamicZDiff;
    double dynamicUpThresh;
    double dynamicDownThresh;

    // [핵심 수정]
    // 처음 3걸음까지는 생성자에서 설정한 '높은 고정값(1.5 / 0.8)'을 강제 사용하여
    // 주머니에 넣거나 들 때 발생하는 초기 노이즈를 방어합니다.
    if (totalStepCount == 0) {
        // 맨 처음 시작할 때의 손 떨림/노이즈를 방어하기 위해 첫발은 약간의 힘(0.6)이 필요함
        dynamicZDiff = 1.5;
        dynamicUpThresh = 0.6;
        dynamicDownThresh = -0.6;
    }
    else if (totalStepCount < 3) {
        // 첫발을 떼고 나면, 2~3번째 걸음부터는 총총걸음도 잡을 수 있게 확 낮춤
        dynamicZDiff = 1.0;
        dynamicUpThresh = 0.4;
        dynamicDownThresh = -0.4;
    }
    else {
        // 3걸음 이후부터는 '적응형'으로 전환하여 부드러운 걸음도 감지
        // 이때는 최소 바닥값을 1.2로 낮춰주어 끊김을 방지합니다.
        const double ADAPTIVE_MIN_FLOOR = 1.0;
        const double RATIO = 0.45;

        dynamicZDiff = std::max(ADAPTIVE_MIN_FLOOR, effectiveRange * RATIO);

        // 적응형일 때는 비율로 상/하한선 계산
        dynamicUpThresh = dynamicZDiff * 0.4;
        dynamicDownThresh = -(dynamicZDiff * 0.4);
    }

#ifdef PDR_SIMULATOR
    // Cache for external visualization
    s_lastDynamicZDiff      = dynamicZDiff;
    s_lastDynamicUpThresh   = dynamicUpThresh;
    s_lastDynamicDownThresh = dynamicDownThresh;
#endif

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
    // 3) Step 완료 감지
    else if (isUpPeak && isDownPeak && !isStepFinished) {
        if (transformedZ < minAccZ) {
            // 오른쪽의 '진짜 깊은 골짜기'를 파고 내려갈 때마다 최저점과 시간을 계속 갱신합니다.
            minAccZ = transformedZ;
            downPeakTime = currentTime;
        }
        else if (transformedZ >= 0.0) {

            bool isRealEnding = ((currentTime - upPeakTime) > 350.0);

            if (isRealEnding) {
                isStepFinished = true;
            }
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
            lastStepPeakToPeak_ = timePeak2Peak;
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

    if (state == 22) {
        double dir = std::fmod(-static_cast<double>(filteredYaw) + 360.0, 360.0);
        out.direction = dir;
    } else if (coordinateTransform && coordinateTransform->isReady()) {
        double yawDegrees = coordinateTransform->getYawDegrees();
        while (yawDegrees < 0) yawDegrees += 360.0;
        while (yawDegrees >= 360.0) yawDegrees -= 360.0;
        out.direction = yawDegrees;
    } else {
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

#ifdef PDR_SIMULATOR
OnHandStepDetection::DetectionState OnHandStepDetection::getCurrentState() const {
    return { isUpPeak, isDownPeak, isStepFinished, maxAccZ, minAccZ, currentUpPeakToUpPeakTime };
}
#endif