#include <chrono>
#include <cmath>
#include "StepDetection/handSwing.h"

// 현재 시간(ms) 유틸리티
static inline int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

HandSwing::HandSwing()
    : rotMovingAveragePitch(10),
      adaptiveFilterYaw(0.07f, 0.8f, 20.0f),
      filteredYaw(0.0f), devicePosture(0), userAttitude(0),
      totalStepCount(0), lastStepTime(0), minStepIntervalMs(300),
      movementMode(0), stepLength(0.0),
      previusrotX(0.0), lastrisepeek(0.0), lastfallpeek(0.0),
      isRising(0), lastrising(0),
      staircount(0), laststaircount(0), suspendStepcount(0) {}

bool HandSwing::isStep(const std::array<float,3>& rotangle,
                       const std::deque<float>& stepQueue,
                       int64_t currentTimeMillis,
                       int statetmp)
{
    if (currentTimeMillis < 0) currentTimeMillis = now_ms();

    // 1) 입력 필터링: pitch(X=0) 평균 + yaw 적응필터
    rotMovingAveragePitch.newData(rotangle[1]);
    filteredYaw = adaptiveFilterYaw.update(rotangle[2]);

    // 최신 스텝 길이 (비어있으면 0)
    stepLength = stepQueue.empty() ? 0.0 : static_cast<double>(stepQueue.front());

    const float currentrotx = rotMovingAveragePitch.getAvg();
    bool stepDetected = false;
    const int64_t timeSinceLastStep = currentTimeMillis - lastStepTime;

    // 2) 상승/하강 판정 (데드밴드 0.1)
    const int wasRising = isRising;
    if (currentrotx > previusrotX + 0.1f) {
        isRising = 1;    // 상승
    } else if (currentrotx < previusrotX - 0.1f) {
        isRising = -1;   // 하강
    }

    // 3) 상태 전이 시 스텝 검출
    if (wasRising != isRising) {
        if (timeSinceLastStep > minStepIntervalMs) {
            switch (isRising) {
                case 1: { // 상승으로 전이: 스텝 카운트
                    stepDetected = true;
                    ++totalStepCount;
                    lastStepTime = currentTimeMillis;
                    movementMode = 0;
                    lastfallpeek = currentrotx; // 이름 유지(호환)
                    break;
                }
                case -1: { // 하강으로 전이: 스텝 카운트
                    stepDetected = true;
                    ++totalStepCount;
                    lastStepTime = currentTimeMillis;
                    movementMode = 0;
                    lastrisepeek = currentrotx;
                    break;
                }
                default:
                    break;
            }
        }
    }

    // 4) 상태 업데이트
    laststaircount = staircount;
    previusrotX = currentrotx;
    return stepDetected;
}

PDR HandSwing::getStatus() const {
    PDR out{};
    // 방향 계산: (-yaw + 360) % 360
    double dir = std::fmod(-static_cast<double>(filteredYaw) + 360.0, 360.0);
    if (dir < 0) dir += 360.0;
    out.direction = dir;
    out.totalStepCount = totalStepCount;
    return out;
}

