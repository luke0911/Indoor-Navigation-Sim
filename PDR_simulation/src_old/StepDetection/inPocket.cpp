#include <chrono>
#include <cmath>
#include <cstdint>
#include "StepDetection/inPocket.h"
#include "PDRresult.h"

namespace {
static inline int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // anonymous


    InPocket::InPocket()
    : rotMovingAveragePitch(10),
      adaptiveFilterYaw(0.07f, 0.8f, 20.0f),
      filteredYaw(0.0f), devicePosture(0), userAttitude(0),
      totalStepCount(0), lastStepTime(0), minStepIntervalMs(300),
      movementMode(0), stepLength(0.0),
      previusrotX(0.0), lastrisepeek(0.0), lastfallpeek(0.0),
      isRising(0), lastrising(0),
      staircount(0), laststaircount(0), suspendStepcount(0) {}

bool InPocket::isStep(const std::array<float,3>& rotangle,
                          const std::deque<float>& stepQueue,
                          int64_t currentTimeMillis,
                          int statetmp)
{
    if (currentTimeMillis < 0) currentTimeMillis = now_ms();

    rotMovingAveragePitch.newData(rotangle[0]);
    filteredYaw = adaptiveFilterYaw.update(rotangle[2]);

    // Kotlin: stepLength = stepQueue.peek().toDouble() * 1.02
    stepLength = stepQueue.empty() ? 0.0 : static_cast<double>(stepQueue.front()) * 1.02;

    const float currentrotx = rotMovingAveragePitch.getAvg();
    bool stepDetected = false;
    const int64_t timeSinceLastStep = currentTimeMillis - lastStepTime;

    // 상승/하강 판정 with deadband 0.1
    const int wasRising = isRising;
    if (currentrotx > previusrotX + 0.1f) {
        isRising = 1;  // 걷는중
    } else if (currentrotx < previusrotX - 0.1f) {
        isRising = -1; // 걷는중
    }

    // isRising 값이 변하면 (피크/바닥) 스텝 검출
    if (wasRising != isRising) {
        if (timeSinceLastStep > minStepIntervalMs) {
            switch (isRising) {
                case 1: {
                    stepDetected = true;
                    ++totalStepCount;
                    lastStepTime = currentTimeMillis;
                    movementMode = 0;
                    lastfallpeek = currentrotx;
                    break;
                }
                case -1: {
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

    previusrotX = currentrotx;
    return stepDetected;
}

PDR InPocket::getStatus() const {
    PDR out{};
    // direction = (-filteredYaw + 360) % 360
    double dir = std::fmod(-static_cast<double>(filteredYaw) + 360.0, 360.0);
    if (dir < 0) dir += 360.0;
    out.direction = dir;
    out.totalStepCount = totalStepCount;
    return out;
}

