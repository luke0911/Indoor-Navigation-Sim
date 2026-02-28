#ifndef STEPDETECTION_INPOCKET_H
#define STEPDETECTION_INPOCKET_H

#include <array>
#include <deque>
#include <cstdint>
#include <functional>

#include "Filter/MovingAverage.h"
#include "Filter/AdaptiveFilter.h"
#include "PDRresult.h"


class InPocket {
public:
    InPocket();

    // rotangle: {pitch, roll, yaw}
    // stepQueue: front() is the latest step length estimate
    bool isStep(const std::array<float,3>& rotangle,
                const std::deque<float>& stepQueue,
                int64_t currentTimeMillis,
                int statetmp);

    PDR getStatus() const;

private:
    // --- 필요한 매니저들 ---
    MovingAverage rotMovingAveragePitch;   // window = 10
    AdaptiveFilter adaptiveFilterYaw;      // alphaSlow=0.07, alphaFast=0.8, threshold=20
    float filteredYaw;
    int devicePosture;
    int userAttitude;

    // --- 스텝 검출 파라미터 ---
    int totalStepCount;
    int64_t lastStepTime;                  // ms
    const int64_t minStepIntervalMs;       // 300 ms

    // --- PDR 관련 ---
    int movementMode;
    double stepLength;

    // --- 피크 검출 보조 ---
    double previusrotX;
    double lastrisepeek;
    double lastfallpeek;
    int isRising;
    int lastrising;                        // 20250307 추가

    // --- 계단 카운팅 ---
    int staircount;
    int laststaircount;
    int suspendStepcount;                  // 20250307 추가

    // External analysis hook
    std::function<void()> analyzeTrendCallback;
};


#endif // PDR_EXT_STEPDETECTION_INPOCKET_H