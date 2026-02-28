#ifndef ONHAND_STEP_DETECTION_H
#define ONHAND_STEP_DETECTION_H

#include <array>
#include <deque>
#include <cstdint>

#include "PDRresult.h"
#include "Filter/MovingAverage.h"
#include "Filter/AdaptiveFilter.h"

class CoordinateTransform;

class OnHandStepDetection {
public:
    OnHandStepDetection();
    OnHandStepDetection(float alphaSlow, float alphaFast, float thresholdDeg);

    ~OnHandStepDetection();
    bool isStep(const std::array<float,3>& rotangle,
                const std::deque<float>& stepQueue,
                int64_t currentTimeMillis,
                int statetmp);

    PDR getStatus() const;
    double getLastRawGlobalZ() const { return lastRawGlobalZ_; }
    double getLastPeakZ() const { return lastPeakZ_; }
    double getLastValleyZ() const { return lastValleyZ_; }

    struct DetectionState {
        bool isUpPeak;
        bool isDownPeak;
        bool isStepFinished;
        double maxAccZ;
        double minAccZ;
        double timePeak2Peak;
    };

    DetectionState getCurrentState() const;

private:
    CoordinateTransform* coordinateTransform;

    MovingAverage movingAvgAccZ;
    MovingAverage movingAvgLinAccZ;

    float filteredYaw;
    int state;

    int totalStepCount;
    int64_t lastStepTime;
    int64_t minStepIntervalMs;
    double stepLength;

    bool isUpPeak;
    bool isDownPeak;
    bool isStepFinished;
    double maxAccZ;
    double minAccZ;

    double upPeakTime;
    double downPeakTime;
    double previousUpPeakTime;
    double currentUpPeakToUpPeakTime;
    bool isFirstStep;

    const double UP_PEAK_THRESHOLD;
    const double DOWN_PEAK_THRESHOLD;
    const double MIN_Z_DIFF_THRESHOLD;
    const double MIN_PEAK2PEAK_MS;
    const double MAX_PEAK2PEAK_MS;

    AdaptiveFilter yawFilter;

    double lastRawGlobalZ_ = 0.0;
    double lastPeakZ_ = 0.0;
    double lastValleyZ_ = 0.0;

    std::deque<double> ampWindow;
    const int WINDOW_SIZE = 25;

    void reset();
};

#endif // ONHAND_STEP_DETECTION_H
