#pragma once
#include <cstdint>

class SLRequire {
public:
    SLRequire(int maWin = 5,
              float upThresh = 0.12f,
              float downThresh = -0.12f,
              int minPeak2PeakMs = 160,
              float minZDiff = 0.25f);

    void resetAll();
    void resetSegment();

    void setThresholds(float upThresh, float downThresh);
    void setMinPeak2PeakMs(int ms);
    void setMinZDiff(float dz);
    void setMaWindow(int win);

    void setPeakValley(float peak, float valley);
    void feed(float zGlobal, int64_t tMs);
    bool isValidStepCandidate(int64_t nowMs) const;

    bool isUpOn() const { return isUpPeak; }
    bool isDownOn() const { return isDownPeak; }

    float amplitude() const { return maxAccZ - minAccZ; }
    float peak() const { return maxAccZ; }
    float valley() const { return minAccZ; }

    int64_t lastUpTimeMs() const { return lastUpTime; }
    int64_t lastDownTimeMs() const { return lastDownTime; }

    float lastFilteredZ() const { return lastZ; }

private:
    float applyMA(float x);

    int   maWindow;
    float thUp;
    float thDown;
    int   minP2Pms;
    float minZDiff;

    float maSum;
    int   maCount;

    bool  isUpPeak;
    bool  isDownPeak;
    float maxAccZ;
    float minAccZ;
    int64_t lastUpTime;
    int64_t lastDownTime;
    int64_t segStartTime;

    float lastZ;
};

SLRequire& SLRequire_Instance();
