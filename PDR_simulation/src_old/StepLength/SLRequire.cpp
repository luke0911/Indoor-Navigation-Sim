#include "StepLength/SLRequire.h"
#include <algorithm>
#include <cmath>

SLRequire::SLRequire(int maWin, float upThresh, float downThresh,
                     int minPeak2PeakMs, float minZDiff_)
: maWindow(maWin), thUp(upThresh), thDown(downThresh),
  minP2Pms(minPeak2PeakMs), minZDiff(minZDiff_),
  maSum(0.0f), maCount(0),
  isUpPeak(false), isDownPeak(false),
  maxAccZ(-1e9f), minAccZ(1e9f),
  lastUpTime(-1), lastDownTime(-1), segStartTime(-1),
  lastZ(0.0f) {}

void SLRequire::resetAll() {
    maSum = 0.0f; maCount = 0;
    isUpPeak = false; isDownPeak = false;
    maxAccZ = -1e9f; minAccZ = 1e9f;
    lastUpTime = -1; lastDownTime = -1; segStartTime = -1;
    lastZ = 0.0f;
}

void SLRequire::resetSegment() {
    isUpPeak = false; isDownPeak = false;
    maxAccZ = -1e9f; minAccZ = 1e9f;
    lastUpTime = -1; lastDownTime = -1;
    segStartTime = -1;
}

void SLRequire::setThresholds(float upThresh, float downThresh) {
    thUp = upThresh; thDown = downThresh;
}

void SLRequire::setMinPeak2PeakMs(int ms) { minP2Pms = ms; }
void SLRequire::setMinZDiff(float dz) { minZDiff = dz; }
void SLRequire::setMaWindow(int win) { maWindow = std::max(1, win); }

float SLRequire::applyMA(float x) {
    // Sliding/leaky moving average; mirrors onHand MA(5) style smoothing
    if (maCount < maWindow) { maSum += x; ++maCount; }
    else { maSum += x - (maSum / maWindow); }
    int denom = std::max(1, std::min(maCount, maWindow));
    return (maSum / denom);
}

void SLRequire::setPeakValley(float peak, float valley) {
    maxAccZ = peak;
    minAccZ = valley;
}

void SLRequire::feed(float zGlobal, int64_t tMs) {
    // Smooth Z to reduce jitter; keep same spirit as onHand
    float z = applyMA(zGlobal);
    lastZ = z;

    // Up crossing (positive lobe)
    if (!isUpPeak && z > thUp) {
        isUpPeak = true;
        isDownPeak = false; // toggle
        maxAccZ = std::max(maxAccZ, z);
        lastUpTime = tMs;
        if (segStartTime < 0) segStartTime = tMs;
    } else if (isUpPeak) {
        maxAccZ = std::max(maxAccZ, z);
    }

    // Down crossing (negative lobe)
    if (!isDownPeak && z < thDown) {
        isDownPeak = true;
        isUpPeak = false;   // toggle
        minAccZ = std::min(minAccZ, z);
        lastDownTime = tMs;
    } else if (isDownPeak) {
        minAccZ = std::min(minAccZ, z);
    }
}

bool SLRequire::isValidStepCandidate(int64_t /*nowMs*/) const {
    // onHand-style validity: order, amplitude, peak-to-peak interval
    const bool orderOK = (lastUpTime >= 0 && lastDownTime >= 0 && lastDownTime >= lastUpTime);
    const float zDiff = (maxAccZ - minAccZ);
    const bool zOK = (zDiff >= minZDiff);

    int64_t p2p = -1;
    if (lastUpTime >= 0 && lastDownTime >= 0) {
        p2p = std::llabs(lastDownTime - lastUpTime);
    }
    const bool intervalOK = (p2p >= 0 ? (p2p >= minP2Pms) : true);

    return orderOK && zOK && intervalOK;
}

SLRequire& SLRequire_Instance() {
    static SLRequire g;
    return g;
}