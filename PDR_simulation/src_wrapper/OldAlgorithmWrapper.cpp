// src_wrapper/OldAlgorithmWrapper.cpp
// Wraps all old PDR code into namespace Old{} via unity-build technique.
// Only this file is compiled — src_old/*.cpp are NOT listed in CMakeLists.

// ================================================================
// [1] Make JNI exports weak BEFORE any jni.h include.
//     Since mock/jni.h uses #ifndef JNIEXPORT, defining it here
//     first ensures the weak attribute survives all subsequent
//     #include <jni.h> directives in the src_old/ files.
//     This prevents linker collisions with the strong-symbol
//     extern "C" JNI functions compiled from src/.
// ================================================================
#define JNIEXPORT __attribute__((weak))

// ================================================================
// [2] Pre-include: system headers used by old sources
// ================================================================
#include <cmath>
#include <vector>
#include <deque>
#include <array>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <optional>
#include <queue>
#include <atomic>
#include <numeric>
#include <functional>
#include <mutex>
#include <ctime>
#include <sys/stat.h>
#include <android/log.h>
#include <jni.h>  // Pre-include to lock #pragma once with weak JNIEXPORT

// ================================================================
// [3] Pre-include SLRequire.h from include/ (new version)
//     — SLRequire.h uses #pragma once (path-based), so including the
//       include/ version first triggers #pragma once for that path.
//       When include_old/ version is included later (different path),
//       #pragma once sees a new path → it gets included too. Good.
//       But the src_old/*.cpp files do #include "StepLength/SLRequire.h"
//       which resolves to include/StepLength/SLRequire.h via include path,
//       and THAT one is already #pragma-once'd → skipped. Perfect.
// ================================================================
#include "StepLength/SLRequire.h"

// ================================================================
// [4] namespace Old — all old code lives here
// ================================================================
namespace Old {

// --- 4a. Headers from include_old/ (sets guards) ---
#include "include_old/Filter/MovingAverage.h"
#include "include_old/Filter/AdaptiveFilter.h"
#include "include_old/PDRresult.h"
#include "include_old/Sensor/SensorManager.h"
#include "include_old/Sensor/CoordinateTransform.h"
#include "include_old/StepLength/SLRequire.h"
#include "include_old/StepLength/StepLength.h"
#include "include_old/StepDetection/onHand.h"
#include "include_old/StepDetection/handSwing.h"
#include "include_old/StepDetection/inPocket.h"

// --- 4b. Source files from src_old/ ---
// Filter (no LOG_TAG)
#include "src_old/Filter/MovingAverage.cpp"
#include "src_old/Filter/AdaptiveFilter.cpp"

// Sensor/CoordinateTransform.cpp defines LOG_TAG
#undef LOG_TAG
#undef LOGD
#include "src_old/Sensor/CoordinateTransform.cpp"

// Sensor/SensorManager.cpp redefines LOG_TAG
#undef LOG_TAG
#undef LOGD
#include "src_old/Sensor/SensorManager.cpp"

// StepLength (SLRequire has no LOG_TAG)
#undef LOG_TAG
#undef LOGD
#include "src_old/StepLength/SLRequire.cpp"

// StepLength.cpp defines LOG_TAG and also defines a local class
// MovingAverage that conflicts with Filter/MovingAverage.h.
// Rename it to avoid redefinition within this namespace.
#undef LOG_TAG
#undef LOGD
#define MovingAverage MovingAverage_StepLength
#include "src_old/StepLength/StepLength.cpp"
#undef MovingAverage

// StepDetection/onHand.cpp defines LOG_TAG and now_ms()
#undef LOG_TAG
#undef LOGD
#define now_ms now_ms_onhand
#include "src_old/StepDetection/onHand.cpp"
#undef now_ms

// StepDetection/handSwing.cpp defines now_ms()
#define now_ms now_ms_handswing
#include "src_old/StepDetection/handSwing.cpp"
#undef now_ms

// StepDetection/inPocket.cpp defines now_ms()
#define now_ms now_ms_inpocket
#include "src_old/StepDetection/inPocket.cpp"
#undef now_ms

} // namespace Old

// Clean up macros leaked from old code
#undef LOG_TAG
#undef LOGD
#undef JNIEXPORT
#define JNIEXPORT

// ================================================================
// [5] Wrapper API implementation
// ================================================================
#include "OldAlgorithmWrapper.h"

struct OldPDRContext {
    Old::SensorManager* sensorManager;
    Old::OnHandStepDetection* detector;
    std::deque<float> stepQueue;
    std::array<float, 3> rotangle;

    OldPDRContext()
        : sensorManager(nullptr)
        , detector(nullptr)
        , stepQueue({0.65f, 0.65f, 0.65f, 0.65f})
        , rotangle({0.f, 0.f, 0.f})
    {}
};

OldPDRContext* Old_createContext() {
    auto* ctx = new OldPDRContext();
    ctx->sensorManager = Old::getSensorManager();
    ctx->detector = new Old::OnHandStepDetection();
    return ctx;
}

void Old_destroyContext(OldPDRContext* ctx) {
    if (!ctx) return;
    delete ctx->detector;
    delete ctx;
}

void Old_initSensor(OldPDRContext* ctx) {
    if (!ctx) return;
    ctx->sensorManager = Old::getSensorManager();
}

void Old_feedSensorSample(OldPDRContext* ctx, float globalZ, int64_t tsMs) {
    if (!ctx || !ctx->sensorManager) return;
    float linAcc[3] = {0.0f, 0.0f, globalZ};
    ctx->sensorManager->updateLinearAccelerometer(linAcc, (long)tsMs);

    Old::StepLength_onSensorSample(static_cast<double>(globalZ), tsMs);
}

void Old_feedRotationVector(OldPDRContext* ctx, float rotV[4], long timestamp) {
    if (!ctx || !ctx->sensorManager) return;
    ctx->sensorManager->updateRotationVector(rotV, timestamp);
}

bool Old_runStepDetection(OldPDRContext* ctx, float gyroAngleDeg, int64_t tsMs) {
    if (!ctx || !ctx->detector) return false;
    ctx->rotangle[2] = gyroAngleDeg;
    return ctx->detector->isStep(ctx->rotangle, ctx->stepQueue, tsMs, 3);
}

void Old_onStepDetected(OldPDRContext* ctx, int64_t tsMs) {
    if (!ctx) return;
    // Bridge peak/valley from old detector to old SLRequire
    Old::SLRequire_Instance().setPeakValley(
        (float)ctx->detector->getLastPeakZ(),
        (float)ctx->detector->getLastValleyZ()
    );
    Old::StepLength_onStepDetected(tsMs);
}

double Old_getLastStepLength(OldPDRContext* ctx) {
    (void)ctx;
    return Old::StepLength_getLastStepLength();
}

double Old_getLastRawStepLength(OldPDRContext* ctx) {
    (void)ctx;
    return Old::StepLength_getLastRawStepLength();
}

double Old_getLastAmplitudeZ(OldPDRContext* ctx) {
    (void)ctx;
    return Old::StepLength_getLastAmplitudeZ();
}

double Old_getLastFrequencyHz(OldPDRContext* ctx) {
    (void)ctx;
    return Old::StepLength_getLastFrequencyHz();
}

double Old_getLastPeakZ(OldPDRContext* ctx) {
    (void)ctx;
    return Old::StepLength_getLastPeakZ();
}

double Old_getLastValleyZ(OldPDRContext* ctx) {
    (void)ctx;
    return Old::StepLength_getLastValleyZ();
}

void Old_setScale(double s) {
    Old::StepLength_setScale(s);
}

void Old_resetGlobals() {
    Old::getSensorManager()->reset();
    Old::SLRequire_Instance().resetAll();
}
