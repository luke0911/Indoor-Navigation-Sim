// src_wrapper/OldAlgorithmWrapper.h
// Opaque API for Old PDR algorithm — no Old:: namespace leaks
#pragma once

#include <cstdint>

// Opaque handle for old step detector
struct OldPDRContext;

// Lifecycle
OldPDRContext* Old_createContext();
void Old_destroyContext(OldPDRContext* ctx);

// Sensor feed (same data as new pipeline)
void Old_initSensor(OldPDRContext* ctx);
void Old_feedSensorSample(OldPDRContext* ctx, float globalZ, int64_t tsMs);
void Old_feedRotationVector(OldPDRContext* ctx, float rotV[4], long timestamp);

// Step detection
bool Old_runStepDetection(OldPDRContext* ctx, float gyroAngleDeg, int64_t tsMs);

// Step length (call after step detected)
void Old_onStepDetected(OldPDRContext* ctx, int64_t tsMs);
double Old_getLastStepLength(OldPDRContext* ctx);
double Old_getLastRawStepLength(OldPDRContext* ctx);

// Step metrics
double Old_getLastAmplitudeZ(OldPDRContext* ctx);
double Old_getLastFrequencyHz(OldPDRContext* ctx);
double Old_getLastPeakZ(OldPDRContext* ctx);
double Old_getLastValleyZ(OldPDRContext* ctx);

// Configuration
void Old_setScale(double s);

// Reset old global singletons (for data reload)
void Old_resetGlobals();
