#ifndef COORDINATE_TRANSFORM_H
#define COORDINATE_TRANSFORM_H

#include "SensorManager.h"
#include <array>
#include <atomic>

struct EulerAngles {
    double roll;
    double pitch;
    double yaw;
    double timestamp;
};

struct TransformedAcceleration {
    double x, y, z;
    double timestamp;
};

class CoordinateTransform {
private:
    double quaternion[4] = {0.0, 0.0, 0.0, 1.0};
    std::atomic<bool> isQuaternionValid{false};
    EulerAngles currentEulerAngles = {0.0, 0.0, 0.0, 0.0};

    double yawOffset = 0.0;
    bool isYawOffsetInitialized = false;

public:
    CoordinateTransform();
    ~CoordinateTransform();

    void updateFromRotationVector(double x, double y, double z, double w, double timestamp);
    void updateFromSensorManager();
    void calculateEulerAngles(double timestamp);
    TransformedAcceleration transformToGlobal(double localX, double localY, double localZ, double timestamp);
    TransformedAcceleration transformLinearAccelerationFromSensorManager();
    double transformAxis(char axis, double localX, double localY, double localZ);

    EulerAngles getCurrentEulerAngles() const;
    double getRoll() const;
    double getPitch() const;
    double getYaw() const;
    double getRollDegrees() const;
    double getPitchDegrees() const;
    double getYawDegrees() const;
    std::array<double, 4> getQuaternion() const;
    bool isReady() const;

    void reset();
    void setYawOffset(double offsetRadians);
    void calibrateYaw();
};

inline CoordinateTransform* getSharedCoordinateTransform() {
    static CoordinateTransform instance;
    return &instance;
}

#endif // COORDINATE_TRANSFORM_H
