#ifndef SENSOR_MANAGER_H
#define SENSOR_MANAGER_H

#include <vector>
#include <algorithm>

struct AccelerometerData {
    float x, y, z;
    long timestamp;
};

struct LinearAccelerometerData {
    float x, y, z;
    long timestamp;
};

struct RotationVectorData {
    float x, y, z, w;
    long timestamp;
};

struct LightSensorData {
    float lux;
    long timestamp;
};

struct PressureSensorData {
    float pressure;
    long timestamp;
};

struct RotationAngle{
    float pitch;
    float roll;
    float yaw;
};

static RotationAngle g_cachedRotationAngle{0.0, 0.0, 0.0};

class ISensorDataCallback {
public:
    virtual ~ISensorDataCallback() = default;
    virtual void onAccelerometerData(const AccelerometerData& data) = 0;
    virtual void onLinearAccelerometerData(const LinearAccelerometerData& data) = 0;
    virtual void onRotationVectorData(const RotationVectorData& data) = 0;
    virtual void onLightSensorData(const LightSensorData& data) = 0;
    virtual void onPressureSensorData(const PressureSensorData& data) = 0;
};

class SensorManager {
private:
    AccelerometerData latestAccelerometer{0, 0, 0, 0};
    LinearAccelerometerData latestLinearAccelerometer{0, 0, 0, 0};
    RotationVectorData latestRotationVector{0, 0, 0, 1, 0};
    LightSensorData latestLight{0, 0};
    PressureSensorData latestPressure{0, 0};
    RotationAngle latestRotangle{0,0,0};
    std::vector<ISensorDataCallback*> callbacks;

    bool hasAccelerometerData = false;
    bool hasLinearAccelerometerData = false;
    bool hasRotationVectorData = false;
    bool hasLightSensorData = false;
    bool hasPressureData = false;

    static double getCurrentTimestamp();

public:
    SensorManager();
    ~SensorManager();

    void registerCallback(ISensorDataCallback* callback);
    void unregisterCallback(ISensorDataCallback* callback);

    void updateAccelerometer(float acc[3], long timestamp = 0);
    void updateLinearAccelerometer(float linacc[3], long timestamp = 0);
    void updateRotationVector(float rotV[4], long timestamp = 0);
    void updateLightSensor(float lux, long timestamp = 0);
    void updatePressureSensor(float pressure, long timestamp = 0);

    static void getRotationFromQuaternion(RotationVectorData);

    AccelerometerData getLatestAccelerometer() const;
    LinearAccelerometerData getLatestLinearAccelerometer() const;
    RotationVectorData getLatestRotationVector() const;
    LightSensorData getLatestLight() const;
    PressureSensorData getLatestPressure() const;
    RotationAngle getRotangle() const;

    bool isAccelerometerReady() const;
    bool isLinearAccelerometerReady() const;
    bool isRotationVectorReady() const;
    bool isLightSensorReady() const;
    bool isPressureSensorReady() const;
    bool isAllSensorsReady() const;

    void reset();
};

SensorManager* getSensorManager();

#endif // SENSOR_MANAGER_H
