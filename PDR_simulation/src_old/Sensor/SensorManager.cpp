#include "Sensor/SensorManager.h"
#include "StepLength/SLRequire.h"
#include <android/log.h>
#include <chrono>
#include <cmath>
#include <jni.h>
#include <cstdint>

#define LOG_TAG "SensorManager"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

extern SensorManager* getSensorManager();


// === SensorManager 클래스 구현 ===

double SensorManager::getCurrentTimestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

SensorManager::SensorManager() {
    LOGD("SensorManager 초기화");
}

SensorManager::~SensorManager() {
    LOGD("SensorManager 소멸");
}

// 콜백 등록/해제
void SensorManager::registerCallback(ISensorDataCallback* callback) {
    if (callback) {
        callbacks.push_back(callback);
        LOGD("센서 콜백 등록 완료");
    }
}

void SensorManager::unregisterCallback(ISensorDataCallback* callback) {
    auto it = std::find(callbacks.begin(), callbacks.end(), callback);
    if (it != callbacks.end()) {
        callbacks.erase(it);
        LOGD("센서 콜백 해제 완료");
    }
}

// === 센서 데이터 입력 함수들 ===

void SensorManager::updateAccelerometer(float acc[3], long timestamp) {
    if (timestamp == 0) timestamp = long(getCurrentTimestamp());

    latestAccelerometer = {acc[0], acc[1], acc[2], timestamp};
    hasAccelerometerData = true;

//    LOGD("[Accelerometer] x=%.3f, y=%.3f, z=%.3f, time=%ldf", acc[0], acc[1], acc[2], timestamp);

    // 모든 콜백에 전달
    for (auto* callback : callbacks) {
        callback->onAccelerometerData(latestAccelerometer);
    }
}

void SensorManager::updateLinearAccelerometer(float linacc[3], long timestamp) {
    if (timestamp == 0) timestamp = long(getCurrentTimestamp());

    latestLinearAccelerometer = {linacc[0], linacc[1], linacc[2], timestamp};
    hasLinearAccelerometerData = true;

//    LOGD("[LinearAccel] x=%.3f, y=%.3f, z=%.3f, time=%ld", linacc[0], linacc[1], linacc[2], timestamp);

    // Feed global Z acceleration into SLRequire (Peak/Valley detector)
    {
        float globalZ = linacc[2]; // assuming z-axis is global vertical here
        SLRequire_Instance().feed(globalZ, static_cast<int64_t>(timestamp));
    }

    // 모든 콜백에 전달
    for (auto* callback : callbacks) {
        callback->onLinearAccelerometerData(latestLinearAccelerometer);
    }
}

void SensorManager::updateRotationVector(float rotV[4], long timestamp) {
    if (timestamp == 0) timestamp = long(getCurrentTimestamp());

    latestRotationVector = {rotV[0], rotV[1], rotV[2], rotV[3], timestamp};
    hasRotationVectorData = true;

    getRotationFromQuaternion(latestRotationVector);
    // 모든 콜백에 전달
    for (auto* callback : callbacks) {
        callback->onRotationVectorData(latestRotationVector);
    }
}

void SensorManager::updateLightSensor(float lux, long timestamp) {
    if (timestamp == 0) timestamp = long(getCurrentTimestamp());

    latestLight = {lux, timestamp};
    hasLightSensorData = true;

//    LOGD("[Light] lux=%.1f, time=%ld", lux, timestamp);

    // 모든 콜백에 전달
    for (auto* callback : callbacks) {
        callback->onLightSensorData(latestLight);
    }
}

void SensorManager::updatePressureSensor(float pressure, long timestamp) {
    if (timestamp == 0) timestamp = long(getCurrentTimestamp());

    latestPressure = {pressure, timestamp};
    hasPressureData = true;

//    LOGD("[Pressure] pressure=%.2f hPa, time=%ld", pressure, timestamp);

    // 모든 콜백에 전달
    for (auto* callback : callbacks) {
        callback->onPressureSensorData(latestPressure);
    }
}

void SensorManager::getRotationFromQuaternion(RotationVectorData q) {
    const double x = q.x;
    const double y = q.y;
    const double z = q.z;
    const double w = q.w;

    // roll (x-axis)
    const double roll  = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));

    // pitch (y-axis) with clamp
    double t = 2.0 * (w * y - z * x);
    if (t > 1.0) t = 1.0;
    if (t < -1.0) t = -1.0;
    const double pitch = std::asin(t);

    // yaw (z-axis)
    const double yaw   = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));

    constexpr double RAD2DEG = 57.29577951308232;
    g_cachedRotationAngle = RotationAngle{ static_cast<float>(pitch * RAD2DEG),
                                           static_cast<float>(roll * RAD2DEG),
                                           static_cast<float>(yaw * RAD2DEG)};

//    LOGD("[RotAngle] roll=%.2f, pitch=%.2f, yaw=%.2f",
         (float)g_cachedRotationAngle.roll,
         (float)g_cachedRotationAngle.pitch,
         (float)g_cachedRotationAngle.yaw;
}

// === 데이터 접근 함수들 ===

AccelerometerData SensorManager::getLatestAccelerometer() const {
    return latestAccelerometer;
}

LinearAccelerometerData SensorManager::getLatestLinearAccelerometer() const {
    return latestLinearAccelerometer;
}

RotationVectorData SensorManager::getLatestRotationVector() const {
    return latestRotationVector;
}

LightSensorData SensorManager::getLatestLight() const {
    return latestLight;
}

PressureSensorData SensorManager::getLatestPressure() const {
    return latestPressure;
}

RotationAngle SensorManager::getRotangle() const {
    return g_cachedRotationAngle; // if not ready yet
}

// === 상태 확인 함수들 ===

bool SensorManager::isAccelerometerReady() const {
    return hasAccelerometerData;
}

bool SensorManager::isLinearAccelerometerReady() const {
    return hasLinearAccelerometerData;
}

bool SensorManager::isRotationVectorReady() const {
    return hasRotationVectorData;
}

bool SensorManager::isLightSensorReady() const {
    return hasLightSensorData;
}

bool SensorManager::isPressureSensorReady() const {
    return hasPressureData;
}

bool SensorManager::isAllSensorsReady() const {
    return hasAccelerometerData && hasLinearAccelerometerData && hasRotationVectorData;
}

// === 리셋 함수 ===

void SensorManager::reset() {
    hasAccelerometerData = false;
    hasLinearAccelerometerData = false;
    hasRotationVectorData = false;
    hasLightSensorData = false;
    hasPressureData = false;

    LOGD("SensorManager 리셋 완료");
}

extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_SMcpp_updateAccelerometer(JNIEnv* env, jobject /*thiz*/,
                                                    jfloatArray acc, jlong tMillis) {
    if (!acc) return;
    jsize len = env->GetArrayLength(acc);
    jfloat buf[3] = {0.f, 0.f, 0.f};
    const jsize n = (len < 3 ? len : 3);
    env->GetFloatArrayRegion(acc, 0, n, buf);
    getSensorManager()->updateAccelerometer(buf, static_cast<long>(tMillis));
}


extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_SMcpp_updateLinearAccelerometer(JNIEnv* env, jobject /*thiz*/,
                                                          jfloatArray linacc, jlong tMillis) {
    if (!linacc) return;
    jsize len = env->GetArrayLength(linacc);
    jfloat buf[3] = {0.f, 0.f, 0.f};
    const jsize n = (len < 3 ? len : 3);
    env->GetFloatArrayRegion(linacc, 0, n, buf);
    getSensorManager()->updateLinearAccelerometer(buf, static_cast<long>(tMillis));
}



extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_SMcpp_updateRotationQuaternion(JNIEnv* env, jobject /*thiz*/,
                                                         jfloatArray rotV, jlong tMillis) {
    if (!rotV) return;
    jsize len = env->GetArrayLength(rotV);
    jfloat buf[4] = {0.f, 0.f, 0.f, 0.0f};
    const jsize n = (len < 4 ? len : 4);
    env->GetFloatArrayRegion(rotV, 0, n, buf);
    getSensorManager()->updateRotationVector(buf, static_cast<long>(tMillis));
}

extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_SMcpp_updateLight(JNIEnv* env, jobject /*thiz*/, jfloat lux, jlong tMillis) {
    (void)env;
    getSensorManager()->updateLightSensor((float)lux, (long)tMillis);
}

extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_SMcpp_updatePressure(JNIEnv* env, jobject /*thiz*/, jfloat hPa, jlong tMillis) {
    (void)env;
    getSensorManager()->updatePressureSensor((float)hPa, (long)tMillis);
}

extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_SMcpp_onSensorEvent(JNIEnv* env, jobject /*thiz*/, jobject sensorEvent) {
    if (!sensorEvent) return;

    jclass clsEvent = env->GetObjectClass(sensorEvent);
    if (!clsEvent) return;

    // fields: float[] values; long timestamp; Sensor sensor; int accuracy;
    jfieldID fidValues    = env->GetFieldID(clsEvent, "values", "[F");
    jfieldID fidTimestamp = env->GetFieldID(clsEvent, "timestamp", "J");
    jfieldID fidSensor    = env->GetFieldID(clsEvent, "sensor", "Landroid/hardware/Sensor;");
    jfieldID fidAccuracy  = env->GetFieldID(clsEvent, "accuracy", "I");
    if (!fidValues || !fidTimestamp || !fidSensor) return;

    jlong  ts         = env->GetLongField(sensorEvent, fidTimestamp);
    jobject sensorObj = env->GetObjectField(sensorEvent, fidSensor);
    jfloatArray valuesArr = (jfloatArray)env->GetObjectField(sensorEvent, fidValues);

    if (!sensorObj || !valuesArr) return;

    jclass clsSensor = env->GetObjectClass(sensorObj);
    if (!clsSensor) return;

    // int getType()
    jmethodID midGetType = env->GetMethodID(clsSensor, "getType", "()I");
    if (!midGetType) return;
    jint type = env->CallIntMethod(sensorObj, midGetType);

    jsize len = env->GetArrayLength(valuesArr);
    jfloat* vals = env->GetFloatArrayElements(valuesArr, nullptr);
    if (!vals) return;

    // Android sensor type constants (subset)
    constexpr int TYPE_ACCELEROMETER      = 1;
    constexpr int TYPE_MAGNETIC_FIELD     = 2;
    constexpr int TYPE_GYROSCOPE          = 4;
    constexpr int TYPE_LIGHT              = 5;
    constexpr int TYPE_PRESSURE           = 6;
    constexpr int TYPE_GRAVITY            = 9;
    constexpr int TYPE_LINEAR_ACCELERATION= 10;
    constexpr int TYPE_ROTATION_VECTOR    = 11;

    switch (type) {
        case TYPE_ACCELEROMETER: {
            float acc[3] = {0.f, 0.f, 0.f};
            for (int i = 0; i < 3 && i < len; ++i) acc[i] = vals[i];
            getSensorManager()->updateAccelerometer(acc, static_cast<long>(ts));
            break;
        }
        case TYPE_LINEAR_ACCELERATION: {
            float la[3] = {0.f, 0.f, 0.f};
            for (int i = 0; i < 3 && i < len; ++i) la[i] = vals[i];
            getSensorManager()->updateLinearAccelerometer(la, static_cast<long>(ts));
            break;
        }
        case TYPE_ROTATION_VECTOR: {
            float rv[4] = {0.f, 0.f, 0.f, 0.f};
            // Android의 rotation vector는 x,y,z,(w optional) 순
            for (int i = 0; i < 4 && i < len; ++i) rv[i] = vals[i];
            // w가 비어 있으면 0으로 둠 (SensorManager 쪽에서 처리함)
            getSensorManager()->updateRotationVector(rv, static_cast<long>(ts));
            break;
        }
        case TYPE_LIGHT: {
            float lux = (len > 0 ? vals[0] : 0.f);
            getSensorManager()->updateLightSensor(lux, static_cast<long>(ts));
            break;
        }
        case TYPE_PRESSURE: {
            float hPa = (len > 0 ? vals[0] : 0.f);
            getSensorManager()->updatePressureSensor(hPa, static_cast<long>(ts));
            break;
        }
        default:
            // 다른 타입은 현재 사용하지 않음
            break;
    }

    env->ReleaseFloatArrayElements(valuesArr, vals, JNI_ABORT);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_fifth_pdr_1ext_SMcpp_getRotationAnglesDeg(JNIEnv* env, jobject /*thiz*/) {
    const RotationAngle a = getSensorManager()->getRotangle();
    jfloat out[3] = { (jfloat)a.roll, (jfloat)a.pitch, (jfloat)a.yaw };
    jfloatArray arr = env->NewFloatArray(3);
    if (!arr) return nullptr;
    env->SetFloatArrayRegion(arr, 0, 3, out);
    return arr;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_fifth_pdr_1ext_SMcpp_isAllSensorsReady(JNIEnv* env, jobject /*thiz*/) {
    (void)env;
    const bool ready = getSensorManager()->isAllSensorsReady();
    return ready ? JNI_TRUE : JNI_FALSE;
}

// === 전역 SensorManager 인스턴스 ===

static SensorManager* g_sensorManager = nullptr;

SensorManager* getSensorManager() {
    if (!g_sensorManager) {
        g_sensorManager = new SensorManager();
    }
    return g_sensorManager;
}