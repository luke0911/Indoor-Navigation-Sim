#include "Sensor/CoordinateTransform.h"
#include <android/log.h>
#include <cmath>

#define LOG_TAG "CoordinateTransform"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// === CoordinateTransform 클래스 구현 ===

CoordinateTransform::CoordinateTransform() {
    LOGD("CoordinateTransform 초기화 this=%p", this);
}

CoordinateTransform::~CoordinateTransform() {
    LOGD("CoordinateTransform 소멸");
}

// === Rotation Vector에서 쿼터니언 업데이트 ===
void CoordinateTransform::updateFromRotationVector(double x, double y, double z, double w, double timestamp) {
    quaternion[0] = x;
    quaternion[1] = y;
    quaternion[2] = z;
    quaternion[3] = w;
    isQuaternionValid.store(true, std::memory_order_release);
//    LOGD("[QuaternionUpdate] this=%p, valid=%d, ts=%.3f", this, isQuaternionValid.load(std::memory_order_acquire) ? 1 : 0, timestamp);

    // 오일러 각도 계산
    calculateEulerAngles(timestamp);

//    LOGD("[Quaternion] x=%.4f, y=%.4f, z=%.4f, w=%.4f", x, y, z, w);
}

// === SensorManager에서 자동으로 rotation vector 가져와서 업데이트 ===
void CoordinateTransform::updateFromSensorManager() {
    SensorManager* sm = getSensorManager();

    if (sm && sm->isRotationVectorReady()) {
        auto rotData = sm->getLatestRotationVector();
        updateFromRotationVector(rotData.x, rotData.y, rotData.z, rotData.w, rotData.timestamp);
    }
}

// === 쿼터니언에서 오일러 각도 계산 ===
void CoordinateTransform::calculateEulerAngles(double timestamp) {
    if (!isQuaternionValid.load(std::memory_order_acquire)) return;

    double qx = quaternion[0], qy = quaternion[1], qz = quaternion[2], qw = quaternion[3];

    // Roll (X축 회전)
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    double roll = atan2(sinr_cosp, cosr_cosp);

    // Pitch (Y축 회전)
    double sinp = 2.0 * (qw * qy - qz * qx);
    double pitch;
    if (abs(sinp) >= 1) {
        pitch = copysign(M_PI / 2, sinp); // 90도 제한
    } else {
        pitch = asin(sinp);
    }

    // Yaw (Z축 회전)
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    double yaw = atan2(siny_cosp, cosy_cosp);

    // 초기 yaw 오프셋 설정 (첫 번째 측정값 기준)
    if (!isYawOffsetInitialized) {
        yawOffset = -yaw;
        isYawOffsetInitialized = true;
//        LOGD("Yaw 오프셋 초기화: %.3f도", yawOffset * 180.0 / M_PI);
    }

    // 오프셋 적용된 yaw
    double correctedYaw = yaw + yawOffset;

    // 0~2π 범위로 정규화
    while (correctedYaw < 0) correctedYaw += 2.0 * M_PI;
    while (correctedYaw >= 2.0 * M_PI) correctedYaw -= 2.0 * M_PI;

    currentEulerAngles = {roll, pitch, correctedYaw, timestamp};

//    LOGD("[EulerAngles] Roll=%.1f°, Pitch=%.1f°, Yaw=%.1f°",
//         roll * 180.0 / M_PI, pitch * 180.0 / M_PI, correctedYaw * 180.0 / M_PI);
}

// === 글로벌 좌표계로 가속도 변환 ===
TransformedAcceleration CoordinateTransform::transformToGlobal(double localX, double localY, double localZ, double timestamp) {
    if (!isQuaternionValid.load(std::memory_order_acquire)){
        LOGD("쿼터니언이 유효하지 않음 - 변환 불가");
        return {localX, localY, localZ, timestamp}; // 원본 반환
    }

    double qx = quaternion[0], qy = quaternion[1], qz = quaternion[2], qw = quaternion[3];

    // 쿼터니언을 이용한 3D 벡터 회전
    // v' = q * v * q^(-1)
    // 여기서는 회전 행렬로 변환해서 계산

    // 회전 행렬 계산
    double r11 = 1.0 - 2.0 * (qy * qy + qz * qz);
    double r12 = 2.0 * (qx * qy - qw * qz);
    double r13 = 2.0 * (qx * qz + qw * qy);

    double r21 = 2.0 * (qx * qy + qw * qz);
    double r22 = 1.0 - 2.0 * (qx * qx + qz * qz);
    double r23 = 2.0 * (qy * qz - qw * qx);

    double r31 = 2.0 * (qx * qz - qw * qy);
    double r32 = 2.0 * (qy * qz + qw * qx);
    double r33 = 1.0 - 2.0 * (qx * qx + qy * qy);

    // 로컬 좌표를 글로벌 좌표로 변환
    double globalX = r11 * localX + r12 * localY + r13 * localZ;
    double globalY = r21 * localX + r22 * localY + r23 * localZ;
    double globalZ = r31 * localX + r32 * localY + r33 * localZ;

//    LOGD("[Transform] Local(%.3f,%.3f,%.3f) → Global(%.3f,%.3f,%.3f)",
//         localX, localY, localZ, globalX, globalY, globalZ);

    return {globalX, globalY, globalZ, timestamp};
}

// === SensorManager에서 선형가속도 가져와서 변환 ===
TransformedAcceleration CoordinateTransform::transformLinearAccelerationFromSensorManager() {
    SensorManager* sm = getSensorManager();
    if (sm && sm->isLinearAccelerometerReady()) {
        auto linAccData = sm->getLatestLinearAccelerometer();
        return transformToGlobal(linAccData.x, linAccData.y, linAccData.z, linAccData.timestamp);
    }
    return {0.0, 0.0, 0.0, 0.0};
}

// === 특정 축만 변환 (StepDetection에서 Z축만 필요할 때) ===
double CoordinateTransform::transformAxis(char axis, double localX, double localY, double localZ) {
    if (!isQuaternionValid.load(std::memory_order_acquire)) return 0.0;

    TransformedAcceleration global = transformToGlobal(localX, localY, localZ, 0);

    switch (axis) {
        case 'x': case 'X': return global.x;
        case 'y': case 'Y': return global.y;
        case 'z': case 'Z': return global.z;
        default: return 0.0;
    }
}

// === 접근자 함수들 ===

EulerAngles CoordinateTransform::getCurrentEulerAngles() const {
    return currentEulerAngles;
}

double CoordinateTransform::getRoll() const {
    return currentEulerAngles.roll;
}

double CoordinateTransform::getPitch() const {
    return currentEulerAngles.pitch;
}

double CoordinateTransform::getYaw() const {
    return currentEulerAngles.yaw;
}

// 각도를 도(degree) 단위로 반환
double CoordinateTransform::getRollDegrees() const {
    return currentEulerAngles.roll * 180.0 / M_PI;
}

double CoordinateTransform::getPitchDegrees() const {
    return currentEulerAngles.pitch * 180.0 / M_PI;
}

double CoordinateTransform::getYawDegrees() const {
    return currentEulerAngles.yaw * 180.0 / M_PI;
}

std::array<double, 4> CoordinateTransform::getQuaternion() const {
    return {quaternion[0], quaternion[1], quaternion[2], quaternion[3]};
}


bool CoordinateTransform::isReady() const {
    bool ready = isQuaternionValid.load(std::memory_order_acquire);
//    LOGD("[isReady] this=%p, isQuaternionValid=%d", this, ready ? 1 : 0);
    return ready;
}
// === 리셋 함수 ===

void CoordinateTransform::reset() {
    quaternion[0] = quaternion[1] = quaternion[2] = 0.0;
    quaternion[3] = 1.0;
    isQuaternionValid.store(false, std::memory_order_release);
    currentEulerAngles = {0.0, 0.0, 0.0, 0.0};
    yawOffset = 0.0;
    isYawOffsetInitialized = false;
//    LOGD("CoordinateTransform 리셋 완료 this=%p, valid=%d", this, isQuaternionValid.load(std::memory_order_acquire) ? 1 : 0);
}

// === 수동 yaw 오프셋 설정 (캘리브레이션용) ===

void CoordinateTransform::setYawOffset(double offsetRadians) {
    yawOffset = offsetRadians;
    isYawOffsetInitialized = true;
//    LOGD("Yaw 오프셋 수동 설정: %.3f도", offsetRadians * 180.0 / M_PI);
}

void CoordinateTransform::calibrateYaw() {
    if (isQuaternionValid.load(std::memory_order_acquire)) {
        yawOffset = -currentEulerAngles.yaw;
        isYawOffsetInitialized = true;
//        LOGD("Yaw 캘리브레이션 완료: %.3f도", yawOffset * 180.0 / M_PI);
    }
}

