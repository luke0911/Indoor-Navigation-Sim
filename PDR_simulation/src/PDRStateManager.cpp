#include <array>
#include <deque>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>
#include <jni.h>
#include <android/log.h>
#include <vector>

#define LOG_TAG "PDRStateManager"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// PDR 결과 구조체 정의
#include "PDRresult.h"
// 헤더 포함으로 전환 (소스는 CMake에 등록)
#include "StepDetection/handSwing.h"
#include "StepDetection/inPocket.h"
#include "StepDetection/onHand.h"
#include "Sensor/SensorManager.h"
#include "StepLength/StepLength.h"
#include "StepLength/SLRequire.h"


extern SensorManager* getSensorManager();

// 현재 시간(ms) 유틸리티 함수
static inline int64_t now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// PDRStateManager
// - 센서 입력을 저장하고, 상태(state)에 따라 적절한 스텝 검출기를 호출
// - 각 모드별 getStatus()를 통해 PDR 결과를 반환
class PDRStateManager {
public:
    explicit PDRStateManager()
            : statereal(0), statetmp(0),
              rotangle{0.f,0.f,0.f}, linacc{0.f,0.f,0.f}, acc{0.f,0.f,0.f},
              stepQueue{0.65f,0.65f,0.65f,0.65f}, headQueue{0.f,0.f,0.f,0.f},
              step_time(0), past_step_time(0), time_line{99,99,99}, avgFreq(std::numeric_limits<int64_t>::max()),
              stepLength(0.f), stepCount(0), mapBasedGyroAngle(0.f), angletmp(0.f),
              pdrResult{0.0,0}
    {}

    // 회전각/쿼터니언 설정 (현재는 회전각만 저장)
    void set_rot(const std::array<float,3>& inrot, const std::array<float,4>& /*rotQ*/)
    {
        // lateruser (Kotlin stub) — only store rotangle here.
        rotangle = inrot;
    }

    // 가속도(중력 포함) 벡터 설정
    void set_acc(const std::array<float,3>& inacc)    { acc = inacc; }
    // 선형가속도(중력 제거) 벡터 설정
    void set_linacc(const std::array<float,3>& inlinacc) { linacc = inlinacc; }

    // 스텝 길이 큐 업데이트 (front가 최신)
    void add_stepqueue(float num) { stepQueue.push_back(num); stepQueue.pop_front(); }
    // 최신 스텝 길이 조회 (큐 비었으면 0)
    float get_stepqueue_peek() const { return stepQueue.empty() ? 0.f : stepQueue.front(); }
    // 스텝 길이 큐 전체 반환
    const std::deque<float>& get_stepqueue() const { return stepQueue; }

    // head(방향 보조) 큐 업데이트
    void add_headqueue(float num) { headQueue.push_back(num); headQueue.pop_front(); }
    // 최신 head 값 조회
    float get_headqueue_peek() const { return headQueue.empty() ? 0.f : headQueue.front(); }
    // head 큐 전체 반환
    const std::deque<float>& get_headqueue() const { return headQueue; }

    double getLastRawGlobalZ() const { return lastRawGlobalZ_; }

    // 현재 상태(statetmp)에 따라 해당 스텝 검출기 실행
    bool isstep(int tmp, int real)
    {
        statetmp = tmp;
        statereal = real;

        // Pull latest sensors from SensorManager
        if (getSensorManager()) {
            // Rotation angles (SensorManager returns {roll, pitch, yaw} in degrees)
            const RotationAngle a = getSensorManager()->getRotangle();
            // PDRStateManager expects {pitch, roll, yaw}
            rotangle = { static_cast<float>(a.pitch), static_cast<float>(a.roll), static_cast<float>(a.yaw) };

            // Accelerometer (gravity-included)
            AccelerometerData tmpacc = getSensorManager()->getLatestAccelerometer();
            acc = { tmpacc.x, tmpacc.y, tmpacc.z };
        }

        const int64_t t_ms = now_ms();
        bool step = false;

        switch (statetmp) {
            case 0: // 수평파지
            case 25:
            case 15:
            case 22: // 자켓주머니
                StepLength_setPocketMode(false);
                step = onhandPocket.isStep(rotangle, stepQueue, t_ms, statetmp);
                lastRawGlobalZ_ = onhandPocket.getLastRawGlobalZ();
                break;
            case 3:
                StepLength_setPocketMode(false);
                step = onhand.isStep(rotangle, stepQueue, t_ms, statetmp);
                lastRawGlobalZ_ = onhand.getLastRawGlobalZ();
                break;
            case 1: // 쥐고 손에 흔들기
                StepLength_setPocketMode(false);
                step = handheld.isStep(rotangle, stepQueue, t_ms, statetmp);
                break;
            case 32:
            case 2: // 주머니속
                StepLength_setPocketMode(true);
                step = inpocket.isStep(rotangle, stepQueue, t_ms, statetmp);
                break;

            default:
                step = false;
                break;
        }

        if (step) {
            // Write onHand peak/valley to SLRequire bridge before StepLength processes the step
            switch (statetmp) {
                case 0: case 25: case 15: case 22:
                    SLRequire_Instance().setPeakValley(
                        static_cast<float>(onhandPocket.getLastPeakZ()),
                        static_cast<float>(onhandPocket.getLastValleyZ()));
                    break;
                case 3:
                    SLRequire_Instance().setPeakValley(
                        static_cast<float>(onhand.getLastPeakZ()),
                        static_cast<float>(onhand.getLastValleyZ()));
                    break;
                // case 1, 2, 32: SLRequire의 자체 feed() 기반 peak/valley 유지
            }
            StepLength_onStepDetected(t_ms);
        }

        return step;
    }

    // 상태에 따른 PDR 결과 취합 및 반환
    PDR getresult()
    {
        step_time = now_ms();
        avgFreq = checkFreq();

        switch (statetmp) {
            case 0:
            case 22:
            case 3: {
                pdrResult = onhand.getStatus();
                {
                    const double L = StepLength_getLastStepLength();
                    pdrResult.stepLength = static_cast<float>(L);
                    add_stepqueue(static_cast<float>(L));

                }
                break;
            }

            case 2:
            case 32:
                pdrResult = inpocket.getStatus();
                {
                    const double L = StepLength_getLastStepLength();
                    pdrResult.stepLength = static_cast<float>(L);
                    add_stepqueue(static_cast<float>(L));
                }
                break;
            case 1:
                pdrResult = handheld.getStatus();
                {
                    const double L = StepLength_getLastStepLength();
                    pdrResult.stepLength = static_cast<float>(L);
                    add_stepqueue(static_cast<float>(L));
                }
                break;
            default:
                break;
        }

        past_step_time = step_time;
        LOGD("PDRResult: stepLength=%.3f, direction=%.3f, totalStepCount=%d",
             pdrResult.stepLength,
             pdrResult.direction,
             pdrResult.totalStepCount);
        return pdrResult;
    }

    // 최근 스텝 간격의 이동 평균(ms) 계산 (최근 3개)
    int64_t checkFreq()
    {
        const int64_t delta = step_time - past_step_time;
        if (delta != 0) {
            time_line.push_back(delta);
            time_line.pop_front();
        }
        const double avg = std::accumulate(time_line.begin(), time_line.end(), 0.0) / static_cast<double>(time_line.size());
        return static_cast<int64_t>(avg);
    }

private:
    // 내부 상태/버퍼
    int statereal;
    int statetmp;

    std::array<float,3> rotangle;    // 회전각 {pitch, roll, yaw}
    std::array<float,3> linacc;      // 선형가속도
    std::array<float,3> acc;         // 가속도(중력 포함)

    std::deque<float> stepQueue;     // 크기 4, front=최신
    std::deque<float> headQueue;     // 크기 4, front=최신

    int64_t step_time;               // 이번 스텝 시각(ms)
    int64_t past_step_time;          // 직전 스텝 시각(ms)
    std::deque<int64_t> time_line;   // 최근 3개 간격(ms)
    int64_t avgFreq;                 // 평균 스텝 주기(ms)

    float stepLength;                // 최근 스텝 길이
    int   stepCount;                 // 누적 스텝 수 (미사용 시 0)
    float mapBasedGyroAngle;         // 지도 보정 자이로 각(옵션)
    float angletmp;                  // 임시 각도(옵션)

    PDR pdrResult;                   // 최종 반환할 PDR 결과
    double lastRawGlobalZ_ = 0.0;    // 마지막 raw Global Z (MA 필터 전)

    InPocket inpocket;           // 주머니 모드 검출기
    HandSwing handheld;          // 손 흔들기 모드 검출기
    OnHandStepDetection onhand;                   // 수평 파지/손 모드 검출기
    OnHandStepDetection onhandPocket{0.1f, 0.8f, 30.0f}; // state 32 전용 어댑티브 필터
};

static PDRStateManager g_mgr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_fifth_pdr_1ext_PDRM_isStep(JNIEnv* env, jobject /*thiz*/,
                                    jint tmp, jint real) {
    (void)env; // unused
    const bool s = g_mgr.isstep(static_cast<int>(tmp), static_cast<int>(real));
    return s ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_fifth_pdr_1ext_PDRM_getresult(JNIEnv* env, jobject thiz) {
    (void)thiz;

    // Snapshot native result
    const PDR r = g_mgr.getresult();

    // Cache class & field IDs across calls
    static jclass s_cls = nullptr;
    static jmethodID s_ctor = nullptr;
    static jfieldID s_fidStep = nullptr;
    static jfieldID s_fidDir  = nullptr;
    static jfieldID s_fidCnt  = nullptr;

    if (!s_cls) {
        jclass local = env->FindClass("com/fifth/pdr_ext/PDR");
        if (env->ExceptionCheck() || !local) {
            env->ExceptionClear();
            return nullptr;
        }
        s_cls = static_cast<jclass>(env->NewGlobalRef(local));
        env->DeleteLocalRef(local);
        if (!s_cls) return nullptr;

        s_ctor    = env->GetMethodID(s_cls, "<init>", "()V");
        s_fidStep = env->GetFieldID(s_cls, "stepLength", "D");
        s_fidDir  = env->GetFieldID(s_cls, "direction",   "D");
        s_fidCnt  = env->GetFieldID(s_cls, "totalStepCount", "I");
        if (env->ExceptionCheck() || !s_ctor || !s_fidStep || !s_fidDir || !s_fidCnt) {
            env->ExceptionClear();
            return nullptr;
        }
    }

    jobject obj = env->NewObject(s_cls, s_ctor);
    if (env->ExceptionCheck() || !obj) {
        env->ExceptionClear();
        return nullptr;
    }

    // Write fields from native snapshot
    env->SetDoubleField(obj, s_fidStep, static_cast<jdouble>(r.stepLength));
    env->SetDoubleField(obj, s_fidDir,  static_cast<jdouble>(r.direction));
    env->SetIntField(obj,    s_fidCnt,  static_cast<jint>(r.totalStepCount));
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        return nullptr;
    }

    return obj;
}

extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_PDRM_add_1headqueue(JNIEnv* env, jobject /*thiz*/, jfloat headValue) {
    (void)env; // unused if no exceptions are thrown or JNIEnv methods are called
    g_mgr.add_headqueue(static_cast<float>(headValue));
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_fifth_pdr_1ext_PDRM_get_1headqueue_1peek(JNIEnv* env, jobject /*thiz*/) {
    (void)env; // unused
    return static_cast<jfloat>(g_mgr.get_headqueue_peek());
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_fifth_pdr_1ext_PDRM_get_1headqueue(JNIEnv* env, jobject /*thiz*/) {
    // Grab the native deque<float> snapshot
    const std::deque<float>& q = g_mgr.get_headqueue();

    // Create a Java float[] of the same size
    const jsize n = static_cast<jsize>(q.size());
    jfloatArray arr = env->NewFloatArray(n);
    if (!arr) {
        // OOM or failure; return null
        return nullptr;
    }

    // Copy elements into a contiguous buffer in newest→oldest order (front is latest)
    std::vector<jfloat> tmp;
    tmp.reserve(static_cast<size_t>(n));
    for (float v : q) {
        tmp.push_back(static_cast<jfloat>(v));
    }

    env->SetFloatArrayRegion(arr, 0, n, tmp.data());
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        return nullptr;
    }

    return arr;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_fifth_pdr_1ext_PDRM_getNowMs(JNIEnv* env, jobject /*thiz*/) {
    (void)env; // unused
    return static_cast<jlong>(now_ms());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_PDRM_getLastRawGlobalZ(JNIEnv* env, jobject /*thiz*/) {
    (void)env;
    return static_cast<jdouble>(g_mgr.getLastRawGlobalZ());
}
