#include "SimDataLogger.h"
#include <android/log.h>
#include <jni.h>
#include <fstream>
#include <mutex>
#include <string>
#include <chrono>
#include <ctime>
#include <sys/stat.h>

#define LOG_TAG "SimDataLogger"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// =========================================================
// 내부 상태
// =========================================================
static std::ofstream s_file;
static std::mutex    s_mutex;
static bool          s_enabled = false;
static bool          s_initAttempted = false;
static int64_t       s_firstMs = 0;

static const char* DEFAULT_APP_DIR = "/data/data/testapplication/files";

// =========================================================
// 유틸리티
// =========================================================
static bool ensureDir(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    return mkdir(dir.c_str(), 0755) == 0;
}

static std::string makeTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    struct tm* tmNow = localtime(&t);
    char buf[32];
    strftime(buf, sizeof(buf), "%y%m%d-%H%M%S", tmNow);
    return std::string(buf);
}

static void openFile(const std::string& basePath) {
    std::string dir = basePath + "/pdr_data";
    if (!ensureDir(dir)) {
        LOGD("[SimDataLogger] Failed to create dir: %s", dir.c_str());
        return;
    }

    std::string path = dir + "/sim_data_" + makeTimestamp() + ".txt";
    s_file.open(path, std::ios::out | std::ios::trunc);
    if (s_file.is_open()) {
        s_file << "time(s)\tglobalZ\tgyroAngle\n";
        s_enabled = true;
        s_firstMs = 0;
        LOGD("[SimDataLogger] started: %s", path.c_str());
    } else {
        LOGD("[SimDataLogger] Failed to open: %s", path.c_str());
    }
}

// =========================================================
// Auto-start (기본 경로, 첫 write 시 1회 시도)
// =========================================================
static void autoStart() {
    if (s_initAttempted) return;
    s_initAttempted = true;

    std::lock_guard<std::mutex> lock(s_mutex);
    openFile(DEFAULT_APP_DIR);
}

// =========================================================
// Public API
// =========================================================
void SimDataLog_write(int64_t timestampMs, double globalZ, double gyroAngleDeg) {
    if (!s_initAttempted) autoStart();
    if (!s_enabled) return;

    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_file.is_open()) return;

    if (s_firstMs == 0) s_firstMs = timestampMs;
    double timeSec = (double)(timestampMs - s_firstMs) / 1000.0;
    s_file << timeSec << "\t" << globalZ << "\t" << gyroAngleDeg << "\n";
}

void SimDataLog_start(const char* basePath) {
    if (s_enabled) return;
    s_initAttempted = true;

    std::lock_guard<std::mutex> lock(s_mutex);
    openFile(basePath ? std::string(basePath) : std::string(DEFAULT_APP_DIR));
}

void SimDataLog_stop() {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_file.is_open()) {
        s_file.flush();
        s_file.close();
    }
    s_enabled = false;
    s_initAttempted = false;
    s_firstMs = 0;
    LOGD("[SimDataLogger] stopped");
}

bool SimDataLog_isEnabled() {
    return s_enabled;
}

// =========================================================
// JNI — Java/Kotlin에서 제어
// class: com.fifth.pdr_ext.SimDataLogger
// =========================================================
extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_SimDataLogger_nativeStart(JNIEnv* env, jobject, jstring basePath) {
    if (!basePath) return;
    const char* p = env->GetStringUTFChars(basePath, nullptr);
    if (p) {
        SimDataLog_start(p);
        env->ReleaseStringUTFChars(basePath, p);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_SimDataLogger_stop(JNIEnv*, jobject) {
    SimDataLog_stop();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_fifth_pdr_1ext_SimDataLogger_isEnabled(JNIEnv*, jobject) {
    return SimDataLog_isEnabled() ? JNI_TRUE : JNI_FALSE;
}
