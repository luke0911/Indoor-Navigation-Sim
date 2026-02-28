#include <android/log.h>
#include <cmath>
#include <cstdint>
#include <deque>
#include <optional>
#include <algorithm>
#include <fstream>
#include <string>
#include <mutex>
#include <chrono>
#include <ctime>
#include <sys/stat.h>
#include <jni.h>
#include "StepLength/StepLength.h"
#include "StepLength/SLRequire.h"

#define LOG_TAG "NativeStepLength"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// ----------------------------------------------
// File Logging for PDR Data (Auto-start)
// 파일은 /data/data/com.example.koreauniv/files/pdr_data/ 에 자동 저장됨
// ----------------------------------------------
static std::string g_pdrLogDir;           // Directory path for pdr_data
static std::ofstream g_stepLogFile;       // Step data log file
static std::ofstream g_sensorLogFile;     // Raw sensor data log file
static bool g_loggingEnabled = false;
static bool g_loggingInitAttempted = false;  // 자동 시작 시도 여부
static std::mutex g_logMutex;
static int64_t g_logSessionStartMs = 0;

// 앱의 files 디렉터리 경로 (하드코딩)
static const char* APP_FILES_DIR = "/data/data/testapplication/files";

static bool ensureDirectoryExists(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return mkdir(dir.c_str(), 0755) == 0;
}

static void autoStartLogging() {
    if (g_loggingInitAttempted) return;  // 이미 시도했으면 스킵
    g_loggingInitAttempted = true;

    std::lock_guard<std::mutex> lock(g_logMutex);

    g_pdrLogDir = std::string(APP_FILES_DIR) + "/pdr_data";
    if (!ensureDirectoryExists(g_pdrLogDir)) {
        LOGD("[StepLength] Failed to create pdr_data directory: %s", g_pdrLogDir.c_str());
        return;
    }

    // Get current timestamp for filename
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm* tm_now = localtime(&time_t_now);
    char timeStr[64];
    strftime(timeStr, sizeof(timeStr), "%Y%m%d_%H%M%S", tm_now);

    // Open step log file
    std::string stepLogPath = g_pdrLogDir + "/step_log_" + timeStr + ".txt";
    g_stepLogFile.open(stepLogPath, std::ios::out | std::ios::trunc);
    if (g_stepLogFile.is_open()) {
        g_stepLogFile << "# StepLength Log - " << timeStr << "\n";
        g_stepLogFile << "# timestamp_ms,frequency_hz,amplitude_z,peak_z,valley_z,duration_ms,raw_step_length,smoothed_step_length,k_freq,a_4th_root,profile\n";
        LOGD("[StepLength] Step log file opened: %s", stepLogPath.c_str());
    } else {
        LOGD("[StepLength] Failed to open step log file: %s", stepLogPath.c_str());
    }

    // Open sensor log file (for raw Z acceleration data)
    std::string sensorLogPath = g_pdrLogDir + "/sensor_log_" + timeStr + ".txt";
    g_sensorLogFile.open(sensorLogPath, std::ios::out | std::ios::trunc);
    if (g_sensorLogFile.is_open()) {
        g_sensorLogFile << "# Sensor Log - " << timeStr << "\n";
        g_sensorLogFile << "# timestamp_ms,lin_acc_z,current_peak,current_valley\n";
        LOGD("[StepLength] Sensor log file opened: %s", sensorLogPath.c_str());
    } else {
        LOGD("[StepLength] Failed to open sensor log file: %s", sensorLogPath.c_str());
    }

    g_logSessionStartMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    g_loggingEnabled = (g_stepLogFile.is_open() || g_sensorLogFile.is_open());
    if (g_loggingEnabled) {
        LOGD("[StepLength] Auto-logging started in: %s", g_pdrLogDir.c_str());
    }
}

// 수동 시작용 (호출 안해도 자동 시작됨)
static void startLogging(const std::string& basePath) {
    if (g_loggingEnabled) return;  // 이미 로깅 중이면 스킵

    std::lock_guard<std::mutex> lock(g_logMutex);
    g_loggingInitAttempted = true;

    g_pdrLogDir = basePath + "/pdr_data";
    if (!ensureDirectoryExists(g_pdrLogDir)) {
        LOGD("[StepLength] Failed to create pdr_data directory: %s", g_pdrLogDir.c_str());
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm* tm_now = localtime(&time_t_now);
    char timeStr[64];
    strftime(timeStr, sizeof(timeStr), "%Y%m%d_%H%M%S", tm_now);

    std::string stepLogPath = g_pdrLogDir + "/step_log_" + timeStr + ".txt";
    g_stepLogFile.open(stepLogPath, std::ios::out | std::ios::trunc);
    if (g_stepLogFile.is_open()) {
        g_stepLogFile << "# StepLength Log - " << timeStr << "\n";
        g_stepLogFile << "# timestamp_ms,frequency_hz,amplitude_z,peak_z,valley_z,duration_ms,raw_step_length,smoothed_step_length,k_freq,a_4th_root,profile\n";
    }

    std::string sensorLogPath = g_pdrLogDir + "/sensor_log_" + timeStr + ".txt";
    g_sensorLogFile.open(sensorLogPath, std::ios::out | std::ios::trunc);
    if (g_sensorLogFile.is_open()) {
        g_sensorLogFile << "# Sensor Log - " << timeStr << "\n";
        g_sensorLogFile << "# timestamp_ms,lin_acc_z,current_peak,current_valley\n";
    }

    g_loggingEnabled = (g_stepLogFile.is_open() || g_sensorLogFile.is_open());
}

static void stopLogging() {
    std::lock_guard<std::mutex> lock(g_logMutex);

    if (g_stepLogFile.is_open()) {
        g_stepLogFile.flush();
        g_stepLogFile.close();
    }
    if (g_sensorLogFile.is_open()) {
        g_sensorLogFile.flush();
        g_sensorLogFile.close();
    }
    g_loggingEnabled = false;
    g_loggingInitAttempted = false;  // 다시 시작할 수 있도록
}

// ----------------------------------------------
// StepLength: collects per-step Z lin-acc peaks/valleys
// and derives amplitude and frequency between steps.
// Integrate by calling:
//   onSensorSample(linAccZ, ts_ms)  // each sensor tick
//   onStepDetected(ts_ms)           // when PDRStateManager confirms a step
// Then read metrics with getters.
// ----------------------------------------------

//namespace {
//    static inline double clamp(double v, double lo, double hi) {
//        return v < lo ? lo : (v > hi ? hi : v);
//    }
//
//    static inline double safePow(double x, double p) {
//        return (x > 0.0) ? std::pow(x, p) : 0.0;
//    }
//
//    // Frequency → stride gain (sigmoid in sqrt(f))
//    // k(f) = 0.317 + 0.440 / (1 + exp(-4.78 * (sqrt(f) - 1.564)))
//    static inline double kFromFreq(double fHz) {
//        if (fHz <= 0.0) return 0.317; // minimal cadence
//        const double rootf = std::sqrt(fHz);
//        return 0.317 + (0.440 / (1.0 + std::exp(-4.78 * (rootf - 1.564))));
//    }
//
//    // In-pocket stride scale based on frequency
//    // S(f) = 0.0886 f^2 - 0.1336 f + 0.6813
//    static inline double pocketScaleFromFreq(double fHz) {
//        const double s = 0.0886 * fHz * fHz - 0.1336 * fHz + 0.6813;
//        // Clamp to avoid extreme scaling; adjust if your data suggests otherwise
//        return clamp(s, 0.50, 1.30);
//    }
//}
namespace {
    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    static inline double safePow(double x, double p) {
        return (x > 0.0) ? std::pow(x, p) : 0.0;
    }

    // --- 수정된 이중 시그모이드 공식 적용 ---
    static inline double kFromFreq(double fHz) {
        if (fHz <= 0.0) return 0.317;

        const double base = 0.317;

        // 1단계: 걷기(Walking) 구간 - 1.2Hz 중심, 0.457까지 상승
        const double w_center = 1.2;
        const double w_slope  = 6.0;
        const double w_gain   = 0.140;
        double walkingPart = w_gain / (1.0 + std::exp(-w_slope * (fHz - w_center)));

        // 2단계: 달리기(Running) 구간 - 2.8Hz 중심, 최종 0.700까지 상승
        const double r_center = 2.8;
        const double r_slope  = 3.0;
        const double r_gain   = 0.243;
        double runningPart = r_gain / (1.0 + std::exp(-r_slope * (fHz - r_center)));

        return base + walkingPart + runningPart;
    }

    static inline double pocketScaleFromFreq(double fHz) {
        const double s = 0.0886 * fHz * fHz - 0.1336 * fHz + 0.6813;
        return clamp(s, 0.50, 1.30);
    }
}

struct StepSegment {
    int64_t startMs = 0;           // inclusive
    int64_t endMs   = 0;           // inclusive
    double  peakZ   = -1e9;        // max Z lin-acc between steps
    double  valleyZ =  1e9;        // min Z lin-acc between steps
    double  amplitudeZ = 0.0;      // peakZ - valleyZ
    double  durationMs = 0.0;      // endMs - startMs
    double  frequencyHz = 0.0;     // 1000 / durationMs
    double  stepLength = 0.0;      // computed from frequency/amplitude (model below)
};

// 보폭 계산 중간값 저장 구조체
struct StepCalcDetails {
    double ampZ = 0.0;             // 진폭 (peak - valley)
    double fourthRoot = 0.0;       // ampZ^0.25
    double correctedRoot = 0.0;    // fourthRoot - alpha * max(0, fourthRoot - threshold)
    double frequencyHz = 0.0;      // 걸음 주파수
    double kf = 0.0;               // 주파수 계수 k(f)
    double scaleFactor = 1.0;      // 스케일 팩터
    double L0 = 0.0;               // 기본 보폭 (kf * correctedRoot * scaleFactor)
    double pocketScale = 1.0;      // 주머니 스케일 S
    double Lscaled = 0.0;          // 스케일 적용 보폭 (L0 * S)
    double minStride = 0.25;       // 최소 보폭
    double maxStride = 1.40;       // 최대 보폭
    double Lfinal = 0.0;           // 최종 보폭 (clamp 적용)
    double Lsmoothed = 0.0;        // 스무딩된 보폭
    int    profile = 0;            // 0: OnHand, 1: InPocket
};

class MovingAverage {
public:
    explicit MovingAverage(int n = 5) : N(n) {}
    double push(double x) {
        sum += x; q.push_back(x);
        if ((int)q.size() > N) { sum -= q.front(); q.pop_front(); }
        return sum / (double)q.size();
    }
    void reset() { q.clear(); sum = 0.0; }
private:
    int N; std::deque<double> q; double sum = 0.0;
};

class StepLength {
public:
    enum class Profile { OnHand, InPocket };

    StepLength() : maZ(5), maDuration(3) {}

    // Feed each linear-acceleration Z sample with timestamp in milliseconds
    void onSensorSample(double linAccZ, int64_t tsMs) {
        // light smoothing to stabilize peak/valley
        double z = maZ.push(linAccZ);
        if (!segmentOpen) {
            // first sample for a new segment starts when previous step closed
            current.startMs = tsMs;
            current.peakZ = z; current.valleyZ = z; segmentOpen = true;
        } else {
            if (z > current.peakZ)   current.peakZ   = z;
            if (z < current.valleyZ) current.valleyZ = z;
        }
        lastSampleMs = tsMs;
    }

    // Call this exactly when a step is confirmed by PDRStateManager
    void onStepDetected(int64_t tsMs, double explicitDurationMs = -1.0) {
        if (!segmentOpen) {
            // No samples yet; open and close a degenerate segment
            current.startMs = tsMs; current.peakZ = current.valleyZ = 0.0; segmentOpen = true;
        }

        // Close the segment at "now" and compute metrics
        current.endMs = (lastSampleMs > 0) ? lastSampleMs : tsMs;
        double rawDuration = 0.0;
        if (explicitDurationMs > 0.0) {
            rawDuration = explicitDurationMs;
        } else {
            rawDuration = (lastStepMs > 0) ? (double)(tsMs - lastStepMs) : 0.0;
        }
        // 2. [핵심 수정] 이동평균 필터 적용
        // Raw 값이 0이 아닐 때만 필터에 넣고, 그 '평균값'을 current.durationMs로 사용
        if (rawDuration > 0.0) {
            current.durationMs = maDuration.push(rawDuration);
        } else {
            current.durationMs = 0.0;
        }

        // 3. 주파수 계산 (이제 평균 Duration으로 나누므로 값이 안정됨)
        if (current.durationMs > 0.0) {
            current.frequencyHz = 1000.0 / current.durationMs;
        } else {
            current.frequencyHz = 0.0;
        }

        current.amplitudeZ = current.peakZ - current.valleyZ;

        // Override amplitude/peak/valley from SLRequire bridge
        {
            SLRequire& slr = SLRequire_Instance();
            float slrPeak = slr.peak();
            float slrValley = slr.valley();
            if (slrPeak != -1e9f || slrValley != 1e9f) {
                current.peakZ = slrPeak;
                current.valleyZ = slrValley;
                current.amplitudeZ = slrPeak - slrValley;
                LOGD("[StepLength] override by SLRequire peak/valley: peak=%.3f valley=%.3f ampZ=%.3f", current.peakZ, current.valleyZ, current.amplitudeZ);
                slr.resetSegment();
            } else {
                LOGD("[StepLength] no SLRequire override → keeping local amplitude=%.3f", current.amplitudeZ);
            }
        }

        // Compute step length using a tunable model (frequency + amplitude)
        current.stepLength = computeStepLength(current.amplitudeZ, current.frequencyHz);

        // ---- Stride smoothing with initial pass-through ----
        // Always push raw value into the ring buffer
        const double rawL = current.stepLength;
        const int W = (smWin < 1 ? 1 : (smWin > 16 ? 16 : smWin));
        double evict = 0.0;
        if (smSeen >= W) {
            evict = smBuf[smIdx];
        }
        smSum -= evict;
        smBuf[smIdx] = rawL;
        smSum += rawL;
        smIdx = (smIdx + 1) % W;
        smSeen++;

        double maVal;
        if (smSeen <= 2) {
            // first two steps: use raw value directly
            maVal = rawL;
        } else {
            const int denom = (smSeen < W ? smSeen : W);
            maVal = smSum / (double)denom; // denom is >=1 here
        }
        lastSmoothedStepLen = rawL ;
        lastCalcDetails.Lsmoothed = lastSmoothedStepLen;

        // Commit to last and reset for the next segment
        last = current; hasLast = true;
        lastStepMs = tsMs;
        resetCurrent(tsMs);

        const double dbgA4 = safePow(std::max(last.amplitudeZ, 0.0), 0.25);
        const double dbgK  = kFromFreq(last.frequencyHz);
        LOGD("[StepLength] step finalize: ampZ=%.3f (A^1/4=%.3f), peak=%.3f, valley=%.3f, dt=%.1f ms, f=%.2f Hz, k(f)=%.3f, Lraw=%.3f, L=%.3f",
             last.amplitudeZ, dbgA4, last.peakZ, last.valleyZ, last.durationMs, last.frequencyHz, dbgK, last.stepLength, lastSmoothedStepLen);
    }

    // -------- Accessors for the most recent finalized step --------
    double lastStepLength() const { return hasLast ? last.stepLength : defaultStepLen; }
    double lastSmoothedStepLength() const { return hasLast ? lastSmoothedStepLen : defaultStepLen; }

    // Accessors for logging
    double getLastAmplitudeZ() const { return hasLast ? last.amplitudeZ : 0.0; }
    double getLastFrequencyHz() const { return hasLast ? last.frequencyHz : 0.0; }
    double getLastPeakZ() const { return hasLast ? last.peakZ : 0.0; }
    double getLastValleyZ() const { return hasLast ? last.valleyZ : 0.0; }
    double getLastDurationMs() const { return hasLast ? last.durationMs : 0.0; }
    double getCurrentPeakZ() const { return segmentOpen ? current.peakZ : 0.0; }
    double getCurrentValleyZ() const { return segmentOpen ? current.valleyZ : 0.0; }
    Profile getProfile() const { return profile; }

    // 모든 중간 계산값 접근자
    const StepCalcDetails& getLastCalcDetails() const { return lastCalcDetails; }

    // Tunables for the step-length model
    void setBaseLength(double L) { defaultStepLen = L; }
    void setScale(double s) { scaleFactor = s; }
    void setClamp(double minL, double maxL) { minStride = minL; maxStride = maxL; }
    void setProfile(Profile p) { profile = p; }

private:
    // 마지막 계산의 모든 중간값 저장
    mutable StepCalcDetails lastCalcDetails;

    // Simple, robust model: L = clamp(K0 + Kamp * sqrt(max(ampZ,0)) + Kfreq * f, 0.25, 1.2)
    // You can later swap this with your sigmoid/frequency-based model.
    double computeStepLength(double ampZ, double fHz) {
        // Amplitude from Z-linAcc peak-to-valley; use 4th-root to compress range
        const double fourthRoot = safePow(std::max(ampZ, 0.0), 0.25);

        // Threshold-based correction for large amplitudes
        const double threshold = 1.441;
        const double alpha = 0.7;
        const double correctedRoot = fourthRoot - alpha * std::max(0.0, fourthRoot - threshold);

        const double kf = kFromFreq(fHz);
        const double L0 = kf * correctedRoot * scaleFactor; // base length before pocket scaling

        // pocket mode scale
        double S = 1.0;
        if (profile == Profile::InPocket) {
            S = pocketScaleFromFreq(fHz);
        }
        const double Lscaled = L0 * S;
        const double Lfinal = clamp(Lscaled, minStride, maxStride);

        // 모든 중간값 저장
        lastCalcDetails.ampZ = ampZ;
        lastCalcDetails.fourthRoot = fourthRoot;
        lastCalcDetails.correctedRoot = correctedRoot;
        lastCalcDetails.frequencyHz = fHz;
        lastCalcDetails.kf = kf;
        lastCalcDetails.scaleFactor = scaleFactor;
        lastCalcDetails.L0 = L0;
        lastCalcDetails.pocketScale = S;
        lastCalcDetails.Lscaled = Lscaled;
        lastCalcDetails.minStride = minStride;
        lastCalcDetails.maxStride = maxStride;
        lastCalcDetails.Lfinal = Lfinal;
        lastCalcDetails.profile = (profile == Profile::InPocket) ? 1 : 0;

        // Verbose debug: helps diagnose constant 0.25m results
        LOGD("[StepLength:model] ampZ=%.4f A1/4=%.4f corrected=%.4f f=%.3fHz kf=%.4f S=%.4f L0=%.4f Lscaled=%.4f clamp=[%.2f,%.2f] => L=%.4f (profile=%s)",
             ampZ, fourthRoot, correctedRoot, fHz, kf, S, L0, Lscaled, minStride, maxStride, Lfinal,
             (profile == Profile::InPocket ? "InPocket" : "OnHand"));

        return Lfinal;
    }

    void resetCurrent(int64_t tsStart) {
        current = StepSegment{}; segmentOpen = false; maZ.reset();
        // next onSensorSample will open a new segment at the first sample
        // but keep start hint for debugging
        hintNextStart = tsStart;
    }

    // Tunables
    double scaleFactor   = 1.00;  // overall gain on k(f)*A^1/4
    double minStride     = 0.25;  // m
    double maxStride     = 1.40;  // m
    double defaultStepLen = 0.65; // fallback when no finalized metrics yet

    // ---- Stride smoothing (runs only on step finalize) ----
    int    smWin   = 2;           // window size (>=1)
    double smSum   = 0.0;         // rolling sum
    double smBuf[16] = {0.0};     // ring buffer (supports up to 16 window)
    int    smIdx   = 0;           // write index in ring buffer
    int    smSeen  = 0;           // how many steps have been seen (monotonic)
    double lastSmoothedStepLen = 0.0; // published smoothed stride

    // state
    MovingAverage maZ;
    MovingAverage maDuration;
    StepSegment current{};
    StepSegment last{};
    bool segmentOpen = false;
    bool hasLast = false;
    int64_t lastSampleMs = 0;
    int64_t lastStepMs = 0;
    int64_t hintNextStart = 0;
    Profile profile = Profile::OnHand;
};

// ----------------------------------------------
// Global singleton (simple integration option)
// ----------------------------------------------
static StepLength gStepLength;

// Sensor sample counter for downsampling sensor logs
static int g_sensorSampleCounter = 0;

// Feed samples from SensorManager (linear acceleration Z, timestamp in ms)
void StepLength_onSensorSample(double linAccZ, int64_t tsMs) {
    gStepLength.onSensorSample(linAccZ, tsMs);
    // Debug at a low rate: only when |z| is large enough (reduces spam)
    if (std::fabs(linAccZ) > 0.5) {
        LOGD("[StepLength:sample] z=%.3f ts=%lld", linAccZ, (long long)tsMs);
    }

    // Log sensor data to file (downsample to every 5th sample to reduce file size)
    if (g_loggingEnabled && (g_sensorSampleCounter++ % 5 == 0)) {
        std::lock_guard<std::mutex> lock(g_logMutex);
        if (g_sensorLogFile.is_open()) {
            // smoothed_z is approximated; we log raw linAccZ
            g_sensorLogFile << tsMs << ","
                           << linAccZ << ","
                           << gStepLength.getCurrentPeakZ() << ","
                           << gStepLength.getCurrentValleyZ() << "\n";
        }
    }
}
// Notify when PDRStateManager confirms a step
void StepLength_onStepDetected(int64_t tsMs, double explicitDurationMs) {
    gStepLength.onStepDetected(tsMs, explicitDurationMs);

    // Log step data to file
    if (g_loggingEnabled) {
        std::lock_guard<std::mutex> lock(g_logMutex);
        if (g_stepLogFile.is_open()) {
            const double ampZ = gStepLength.getLastAmplitudeZ();
            const double fHz = gStepLength.getLastFrequencyHz();
            const double peakZ = gStepLength.getLastPeakZ();
            const double valleyZ = gStepLength.getLastValleyZ();
            const double durationMs = gStepLength.getLastDurationMs();
            const double rawStepLen = gStepLength.lastStepLength();
            const double smoothedStepLen = gStepLength.lastSmoothedStepLength();
            const double kf = kFromFreq(fHz);
            const double a4 = safePow(std::max(ampZ, 0.0), 0.25);
            const char* profileStr = (gStepLength.getProfile() == StepLength::Profile::InPocket) ? "InPocket" : "OnHand";

            g_stepLogFile << tsMs << ","
                         << fHz << ","
                         << ampZ << ","
                         << peakZ << ","
                         << valleyZ << ","
                         << durationMs << ","
                         << rawStepLen << ","
                         << smoothedStepLen << ","
                         << kf << ","
                         << a4 << ","
                         << profileStr << "\n";
            g_stepLogFile.flush();  // Ensure data is written
        }
    }
}


double StepLength_getLastStepLength() { return gStepLength.lastSmoothedStepLength(); }

double StepLength_getLastRawStepLength() { return gStepLength.lastStepLength(); }

void StepLength_setScale(double s) {
    gStepLength.setScale(s);
}
void StepLength_setClamp(double minL, double maxL) {
    gStepLength.setClamp(minL, maxL);
}
void StepLength_setPocketMode(bool enable) {
    gStepLength.setProfile(enable ? StepLength::Profile::InPocket
                                  : StepLength::Profile::OnHand);
}

// Logging control functions
void StepLength_startLogging(const char* basePath) {
    if (basePath) {
        startLogging(std::string(basePath));
    }
}

void StepLength_stopLogging() {
    stopLogging();
}

bool StepLength_isLoggingEnabled() {
    return g_loggingEnabled;
}

// Accessor implementations (declared in header)
double StepLength_getLastAmplitudeZ() { return gStepLength.getLastAmplitudeZ(); }
double StepLength_getLastFrequencyHz() { return gStepLength.getLastFrequencyHz(); }
double StepLength_getLastPeakZ() { return gStepLength.getLastPeakZ(); }
double StepLength_getLastValleyZ() { return gStepLength.getLastValleyZ(); }
double StepLength_getLastDurationMs() { return gStepLength.getLastDurationMs(); }

// 추가: 보폭 계산에 사용되는 값들
double StepLength_getLastA4thRoot() {
    double ampZ = gStepLength.getLastAmplitudeZ();
    const double fourthRoot = safePow(std::max(ampZ, 0.0), 0.25);
    // Threshold-based correction (same as computeStepLength)
    const double threshold = 1.441;
    const double alpha = 0.7;
    const double correctedRoot = fourthRoot - alpha * std::max(0.0, fourthRoot - threshold);
    return correctedRoot;
}

double StepLength_getLastFourthRoot() {
    double ampZ = gStepLength.getLastAmplitudeZ();
    return safePow(std::max(ampZ, 0.0), 0.25);
}

double StepLength_getLastKFreq() {
    double fHz = gStepLength.getLastFrequencyHz();
    return kFromFreq(fHz);
}

int StepLength_getLastProfile() {
    return (gStepLength.getProfile() == StepLength::Profile::InPocket) ? 1 : 0;
}

double StepLength_getLastRawStepLengthValue() {
    return gStepLength.lastStepLength();
}

double StepLength_getLastSmoothedStepLength() {
    return gStepLength.lastSmoothedStepLength();
}

// 모든 중간값 접근 함수들
double StepLength_getCalc_ampZ() { return gStepLength.getLastCalcDetails().ampZ; }
double StepLength_getCalc_fourthRoot() { return gStepLength.getLastCalcDetails().fourthRoot; }
double StepLength_getCalc_correctedRoot() { return gStepLength.getLastCalcDetails().correctedRoot; }
double StepLength_getCalc_frequencyHz() { return gStepLength.getLastCalcDetails().frequencyHz; }
double StepLength_getCalc_kf() { return gStepLength.getLastCalcDetails().kf; }
double StepLength_getCalc_scaleFactor() { return gStepLength.getLastCalcDetails().scaleFactor; }
double StepLength_getCalc_L0() { return gStepLength.getLastCalcDetails().L0; }
double StepLength_getCalc_pocketScale() { return gStepLength.getLastCalcDetails().pocketScale; }
double StepLength_getCalc_Lscaled() { return gStepLength.getLastCalcDetails().Lscaled; }
double StepLength_getCalc_minStride() { return gStepLength.getLastCalcDetails().minStride; }
double StepLength_getCalc_maxStride() { return gStepLength.getLastCalcDetails().maxStride; }
double StepLength_getCalc_Lfinal() { return gStepLength.getLastCalcDetails().Lfinal; }
double StepLength_getCalc_Lsmoothed() { return gStepLength.getLastCalcDetails().Lsmoothed; }
int StepLength_getCalc_profile() { return gStepLength.getLastCalcDetails().profile; }

// ----------------------------------------------
// JNI Functions for StepLength Logging
// ----------------------------------------------
extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_startLogging(JNIEnv* env, jobject /*thiz*/, jstring basePath) {
    if (!basePath) return;
    const char* pathChars = env->GetStringUTFChars(basePath, nullptr);
    if (pathChars) {
        StepLength_startLogging(pathChars);
        env->ReleaseStringUTFChars(basePath, pathChars);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_stopLogging(JNIEnv* /*env*/, jobject /*thiz*/) {
    StepLength_stopLogging();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_isLoggingEnabled(JNIEnv* /*env*/, jobject /*thiz*/) {
    return StepLength_isLoggingEnabled() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastAmplitudeZ(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastAmplitudeZ());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastFrequencyHz(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastFrequencyHz());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastPeakZ(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastPeakZ());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastValleyZ(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastValleyZ());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastDurationMs(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastDurationMs());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastStepLength(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastStepLength());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastA4thRoot(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastA4thRoot());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastKFreq(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastKFreq());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastProfile(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jint>(StepLength_getLastProfile());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastFourthRoot(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastFourthRoot());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastRawStepLength(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastRawStepLengthValue());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getLastSmoothedStepLength(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getLastSmoothedStepLength());
}

// 모든 중간 계산값 JNI 함수들
extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcAmpZ(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_ampZ());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcFourthRoot(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_fourthRoot());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcCorrectedRoot(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_correctedRoot());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcFrequencyHz(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_frequencyHz());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcKf(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_kf());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcScaleFactor(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_scaleFactor());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcL0(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_L0());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcPocketScale(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_pocketScale());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcLscaled(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_Lscaled());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcMinStride(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_minStride());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcMaxStride(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_maxStride());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcLfinal(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_Lfinal());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcLsmoothed(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(StepLength_getCalc_Lsmoothed());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_fifth_pdr_1ext_StepLengthLogger_getCalcProfile(JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jint>(StepLength_getCalc_profile());
}
