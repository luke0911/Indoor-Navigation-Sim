#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <array>
#include <queue>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <memory>
#include <opencv2/opencv.hpp>

#include "Sensor/SensorManager.h"
#include "StepDetection/onHand.h"
#include "StepLength/StepLength.h"
#include "StepLength/SLRequire.h"
#include "imu_floor_detection.h"
#include "viz.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

// ============================================================
//  Data Structures
// ============================================================
struct FloorSimSample {
    double timeSec;
    double globalX, globalY, globalZ;
    double magX, magY, magZ;
    double gyroAngleDeg;
    double pitch;
    std::string floorLabel;
    // 새 포맷 추가 필드
    double rotX, rotY, rotZ, rotW;
    int    dataStepCount;       // 데이터에 기록된 stepCount
};

// ============================================================
//  Data Loading — 헤더 기반 컬럼 매핑
//
//  필수 컬럼: time(s), globalX, globalY, globalZ, floorLabel/label
//  선택 컬럼: magX/Y/Z, gyroAngle, pitch, quatX~W/rot_x~w, stepCount
//  → 헤더만 있으면 어떤 조합이든 자동 인식
// ============================================================
static std::vector<FloorSimSample> loadFloorData(const char* path) {
    std::vector<FloorSimSample> out;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return out;
    }

    // 헤더 읽기 — 첫 비숫자 행
    std::string headerLine;
    bool hasHeader = false;
    while (std::getline(ifs, headerLine)) {
        if (headerLine.empty()) continue;
        if (!std::isdigit(headerLine[0]) && headerLine[0] != '-') {
            hasHeader = true;
            break;
        }
        break;  // 숫자로 시작 → 헤더 없음
    }

    if (!hasHeader) {
        fprintf(stderr, "ERROR: no header found in %s\n", path);
        return out;
    }

    // 헤더를 탭/스페이스로 분리하여 컬럼명 → 인덱스 매핑
    std::string hdr = headerLine;
    // '#' 접두사 제거 (일부 파일에서 "# time(s) ..." 형태)
    {
        size_t start = hdr.find_first_not_of(" \t");
        if (start != std::string::npos && hdr[start] == '#') {
            hdr = hdr.substr(start + 1);
        }
    }
    for (char& c : hdr) if (c == '\t') c = ' ';
    std::istringstream hss(hdr);
    std::vector<std::string> colNames;
    { std::string tok; while (hss >> tok) colNames.push_back(tok); }

    // 컬럼 인덱스 (-1 = 없음)
    int cTime = -1, cGX = -1, cGY = -1, cGZ = -1;
    int cMagX = -1, cMagY = -1, cMagZ = -1;
    int cGyro = -1, cPitch = -1;
    int cQX = -1, cQY = -1, cQZ = -1, cQW = -1;
    int cStepCnt = -1, cLabel = -1;

    for (int i = 0; i < (int)colNames.size(); ++i) {
        const auto& n = colNames[i];
        if (n == "time(s)" || n == "time")    cTime = i;
        else if (n == "globalX")              cGX = i;
        else if (n == "globalY")              cGY = i;
        else if (n == "globalZ")              cGZ = i;
        else if (n == "magX")                 cMagX = i;
        else if (n == "magY")                 cMagY = i;
        else if (n == "magZ")                 cMagZ = i;
        else if (n == "gyroAngle")            cGyro = i;
        else if (n == "pitch")                cPitch = i;
        else if (n == "rot_x" || n == "quatX" || n == "rotX") cQX = i;
        else if (n == "rot_y" || n == "quatY" || n == "rotY") cQY = i;
        else if (n == "rot_z" || n == "quatZ" || n == "rotZ") cQZ = i;
        else if (n == "rot_w" || n == "quatW" || n == "rotW") cQW = i;
        else if (n == "stepCount")            cStepCnt = i;
        else if (n == "floorLabel" || n == "label") cLabel = i;
    }

    // 필수 컬럼 확인
    if (cTime < 0 || cGX < 0 || cGY < 0 || cGZ < 0 || cLabel < 0) {
        fprintf(stderr, "ERROR: missing required columns in %s\n", path);
        fprintf(stderr, "  Need: time(s), globalX, globalY, globalZ, floorLabel/label\n");
        fprintf(stderr, "  Found: ");
        for (const auto& n : colNames) fprintf(stderr, "%s ", n.c_str());
        fprintf(stderr, "\n");
        return out;
    }

    bool hasQuat = (cQX >= 0 && cQY >= 0 && cQZ >= 0 && cQW >= 0);
    bool hasMag  = (cMagX >= 0 && cMagY >= 0 && cMagZ >= 0);

    printf("  Columns: %d  |  accel:OK  mag:%s  quat:%s  pitch:%s  gyro:%s  stepCnt:%s  label:OK\n",
           (int)colNames.size(),
           hasMag ? "OK" : "--",
           hasQuat ? "OK" : "--",
           cPitch >= 0 ? "OK" : (hasQuat ? "calc" : "--"),
           cGyro >= 0 ? "OK" : "--",
           cStepCnt >= 0 ? "OK" : "--");

    int totalCols = (int)colNames.size();

    // 데이터 행 파싱
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (!std::isdigit(line[0]) && line[0] != '-') continue;

        for (char& c : line) if (c == '\t') c = ' ';
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        { std::string tok; while (iss >> tok) tokens.push_back(tok); }
        if ((int)tokens.size() < totalCols) continue;

        FloorSimSample s{};
        s.timeSec    = std::stod(tokens[cTime]);
        s.globalX    = std::stod(tokens[cGX]);
        s.globalY    = std::stod(tokens[cGY]);
        s.globalZ    = std::stod(tokens[cGZ]);
        s.floorLabel = tokens[cLabel];

        if (cGyro >= 0)    s.gyroAngleDeg = std::stod(tokens[cGyro]);
        if (cPitch >= 0)   s.pitch = std::stod(tokens[cPitch]);
        if (hasMag) {
            s.magX = std::stod(tokens[cMagX]);
            s.magY = std::stod(tokens[cMagY]);
            s.magZ = std::stod(tokens[cMagZ]);
        }
        if (hasQuat) {
            s.rotX = std::stod(tokens[cQX]);
            s.rotY = std::stod(tokens[cQY]);
            s.rotZ = std::stod(tokens[cQZ]);
            s.rotW = std::stod(tokens[cQW]);
            // pitch가 없으면 쿼터니언에서 계산
            if (cPitch < 0) {
                double sinP = 2.0 * (s.rotW * s.rotX - s.rotZ * s.rotY);
                sinP = std::max(-1.0, std::min(1.0, sinP));
                s.pitch = std::asin(sinP) * 180.0 / M_PI;
            }
        }
        if (cStepCnt >= 0) s.dataStepCount = std::stoi(tokens[cStepCnt]);

        out.push_back(s);
    }
    return out;
}

// ============================================================
//  Helpers
// ============================================================
static std::string baseName(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

static std::string dirName(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? "." : path.substr(0, pos);
}

class SimpleMA {
public:
    explicit SimpleMA(int n) : period(n), sum(0.0f) {}
    float push(float v) {
        window.push(v);
        sum += v;
        if ((int)window.size() > period) { sum -= window.front(); window.pop(); }
        return sum / (float)window.size();
    }
private:
    int period;
    float sum;
    std::queue<float> window;
};

// ============================================================
//  Native File Dialog (macOS)
// ============================================================
static std::string nativeOpenFileDialog(const std::string& title,
                                         const std::string& defaultDir = "") {
#ifdef __APPLE__
    std::string cmd = "osascript -e 'POSIX path of (choose file with prompt \"";
    cmd += title;
    cmd += "\"";
    if (!defaultDir.empty() && fs::exists(defaultDir)) {
        cmd += " default location POSIX file \"" + defaultDir + "\"";
    }
    cmd += ")'";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    int rc = pclose(pipe);
    if (rc != 0) return "";

    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
#else
    (void)title; (void)defaultDir;
    return "";
#endif
}

// ============================================================
//  Data Root Finder
// ============================================================
static fs::path findDataRoot(const std::string& exePath) {
    fs::path current = fs::path(exePath).parent_path();
    for (int i = 0; i < 4; ++i) {
        if (fs::exists(current / "data")) {
            return current / "data";
        }
        if (current.has_parent_path()) {
            current = current.parent_path();
        } else {
            break;
        }
    }
    fs::path sourcePath = fs::path(__FILE__).parent_path() / "data";
    if (fs::exists(sourcePath)) return sourcePath;
    return "";
}

// ============================================================
//  Step Analysis 저장 — 스텝별 분석 결과 (결정적 파일명, 덮어쓰기)
// ============================================================
static void saveStepAnalysis(const std::vector<FloorTraceData>& trace,
                              const std::string& dataPath) {
    // 출력 디렉토리: data/output/ (data 폴더 탐색, 못 찾으면 입력 파일 옆 output/)
    fs::path outDir;
    {
        fs::path cur = fs::path(dataPath).parent_path();
        for (int i = 0; i < 4; ++i) {
            if (cur.filename() == "data" || fs::exists(cur / "Floor_sim_data")) {
                outDir = cur / "output";
                break;
            }
            if (cur.has_parent_path() && cur.parent_path() != cur)
                cur = cur.parent_path();
            else break;
        }
        if (outDir.empty()) outDir = fs::path(dataPath).parent_path() / "output";
    }
    fs::create_directories(outDir);

    // 결정적 파일명: step_analysis_<입력파일명>.tsv
    std::string fn = baseName(dataPath);
    auto dot = fn.find_last_of('.');
    if (dot != std::string::npos) fn = fn.substr(0, dot);
    std::string outPath = (outDir / ("step_analysis_" + fn + ".tsv")).string();

    // 스텝만 추출
    struct StepRow {
        int stepIdx; double time_s;
        double localFwdY, pureZ, zEnergyFrac;
        double procRmsZ, stepFreqHz, rmsZdivFreq, stepVarH;
        double dmag, recentStepVarYZ;
        double varYZRatio, dmagRatio, zfRatio, freqRatio;
        double dirR1;
        double energyFracZ, zPeakYRatio, stairConfidence;
        std::string fsmLabel, floorLabel;
        bool armSpinFlag;
    };
    std::vector<StepRow> rows;
    for (const auto& d : trace) {
        if (!d.isStep) continue;
        StepRow r;
        r.stepIdx = d.stepIdx;
        r.time_s = d.time_s;
        r.localFwdY = d.localForwardY;
        r.pureZ = d.pureZ;
        double aY = std::abs(d.localForwardY), aZ = std::abs(d.pureZ);
        r.zEnergyFrac = aZ / (aY + aZ + 1e-6);
        r.procRmsZ = d.procRmsZ;
        r.stepFreqHz = d.stepFreqHz;
        r.rmsZdivFreq = d.rmsZdivFreq;
        r.stepVarH = d.stepVarH;
        r.dmag = d.dmag;
        r.recentStepVarYZ = d.recentStepVarYZ;
        r.varYZRatio = d.varYZRatio;
        r.dmagRatio = d.dmagRatio;
        r.zfRatio = d.zfRatio;
        r.freqRatio = d.freqRatio;
        r.dirR1 = d.dirR1;
        r.energyFracZ = d.energyFracZ;
        r.zPeakYRatio = d.zPeakYRatio;
        r.stairConfidence = d.stairConfidence;
        r.fsmLabel = d.fsmLabel;
        r.floorLabel = d.floorLabel;
        r.armSpinFlag = d.armSpinFlag;
        rows.push_back(r);
    }

    // 5-step rolling average of zEnergyFrac
    std::vector<double> rollingAvg(rows.size(), 0.0);
    {
        double sum = 0.0;
        int win = 5;
        for (int i = 0; i < (int)rows.size(); ++i) {
            sum += rows[i].zEnergyFrac;
            if (i >= win) sum -= rows[i - win].zEnergyFrac;
            int cnt = std::min(i + 1, win);
            rollingAvg[i] = sum / cnt;
        }
    }

    std::ofstream ofs(outPath);
    if (!ofs.is_open()) {
        printf("ERROR: cannot write %s\n", outPath.c_str());
        return;
    }

    ofs << "stepIdx\ttime_s\tlocalFwdY\tpureZ\tzEnergyFrac\tzEF_avg5\t"
        << "procRmsZ\tstepFreqHz\trmsZdivFreq\tstepVarH\t"
        << "dmag\trecentStepVarYZ\t"
        << "varYZRatio\tdmagRatio\tzfRatio\tfreqRatio\t"
        << "dirR1\t"
        << "energyFracZ\tzPeakYRatio\tstairConfidence\t"
        << "fsmLabel\tarmSpin\tfloorGT\n";

    for (int i = 0; i < (int)rows.size(); ++i) {
        const auto& r = rows[i];
        ofs << r.stepIdx << "\t"
            << r.time_s << "\t"
            << r.localFwdY << "\t" << r.pureZ << "\t"
            << r.zEnergyFrac << "\t" << rollingAvg[i] << "\t"
            << r.procRmsZ << "\t" << r.stepFreqHz << "\t"
            << r.rmsZdivFreq << "\t" << r.stepVarH << "\t"
            << r.dmag << "\t" << r.recentStepVarYZ << "\t"
            << r.varYZRatio << "\t" << r.dmagRatio << "\t"
            << r.zfRatio << "\t" << r.freqRatio << "\t"
            << r.dirR1 << "\t"
            << r.energyFracZ << "\t" << r.zPeakYRatio << "\t"
            << r.stairConfidence << "\t"
            << r.fsmLabel << "\t" << (r.armSpinFlag ? 1 : 0) << "\t"
            << r.floorLabel << "\n";
    }
    ofs.close();
    printf("Step analysis: %d steps -> %s\n", (int)rows.size(), outPath.c_str());
}

// ============================================================
//  DataRecorder: TSV 출력
// ============================================================
class DataRecorder {
public:
    DataRecorder() : recording_(false) {}

    bool start(const std::string& outputDir, const std::string& baseFileName) {
        if (recording_) return false;

        // Ensure output directory exists
        fs::create_directories(outputDir);

        // Generate timestamped filename
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
        localtime_r(&t, &tm);
        char tsBuf[32];
        std::strftime(tsBuf, sizeof(tsBuf), "%Y%m%d_%H%M%S", &tm);

        std::string fname = "processed_" + baseFileName + "_" + tsBuf + ".txt";
        outputPath_ = outputDir + "/" + fname;

        ofs_.open(outputPath_);
        if (!ofs_.is_open()) return false;

        // Header
        ofs_ << "time_s\tglobalX\tglobalY\tglobalZ\tpureX\tpureY\tpureZ\t"
             << "localFwdY\tlocalLatX\t"
             << "varYZRatio\tdmagRatio\tzfRatio\tfreqRatio\tstairConfidence\t"
             << "rmsZdivFreq\tstepVarH\tstepFreqHz\tprocRmsZ\t"
             << "dmag\trecentStepVarYZ\t"
             << "energyFracZ\tzPeakYRatio\t"
             << "fsmLabel\tarmSpin\t"
             << "floorGT\tisStep\tstepIdx\n";

        recording_ = true;
        sampleCount_ = 0;
        printf("REC started: %s\n", outputPath_.c_str());
        return true;
    }

    void writeSample(const FloorTraceData& d) {
        if (!recording_) return;
        ofs_ << d.time_s << "\t"
             << d.globalX << "\t" << d.globalY << "\t" << d.globalZ << "\t"
             << d.pureX << "\t" << d.pureY << "\t" << d.pureZ << "\t"
             << d.localForwardY << "\t" << d.localLateralX << "\t"
             << d.varYZRatio << "\t" << d.dmagRatio << "\t"
             << d.zfRatio << "\t" << d.freqRatio << "\t"
             << d.stairConfidence << "\t"
             << d.rmsZdivFreq << "\t" << d.stepVarH << "\t"
             << d.stepFreqHz << "\t" << d.procRmsZ << "\t"
             << d.dmag << "\t" << d.recentStepVarYZ << "\t"
             << d.energyFracZ << "\t" << d.zPeakYRatio << "\t"
             << d.fsmLabel << "\t" << (d.armSpinFlag ? 1 : 0) << "\t"
             << d.floorLabel << "\t" << (d.isStep ? 1 : 0) << "\t"
             << d.stepIdx << "\n";
        sampleCount_++;
    }

    void stop() {
        if (!recording_) return;
        ofs_.close();
        recording_ = false;
        printf("REC stopped: %d samples -> %s\n", sampleCount_, outputPath_.c_str());
    }

    bool isRecording() const { return recording_; }
    const std::string& outputPath() const { return outputPath_; }

private:
    bool recording_;
    std::ofstream ofs_;
    std::string outputPath_;
    int sampleCount_ = 0;
};

// ============================================================
//  LiveSimulation: 샘플 단위 피딩
// ============================================================
class LiveSimulation {
public:
    void init(const std::string& dataPath) {
        allSamples_ = loadFloorData(dataPath.c_str());
        dataFileName_ = baseName(dataPath);
        feedIndex_ = 0;
        trace_.clear();
        trace_.reserve(allSamples_.size());
        totalSteps_ = 0;

        // Reset singletons
        getSensorManager()->reset();
        SLRequire_Instance().resetAll();

        // Reset algorithm instances
        detector_ = std::make_unique<OnHandStepDetection>();
        stepQueue_ = {0.65f, 0.65f, 0.65f, 0.65f};
        rotangle_ = {0.f, 0.f, 0.f};
        maFilter_ = SimpleMA(5);
        floorDetector_ = IMUFloorDetection();

        printf("Loaded: %s (%d samples)\n", dataFileName_.c_str(), (int)allSamples_.size());
    }

    bool feedNextSample() {
        if (feedIndex_ >= (int)allSamples_.size()) return false;

        const FloorSimSample& s = allSamples_[feedIndex_];
        int64_t tsMs = static_cast<int64_t>(s.timeSec * 1000.0);

        // Feed sensor data
        SensorManager* sm = getSensorManager();
        float linAcc[3] = {(float)s.globalX, (float)s.globalY, (float)s.globalZ};
        sm->updateLinearAccelerometer(linAcc, (long)tsMs);
        if (feedIndex_ == 0) {
            float rotV[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            sm->updateRotationVector(rotV, (long)tsMs);
        }
        StepLength_onSensorSample(s.globalZ, tsMs);
        rotangle_[2] = (float)s.gyroAngleDeg;

        // Step detection
        bool step = detector_->isStep(rotangle_, stepQueue_, tsMs, 3);
        float filtZ = maFilter_.push((float)s.globalZ);

        if (step) {
            totalSteps_++;
            SLRequire_Instance().setPeakValley((float)detector_->getLastPeakZ(),
                                                (float)detector_->getLastValleyZ());
            double preciseDuration = detector_->getLastStepPeakToPeak();
            StepLength_onStepDetected(tsMs, preciseDuration);
            double stepLen = StepLength_getLastStepLength();
            stepQueue_.push_back((float)stepLen);
            stepQueue_.pop_front();
        }

        // Floor detection
        FloorDetectionInput fInput{};
        fInput.timeSec = s.timeSec;
        fInput.globalX = s.globalX;
        fInput.globalY = s.globalY;
        fInput.globalZ = s.globalZ;
        fInput.magX = s.magX;
        fInput.magY = s.magY;
        fInput.magZ = s.magZ;
        fInput.gyroAngleDeg = s.gyroAngleDeg;
        fInput.pitch = s.pitch;
        fInput.isStep = step;
        fInput.stepCount = totalSteps_;
        fInput.stepPeakToPeakMs = step ? detector_->getLastStepPeakToPeak() : 0.0;

        FloorDetectionResult fResult = floorDetector_.update(fInput);

        // Build trace
        FloorTraceData td{};
        td.time_s = s.timeSec;
        td.globalX = s.globalX;
        td.globalY = s.globalY;
        td.globalZ = s.globalZ;
        td.magX = s.magX;
        td.magY = s.magY;
        td.magZ = s.magZ;
        td.gyroAngleDeg = s.gyroAngleDeg;
        td.pitch = s.pitch;
        td.filteredZ = filtZ;
        td.floorLabel = s.floorLabel;
        td.predictedLabel = fResult.predictedLabel;
        td.confidence = fResult.confidence;
        td.floorGT = floorLabelToValue(s.floorLabel);
        td.floorPred = floorLabelToValue(fResult.predictedLabel);

        td.pureX = fResult.pureX;
        td.pureY = fResult.pureY;
        td.pureZ = fResult.pureZ;
        td.localForwardY = fResult.localForwardY;
        td.localLateralX = fResult.localLateralX;

        // 핵심 피처
        td.procRmsZ = fResult.stepRmsZ;
        td.stepFreqHz = fResult.stepFreqHz;
        td.rmsZdivFreq = fResult.rmsZdivFreq;
        td.stepVarH = fResult.stepVarH;

        // 스텝 시작점 피처
        td.dmag = fResult.dmag;
        td.recentStepVarYZ = fResult.recentStepVarYZ;

        // Self-normalizing ratios
        td.varYZRatio = fResult.varYZRatio;
        td.dmagRatio = fResult.dmagRatio;
        td.zfRatio = fResult.zfRatio;
        td.freqRatio = fResult.freqRatio;
        td.stairConfidence = fResult.stairConfidence;
        td.totalStairScore = fResult.totalStairScore;

        // 궤적 방향 일관성
        td.dirR1 = fResult.dirR1;

        // 보조 피처
        td.energyFracZ = fResult.energyFracZ;
        td.zPeakYRatio = fResult.zPeakYRatio;

        // Envelope (참고용)
        td.envZ = fResult.envZ;
        td.envH = fResult.envH;
        td.stairScore = fResult.stairScore;

        // FSM
        td.fsmLabel = fResult.fsmLabel;
        td.armSpinFlag = fResult.armSpinFlag;
        td.procStepFreq = fResult.procStepFreq;

        // legacy
        td.rmsX = fResult.rmsX;
        td.rmsY = fResult.rmsY;
        td.rmsZ = fResult.rmsZ;

        td.isStep = step;
        td.stepIdx = step ? totalSteps_ : -1;
        td.totalSteps = totalSteps_;

        trace_.push_back(td);
        feedIndex_++;
        return true;
    }

    // Feed multiple samples at once (for batch/scrub)
    void feedUpTo(int targetIdx) {
        while (feedIndex_ <= targetIdx && feedIndex_ < (int)allSamples_.size()) {
            feedNextSample();
        }
    }

    // Feed all remaining samples
    void feedAll() {
        while (feedNextSample()) {}
    }

    const std::vector<FloorTraceData>& getTrace() const { return trace_; }
    int getCurrentIndex() const { return feedIndex_ - 1; }
    int getTotalSamples() const { return (int)allSamples_.size(); }
    bool isFinished() const { return feedIndex_ >= (int)allSamples_.size(); }
    const std::string& getDataFileName() const { return dataFileName_; }

    // 전체 데이터의 GT label 접근 (미니맵용)
    const std::string& getGTLabel(int idx) const { return allSamples_[idx].floorLabel; }
    const std::vector<FloorSimSample>& getAllSamples() const { return allSamples_; }

private:
    std::vector<FloorSimSample> allSamples_;
    int feedIndex_ = 0;
    std::vector<FloorTraceData> trace_;

    std::unique_ptr<OnHandStepDetection> detector_ = std::make_unique<OnHandStepDetection>();
    std::deque<float> stepQueue_ = {0.65f, 0.65f, 0.65f, 0.65f};
    std::array<float, 3> rotangle_ = {0.f, 0.f, 0.f};
    SimpleMA maFilter_{5};
    IMUFloorDetection floorDetector_;
    int totalSteps_ = 0;
    std::string dataFileName_;
};

// ============================================================
//  Main
// ============================================================
int main(int argc, char* argv[]) {
    fs::path dataRoot = findDataRoot(argv[0]);
    std::string floorDataDir = dataRoot.empty() ? "" : (dataRoot / "Floor_sim_data").string();

    // --- Check for --batch mode ---
    bool batchMode = false;
    std::string batchOutputPath;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--batch") {
            batchMode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                batchOutputPath = argv[i + 1];
            }
        }
    }

    // --- Select Floor Data File ---
    std::string dataPath;
    // Find first non-flag argument as data path
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--batch") { continue; }
        if (batchMode && argv[i] == batchOutputPath) { continue; }
        dataPath = argv[i];
        break;
    }
    if (dataPath.empty()) {
        dataPath = nativeOpenFileDialog("Select Floor Sim Data File", floorDataDir);
    }
    if (dataPath.empty()) { printf("No data selected.\n"); return 1; }

    // --- Initialize LiveSimulation ---
    LiveSimulation sim;
    sim.init(dataPath);
    if (sim.getTotalSamples() == 0) { printf("No valid data.\n"); return 1; }

    // --- Batch mode: process all data, save TSV, exit ---
    if (batchMode) {
        sim.feedAll();
        // Determine output path
        if (batchOutputPath.empty()) {
            fs::path outDir = fs::path(dataPath).parent_path().parent_path() / "output";
            fs::create_directories(outDir);
            std::string fn = baseName(dataPath);
            auto dot = fn.find_last_of('.');
            if (dot != std::string::npos) fn = fn.substr(0, dot);
            batchOutputPath = (outDir / ("batch_" + fn + ".tsv")).string();
        }
        std::ofstream ofs(batchOutputPath);
        if (!ofs.is_open()) { printf("Cannot write %s\n", batchOutputPath.c_str()); return 1; }
        // Header
        ofs << "time_s\tglobalX\tglobalY\tglobalZ\tpureX\tpureY\tpureZ\t"
            << "localFwdY\tlocalLatX\t"
            << "varYZRatio\tdmagRatio\tzfRatio\tfreqRatio\tstairConfidence\t"
            << "rmsZdivFreq\tstepVarH\tstepFreqHz\tprocRmsZ\t"
            << "dmag\trecentStepVarYZ\t"
            << "dirR1\t"
            << "energyFracZ\tzPeakYRatio\tenvZ\tenvH\tstairScore\t"
            << "fsmLabel\tarmSpin\t"
            << "floorGT\tisStep\tstepIdx\ttotalSteps\n";
        for (const auto& d : sim.getTrace()) {
            ofs << d.time_s << "\t"
                << d.globalX << "\t" << d.globalY << "\t" << d.globalZ << "\t"
                << d.pureX << "\t" << d.pureY << "\t" << d.pureZ << "\t"
                << d.localForwardY << "\t" << d.localLateralX << "\t"
                << d.varYZRatio << "\t" << d.dmagRatio << "\t"
                << d.zfRatio << "\t" << d.freqRatio << "\t"
                << d.stairConfidence << "\t"
                << d.rmsZdivFreq << "\t" << d.stepVarH << "\t"
                << d.stepFreqHz << "\t" << d.procRmsZ << "\t"
                << d.dmag << "\t" << d.recentStepVarYZ << "\t"
                << d.dirR1 << "\t"
                << d.energyFracZ << "\t" << d.zPeakYRatio << "\t"
                << d.envZ << "\t" << d.envH << "\t" << d.stairScore << "\t"
                << d.fsmLabel << "\t" << (d.armSpinFlag ? 1 : 0) << "\t"
                << d.floorLabel << "\t" << (d.isStep ? 1 : 0) << "\t"
                << d.stepIdx << "\t" << d.totalSteps << "\n";
        }
        ofs.close();
        printf("Batch done: %d samples, %d steps -> %s\n",
               (int)sim.getTrace().size(), sim.getTrace().back().totalSteps,
               batchOutputPath.c_str());
        // 스텝별 분석 결과 자동 저장
        saveStepAnalysis(sim.getTrace(), dataPath);
        return 0;
    }

    // Feed first sample so trace is not empty
    sim.feedNextSample();

    // Build GT label vector for minimap (전체 데이터 GT 분포)
    auto buildGTLabels = [](const LiveSimulation& s) {
        std::vector<std::string> labels;
        labels.reserve(s.getTotalSamples());
        for (int i = 0; i < s.getTotalSamples(); ++i) {
            labels.push_back(s.getGTLabel(i));
        }
        return labels;
    };
    std::vector<std::string> gtLabels = buildGTLabels(sim);

    // --- State ---
    int sampleIdx = 0;
    int stride = 25;
    bool liveMode = false;    // Space toggles live feeding
    double playbackSpeed = 1.0;  // 1x/2x/5x/0.5x
    DataRecorder recorder;

    // Output directory
    std::string outputDir = dirName(dataPath);
    auto pos = outputDir.find_last_of("/\\");
    std::string projectRoot = (pos != std::string::npos) ? outputDir.substr(0, pos) : ".";
    // Try to find project-level output dir
    fs::path outPath = fs::path(dataPath).parent_path().parent_path() / "output";
    if (!fs::exists(outPath)) {
        outPath = fs::path(dataPath).parent_path() / "output";
    }
    std::string outputDirStr = outPath.string();

    cv::namedWindow("Floor Detection", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Analysis", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Step Energy", cv::WINDOW_AUTOSIZE);

    // Initial display
    showMainWindow(sim.getTrace(), sampleIdx, sim.getDataFileName(), liveMode, recorder.isRecording(),
                   sim.getTotalSamples(), gtLabels);
    showAnalysisWindow(sim.getTrace(), sampleIdx);
    showStepEnergyWindow(sim.getTrace(), sampleIdx);

    printf("\n=== Floor Detection Simulation (Live Mode) ===\n");
    printf("  [Space]: Live/Pause toggle\n");
    printf("  [A]/[D]: Prev/Next Frame (scrub)\n");
    printf("  [W]/[S]: Stride Up/Down\n");
    printf("  [1/2/5/0]: Speed 1x/2x/5x/0.5x\n");
    printf("  [R]: Toggle recording\n");
    printf("  [F]: Open new data file\n");
    printf("  [Q]: Quit\n");

    auto lastFeedTime = std::chrono::steady_clock::now();
    double feedIntervalMs = 20.0;  // 50Hz base

    while (true) {
        int waitMs = 5;  // fast polling
        int key = cv::waitKey(waitMs);
        bool update = false;
        int charKey = key & 0xFF;

        if (charKey == 'q' || charKey == 27) break;

        // --- File Reload ---
        if (charKey == 'f') {
            std::string lastDir = dirName(dataPath);
            std::string newPath = nativeOpenFileDialog("Select Floor Sim Data File", lastDir);
            if (!newPath.empty()) {
                if (recorder.isRecording()) recorder.stop();
                sim.init(newPath);
                if (sim.getTotalSamples() > 0) {
                    sim.feedNextSample();
                    dataPath = newPath;
                    sampleIdx = 0;
                    liveMode = false;
                    update = true;
                    gtLabels = buildGTLabels(sim);
                    printf("Loaded: %s\n", sim.getDataFileName().c_str());
                } else {
                    printf("Failed to load: %s\n", baseName(newPath).c_str());
                }
            }
        }

        // --- Speed Control ---
        if (charKey == '1') { playbackSpeed = 1.0; printf("Speed: 1x\n"); }
        if (charKey == '2') { playbackSpeed = 2.0; printf("Speed: 2x\n"); }
        if (charKey == '5') { playbackSpeed = 5.0; printf("Speed: 5x\n"); }
        if (charKey == '0') { playbackSpeed = 0.5; printf("Speed: 0.5x\n"); }

        // --- Playback Control ---
        if (charKey == 'd') {
            int target = std::min(sampleIdx + stride, sim.getTotalSamples() - 1);
            sim.feedUpTo(target);
            sampleIdx = target;
            update = true;
        }
        if (charKey == 'a') {
            sampleIdx = std::max(sampleIdx - stride, 0);
            update = true;
        }
        if (charKey == 'w') { stride = std::min(stride * 2, 500); printf("Stride: %d\n", stride); }
        if (charKey == 's') { stride = std::max(stride / 2, 1); printf("Stride: %d\n", stride); }
        if (charKey == 32) {
            liveMode = !liveMode;
            printf(liveMode ? "LIVE mode\n" : "PAUSED\n");
            lastFeedTime = std::chrono::steady_clock::now();
            update = true;
        }

        // --- Recording ---
        if (charKey == 'r') {
            if (recorder.isRecording()) {
                recorder.stop();
            } else {
                // Strip extension from filename for output naming
                std::string fn = sim.getDataFileName();
                auto dotPos = fn.find_last_of('.');
                if (dotPos != std::string::npos) fn = fn.substr(0, dotPos);
                recorder.start(outputDirStr, fn);
            }
            update = true;
        }

        // --- Live Feeding ---
        if (liveMode && !sim.isFinished()) {
            auto now = std::chrono::steady_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(now - lastFeedTime).count();
            double adjustedInterval = feedIntervalMs / playbackSpeed;

            if (elapsedMs >= adjustedInterval) {
                // Feed samples proportional to elapsed time
                int samplesToFeed = std::max(1, (int)(elapsedMs / adjustedInterval));
                for (int i = 0; i < samplesToFeed && !sim.isFinished(); ++i) {
                    sim.feedNextSample();
                    sampleIdx = sim.getCurrentIndex();

                    // Record if active
                    if (recorder.isRecording() && !sim.getTrace().empty()) {
                        recorder.writeSample(sim.getTrace().back());
                    }
                }
                lastFeedTime = now;
                update = true;

                if (sim.isFinished()) {
                    liveMode = false;
                    printf("Data ended. Paused.\n");
                    // 재생 완료 시 스텝 분석 자동 저장
                    saveStepAnalysis(sim.getTrace(), dataPath);
                }
            }
        }

        if (update) {
            // Clamp sampleIdx
            if (!sim.getTrace().empty()) {
                sampleIdx = std::max(0, std::min(sampleIdx, (int)sim.getTrace().size() - 1));
            }
            showMainWindow(sim.getTrace(), sampleIdx, sim.getDataFileName(),
                           liveMode, recorder.isRecording(), sim.getTotalSamples(), gtLabels);
            showAnalysisWindow(sim.getTrace(), sampleIdx);
            showStepEnergyWindow(sim.getTrace(), sampleIdx);
        }
    }

    if (recorder.isRecording()) recorder.stop();

    // GUI 종료 시 스텝 분석 자동 저장 (스텝이 1개 이상 있을 때)
    if (!sim.getTrace().empty() && sim.getTrace().back().totalSteps > 0) {
        saveStepAnalysis(sim.getTrace(), dataPath);
    }

    cv::destroyAllWindows();
    return 0;
}
