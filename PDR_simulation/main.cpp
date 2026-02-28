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
#include <opencv2/opencv.hpp>

#include "Sensor/SensorManager.h"
#include "StepDetection/onHand.h"
#include "StepLength/StepLength.h"
#include "StepLength/SLRequire.h"
#include "OldAlgorithmWrapper.h"
#include "viz.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations for StepLength calc detail accessors (defined in StepLength.cpp)
double StepLength_getCalc_fourthRoot();
double StepLength_getCalc_correctedRoot();
double StepLength_getCalc_kf();
double StepLength_getCalc_L0();
double StepLength_getCalc_Lfinal();

// Forward declarations for OnHand dynamic threshold accessors (defined in onHand.cpp)
double OnHand_getDynamicZDiff();
double OnHand_getDynamicUpThresh();
double OnHand_getDynamicDownThresh();

// Helper: compute step length intermediates from ampZ/freqHz (for Old pipeline reverse-calc)
static void computeOldStepIntermediates(StepResult& sr, double scaleFactor = 1.0) {
    double ampZ = sr.ampZ;
    double fHz = sr.freqHz;
    sr.fourthRoot = (ampZ > 0.0) ? std::pow(ampZ, 0.25) : 0.0;
    const double threshold = 1.441;
    const double alpha = 0.7;
    sr.correctedRoot = sr.fourthRoot - alpha * std::max(0.0, sr.fourthRoot - threshold);
    if (fHz <= 0.0) {
        sr.kf = 0.317;
        sr.kf_walk = 0.0;
        sr.kf_run = 0.0;
    } else {
        double base = 0.317;
        sr.kf_walk = 0.140 / (1.0 + std::exp(-6.0 * (fHz - 1.2)));
        sr.kf_run = 0.243 / (1.0 + std::exp(-3.0 * (fHz - 2.8)));
        sr.kf = base + sr.kf_walk + sr.kf_run;
    }
    sr.L0 = sr.kf * sr.correctedRoot * scaleFactor;
    sr.Lfinal = sr.rawStepLen;
}

namespace fs = std::filesystem;

// ============================================================
//  Data Structures
// ============================================================
struct SimSample {
    double timeSec;
    double globalZ;
    double gyroAngleDeg;
};

struct SimResult {
    std::vector<StepResult> steps;
    std::vector<StepResult> oldSteps;
    std::vector<SampleData> trace;
    std::string dataFileName;
    bool isFemale = false;
};

// ============================================================
//  Helper Functions
// ============================================================
static std::vector<SimSample> loadData(const char* path) {
    std::vector<SimSample> out;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return out;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        SimSample s{};
        if (iss >> s.timeSec >> s.globalZ >> s.gyroAngleDeg) {
            out.push_back(s);
        }
    }
    return out;
}

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
    if (rc != 0) return "";  // User cancelled

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
//  Map Loading
// ============================================================
static cv::Mat loadZeroMapText(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return cv::Mat();
    std::vector<std::vector<int>> grid;
    std::string line;
    int maxCols = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<int> row;
        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream iss(line);
        int val;
        while (iss >> val) row.push_back(val);
        if (!row.empty()) {
            grid.push_back(row);
            if ((int)row.size() > maxCols) maxCols = row.size();
        }
    }
    if (grid.empty()) return cv::Mat();
    int rows = grid.size();
    int cols = maxCols;
    cv::Mat mapImg(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < (int)grid[r].size(); ++c) {
            if (grid[r][c] == 1) {
                mapImg.at<cv::Vec3b>(r, c) = cv::Vec3b(100, 100, 100);
            }
        }
    }
    return mapImg;
}

static cv::Mat loadMapFile(const std::string& path) {
    if (path.empty()) return cv::Mat();
    std::string ext = baseName(path);
    size_t lastDot = ext.find_last_of(".");
    if (lastDot != std::string::npos) {
        std::string extension = ext.substr(lastDot);
        if (extension == ".txt") {
            printf("Parsing ZeroMap text file: %s\n", baseName(path).c_str());
            return loadZeroMapText(path);
        }
    }
    cv::Mat img = cv::imread(path);
    if (!img.empty()) printf("Loaded Map Image: %s\n", baseName(path).c_str());
    return img;
}

static cv::Mat makeBlankMap() {
    cv::Mat m(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    for(int i=0; i<800; i+=50) cv::line(m, {i,0}, {i,800}, cv::Scalar(220,220,220), 1);
    for(int i=0; i<800; i+=50) cv::line(m, {0,i}, {800,i}, cv::Scalar(220,220,220), 1);
    return m;
}

// ============================================================
//  Simulation Engine (runs both New & Old pipelines)
// ============================================================
// femaleOverride: -1=auto-detect from filename, 0=force male, 1=force female
static SimResult runSimulation(const std::string& dataPath, int femaleOverride = -1) {
    SimResult result;
    auto samples = loadData(dataPath.c_str());
    if (samples.empty()) return result;

    result.dataFileName = baseName(dataPath);
    printf("Processing: %s (%d samples)...\n", result.dataFileName.c_str(), (int)samples.size());

    // Reset singletons for clean re-run
    getSensorManager()->reset();
    SLRequire_Instance().resetAll();
    Old_resetGlobals();

    // Detect female mode from filename (_여) or use override
    if (femaleOverride >= 0) {
        result.isFemale = (femaleOverride == 1);
    } else {
        result.isFemale = (dataPath.find("_\xEC\x97\xAC") != std::string::npos); // UTF-8 for 여
    }
    const double femaleScale = result.isFemale ? 0.90 : 1.00;
    StepLength_setScale(femaleScale);
    Old_setScale(femaleScale);
    if (result.isFemale) printf("  >> Female mode: scaleFactor = %.2f\n", femaleScale);

    // --- New Pipeline ---
    OnHandStepDetection detector;
    std::deque<float> stepQueue = {0.65f, 0.65f, 0.65f, 0.65f};
    std::array<float, 3> rotangle = {0.f, 0.f, 0.f};
    SensorManager* sm = getSensorManager();
    SimpleMA maFilter(5);

    double x = 0.0, y = 0.0;
    double initialHeading = -999.0;

    // --- Old Pipeline ---
    OldPDRContext* oldCtx = Old_createContext();
    double oldX = 0.0, oldY = 0.0;
    double oldInitialHeading = -999.0;

    result.trace.reserve(samples.size());

    for (size_t i = 0; i < samples.size(); ++i) {
        const SimSample& s = samples[i];
        int64_t tsMs = static_cast<int64_t>(s.timeSec * 1000.0);
        float linAcc[3] = {0.0f, 0.0f, static_cast<float>(s.globalZ)};
        sm->updateLinearAccelerometer(linAcc, (long)tsMs);
        if (i == 0) {
            float rotV[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            sm->updateRotationVector(rotV, (long)tsMs);
        }
        StepLength_onSensorSample(s.globalZ, tsMs);
        rotangle[2] = static_cast<float>(s.gyroAngleDeg);
        bool step = detector.isStep(rotangle, stepQueue, tsMs, 3);
        float filtZ = maFilter.push((float)s.globalZ);

        SampleData sd{};
        sd.time_s = s.timeSec;
        sd.globalZ = s.globalZ;
        sd.filteredZ = filtZ;
        sd.isStep = step;
        sd.stepIdx = -1;
        sd.isOldStep = false;
        sd.oldStepIdx = -1;
        sd.dynamicZDiff      = OnHand_getDynamicZDiff();
        sd.dynamicUpThresh   = OnHand_getDynamicUpThresh();
        sd.dynamicDownThresh = OnHand_getDynamicDownThresh();

        if (step) {
             SLRequire_Instance().setPeakValley((float)detector.getLastPeakZ(), (float)detector.getLastValleyZ());
             double preciseDuration = detector.getLastStepPeakToPeak();
             StepLength_onStepDetected(tsMs, preciseDuration);
             double stepLen = StepLength_getLastStepLength();
             double rawStepLen = StepLength_getLastRawStepLength();

             if (initialHeading == -999.0) initialHeading = s.gyroAngleDeg;
             double alignedGyro = s.gyroAngleDeg - initialHeading + 90.0;
             double dirRad = alignedGyro * M_PI / 180.0;
             x += stepLen * std::sin(dirRad);
             y += stepLen * std::cos(dirRad);
             stepQueue.push_back((float)stepLen);
             stepQueue.pop_front();

             StepResult sr{};
             sr.stepNum = (int)result.steps.size() + 1;
             sr.time_s = s.timeSec;
             sr.stepLen = stepLen;
             sr.rawStepLen = rawStepLen;
             sr.directionDeg = s.gyroAngleDeg;
             sr.ampZ = StepLength_getLastAmplitudeZ();
             sr.freqHz = StepLength_getLastFrequencyHz();
             sr.peak = StepLength_getLastPeakZ();
             sr.valley = StepLength_getLastValleyZ();
             sr.x = x;
             sr.y = y;
             sr.sampleIdx = (int)result.trace.size();
             // Fill pipeline intermediate values from StepLength calc details
             sr.fourthRoot = StepLength_getCalc_fourthRoot();
             sr.correctedRoot = StepLength_getCalc_correctedRoot();
             sr.durationMs = StepLength_getLastDurationMs();
             sr.kf = StepLength_getCalc_kf();
             // Compute sigmoid components from freqHz
             if (sr.freqHz > 0.0) {
                 sr.kf_walk = 0.140 / (1.0 + std::exp(-6.0 * (sr.freqHz - 1.2)));
                 sr.kf_run = 0.243 / (1.0 + std::exp(-3.0 * (sr.freqHz - 2.8)));
             }
             sr.L0 = StepLength_getCalc_L0();
             sr.Lfinal = StepLength_getCalc_Lfinal();
             sd.stepIdx = (int)result.steps.size();
             result.steps.push_back(sr);
        }

        // --- Old PDR Pipeline (same sensor data) ---
        Old_feedSensorSample(oldCtx, static_cast<float>(s.globalZ), tsMs);
        if (i == 0) {
            float rotV[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            Old_feedRotationVector(oldCtx, rotV, (long)tsMs);
        }
        bool oldStep = Old_runStepDetection(oldCtx, static_cast<float>(s.gyroAngleDeg), tsMs);
        sd.isOldStep = oldStep;
        if (oldStep) {
            Old_onStepDetected(oldCtx, tsMs);
            double oldStepLen = Old_getLastStepLength(oldCtx);
            double oldRawStepLen = Old_getLastRawStepLength(oldCtx);

            if (oldInitialHeading == -999.0) oldInitialHeading = s.gyroAngleDeg;
            double oldAlignedGyro = s.gyroAngleDeg - oldInitialHeading + 90.0;
            double oldDirRad = oldAlignedGyro * M_PI / 180.0;
            oldX += oldStepLen * std::sin(oldDirRad);
            oldY += oldStepLen * std::cos(oldDirRad);

            StepResult osr{};
            osr.stepNum = (int)result.oldSteps.size() + 1;
            osr.time_s = s.timeSec;
            osr.stepLen = oldStepLen;
            osr.rawStepLen = oldRawStepLen;
            osr.directionDeg = s.gyroAngleDeg;
            osr.ampZ = Old_getLastAmplitudeZ(oldCtx);
            osr.freqHz = Old_getLastFrequencyHz(oldCtx);
            osr.peak = Old_getLastPeakZ(oldCtx);
            osr.valley = Old_getLastValleyZ(oldCtx);
            osr.x = oldX;
            osr.y = oldY;
            osr.sampleIdx = (int)result.trace.size();
            // durationMs from consecutive step times
            if (!result.oldSteps.empty()) {
                osr.durationMs = (s.timeSec - result.oldSteps.back().time_s) * 1000.0;
            }
            // Compute pipeline intermediate values for old step
            computeOldStepIntermediates(osr, femaleScale);
            sd.oldStepIdx = (int)result.oldSteps.size();
            result.oldSteps.push_back(osr);
        }

        result.trace.push_back(sd);
    }

    Old_destroyContext(oldCtx);
    printf("Done: %d new steps, %d old steps detected.\n",
           (int)result.steps.size(), (int)result.oldSteps.size());
    return result;
}

// ============================================================
//  Start Point Selection (mouse callback)
// ============================================================
struct MapSelection {
    cv::Point pt;
    bool clicked = false;
};

static void onMouseMap(int event, int x, int y, int, void* userdata) {
    MapSelection* sel = (MapSelection*)userdata;
    if (event == cv::EVENT_LBUTTONDOWN) {
        sel->pt = cv::Point(x, y);
        sel->clicked = true;
    }
}

// Mouse callback for World Map View (toggle buttons + start point click)
struct MapViewClick {
    MapViewState* state;
    cv::Point* startPt;
    bool changed;
};

static void onMouseWorldMap(int event, int x, int y, int, void* userdata) {
    MapViewClick* mvc = (MapViewClick*)userdata;
    if (event != cv::EVENT_LBUTTONDOWN) return;

    // Button geometry must match drawWorldMapWindow
    const int btnW = 80, btnH = 28, btnY = 8, btnGap = 6;
    int btnX0 = 8;
    int btnX1 = btnX0 + btnW + btnGap;

    // New button
    if (x >= btnX0 && x <= btnX0 + btnW && y >= btnY && y <= btnY + btnH) {
        mvc->state->showNewPath = !mvc->state->showNewPath;
        mvc->changed = true;
    }
    // Old button
    else if (x >= btnX1 && x <= btnX1 + btnW && y >= btnY && y <= btnY + btnH) {
        mvc->state->showOldPath = !mvc->state->showOldPath;
        mvc->changed = true;
    }
    // Anywhere else → move start point (subtract canvas offset to get map-image coords)
    else {
        cv::Point off = getWorldMapCanvasOffset();
        mvc->startPt->x = x - off.x;
        mvc->startPt->y = y - off.y;
        mvc->state->moveX = 0;
        mvc->state->moveY = 0;
        mvc->changed = true;
    }
}

static cv::Point selectStartPoint(const cv::Mat& mapImage) {
    MapSelection mapSel;
    mapSel.pt = {mapImage.cols/2, mapImage.rows/2};

    cv::namedWindow("Set Start Point");
    cv::setMouseCallback("Set Start Point", onMouseMap, &mapSel);

    printf("\n=== Map Setup ===\n");
    printf("1. Click start point on map.\n2. Press SPACE to start.\n");

    while(true) {
        cv::Mat temp = mapImage.clone();
        cv::circle(temp, mapSel.pt, 6, cv::Scalar(0,0,255), cv::FILLED);
        cv::circle(temp, mapSel.pt, 10, cv::Scalar(0,0,255), 2);
        cv::putText(temp, "Click Start Point", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
        cv::imshow("Set Start Point", temp);
        if (cv::waitKey(20) == 32) break;
    }
    cv::destroyWindow("Set Start Point");
    return mapSel.pt;
}

// ============================================================
//  Dashboard Mouse Callback (waveform click inspect)
// ============================================================
struct DashboardClickState {
    WaveformClick wfClick;
    const std::vector<SampleData>* trace = nullptr;
    int currentSampleIdx = 0;
    int pdrPathWidth = 0;
    bool changed = false;
};

static void onMouseDashboard(int event, int x, int y, int, void* userdata) {
    DashboardClickState* st = (DashboardClickState*)userdata;
    if (event != cv::EVENT_LBUTTONDOWN || !st->trace || st->trace->empty()) return;

    const int WAV_W = 800, WAV_H = 220;
    const int ML = 40, BOT_MARGIN = 15;
    const int SCROLL_WIN = 400;
    int leftW = st->pdrPathWidth;

    // Check if click is in waveform area (either new or old)
    int relX = x - leftW;
    if (relX < ML || relX >= WAV_W - 5) { st->wfClick.active = false; st->changed = true; return; }
    if (y < 0 || y >= WAV_H * 2) { st->wfClick.active = false; st->changed = true; return; }

    // Reverse map pixel to sample index
    int sampleIdx = st->currentSampleIdx;
    int start = std::max(0, sampleIdx - SCROLL_WIN + 1);
    int count = sampleIdx - start + 1;
    int xOff = SCROLL_WIN - count;
    double ppx = (double)(WAV_W - ML - 5) / SCROLL_WIN;

    double localI = (relX - ML) / ppx - xOff;
    int clickedSample = start + (int)std::round(localI);
    clickedSample = std::max(start, std::min(sampleIdx, clickedSample));

    if (clickedSample < 0 || clickedSample >= (int)st->trace->size()) {
        st->wfClick.active = false;
        st->changed = true;
        return;
    }

    st->wfClick.active = true;
    st->wfClick.sampleIdx = clickedSample;
    st->wfClick.value = (*st->trace)[clickedSample].globalZ;
    st->wfClick.filteredValue = (*st->trace)[clickedSample].filteredZ;
    st->wfClick.time_s = (*st->trace)[clickedSample].time_s;
    st->changed = true;
}

// ============================================================
//  Step Analysis Mouse Callback (scale box toggle)
// ============================================================
struct StepAnalysisClick {
    bool scaleToggled = false;  // set true when user clicks the scale box
};

static void onMouseStepAnalysis(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    StepAnalysisClick* sa = (StepAnalysisClick*)userdata;

    // Scale box hit area (matches showStepAnalysisWindow layout)
    // OUT_START_X = 15 + 3*(110+22) + 30 = 441, BOX_W = 110
    const int SBX0 = 441, SBX1 = 441 + 110;
    // New scale box Y: ~57-95, Old scale box Y: ~157-195
    // Use generous range covering both
    if (x >= SBX0 && x <= SBX1 && ((y >= 40 && y <= 100) || (y >= 140 && y <= 200))) {
        sa->scaleToggled = true;
    }
}

// ============================================================
//  Main
// ============================================================
// Compute Freq CV (coefficient of variation) from a vector of StepResult
static double computeFreqCV(const std::vector<StepResult>& steps) {
    if (steps.size() < 3) return 0.0;
    // Skip first 2 steps (warmup)
    double sum = 0.0, sum2 = 0.0;
    int n = 0;
    for (size_t i = 2; i < steps.size(); ++i) {
        double f = steps[i].freqHz;
        if (f > 0.0) { sum += f; sum2 += f * f; ++n; }
    }
    if (n < 2) return 0.0;
    double mean = sum / n;
    double var = sum2 / n - mean * mean;
    if (var < 0.0) var = 0.0;
    return (mean > 0.0) ? std::sqrt(var) / mean : 0.0;
}

// Compute mean step length
static double computeMeanStepLen(const std::vector<StepResult>& steps) {
    if (steps.empty()) return 0.0;
    double sum = 0.0;
    for (auto& s : steps) sum += s.stepLen;
    return sum / steps.size();
}

// Compute mean frequency
static double computeMeanFreq(const std::vector<StepResult>& steps) {
    if (steps.size() < 3) return 0.0;
    double sum = 0.0; int n = 0;
    for (size_t i = 2; i < steps.size(); ++i) {
        if (steps[i].freqHz > 0.0) { sum += steps[i].freqHz; ++n; }
    }
    return n > 0 ? sum / n : 0.0;
}

// Compute mean amplitude
static double computeMeanAmp(const std::vector<StepResult>& steps) {
    if (steps.empty()) return 0.0;
    double sum = 0.0;
    for (auto& s : steps) sum += s.ampZ;
    return sum / steps.size();
}

// Compute mean duration (ms) from freqHz, skip first 2
static double computeMeanDuration(const std::vector<StepResult>& steps) {
    if (steps.size() < 3) return 0.0;
    double sum = 0.0; int n = 0;
    for (size_t i = 2; i < steps.size(); ++i) {
        if (steps[i].durationMs > 0.0) { sum += steps[i].durationMs; ++n; }
    }
    return n > 0 ? sum / n : 0.0;
}

// Compute duration CV
static double computeDurationCV(const std::vector<StepResult>& steps) {
    if (steps.size() < 3) return 0.0;
    double sum = 0.0, sum2 = 0.0; int n = 0;
    for (size_t i = 2; i < steps.size(); ++i) {
        double d = steps[i].durationMs;
        if (d > 0.0) { sum += d; sum2 += d * d; ++n; }
    }
    if (n < 2) return 0.0;
    double mean = sum / n;
    double var = sum2 / n - mean * mean;
    if (var < 0.0) var = 0.0;
    return (mean > 0.0) ? std::sqrt(var) / mean : 0.0;
}

// Extract expected step count from filename (e.g., "120걸음" → 120)
static int extractExpectedSteps(const std::string& fname) {
    // Look for a number followed by 걸음 (UTF-8: \xEA\xB1\xB8\xEC\x9D\x8C)
    std::string pattern = "\xEA\xB1\xB8\xEC\x9D\x8C";
    auto pos = fname.find(pattern);
    if (pos == std::string::npos) return -1;
    // Walk backwards from pos to find digits
    int end = (int)pos;
    int start = end - 1;
    while (start >= 0 && fname[start] >= '0' && fname[start] <= '9') --start;
    ++start;
    if (start >= end) return -1;
    return std::stoi(fname.substr(start, end - start));
}

int main(int argc, char* argv[]) {
    fs::path dataRoot = findDataRoot(argv[0]);
    std::string pdrDataDir = dataRoot.empty() ? "" : (dataRoot / "PDR_Data").string();
    std::string mapDataDir = dataRoot.empty() ? "" : (dataRoot / "mapdata").string();

    // ========== BATCH MODE ==========
    bool batchMode = (argc >= 2 && std::string(argv[1]) == "--batch");
    bool batchDetail = (argc >= 2 && std::string(argv[1]) == "--batch-detail");
    if (batchMode || batchDetail) {
        // Collect all .txt data files under PDR_Data
        std::vector<std::string> dataFiles;
        int argStart = batchDetail ? 2 : (batchMode ? 2 : 2);
        if (argc >= 3) {
            for (int i = argStart; i < argc; ++i) dataFiles.push_back(argv[i]);
        } else {
            for (auto& entry : fs::recursive_directory_iterator(pdrDataDir)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::string fname = entry.path().filename().string();
                    if (ext == ".txt" && fname.find("sim_data") == std::string::npos
                        && fname.find("제자리") == std::string::npos) {
                        dataFiles.push_back(entry.path().string());
                    }
                }
            }
            std::sort(dataFiles.begin(), dataFiles.end());
        }

        if (batchDetail) {
            // ===== DETAILED OUTPUT FOR REPORT =====
            printf("\n========== Step Detection & Frequency Analysis (Detailed) ==========\n\n");
            for (auto& path : dataFiles) {
                SimResult sim = runSimulation(path);
                std::string fname = baseName(path);
                int expected = extractExpectedSteps(fname);
                int nNew = (int)sim.steps.size();
                int nOld = (int)sim.oldSteps.size();

                double newCV = computeFreqCV(sim.steps);
                double oldCV = computeFreqCV(sim.oldSteps);
                double newMeanF = computeMeanFreq(sim.steps);
                double oldMeanF = computeMeanFreq(sim.oldSteps);
                double newMeanSL = computeMeanStepLen(sim.steps);
                double oldMeanSL = computeMeanStepLen(sim.oldSteps);
                double newMeanAmp = computeMeanAmp(sim.steps);
                double oldMeanAmp = computeMeanAmp(sim.oldSteps);
                double newMeanDur = computeMeanDuration(sim.steps);
                double newDurCV = computeDurationCV(sim.steps);

                printf("--- %s ---\n", fname.c_str());
                if (expected > 0) {
                    double accuracy = 100.0 * nNew / expected;
                    printf("  Expected: %d steps | New: %d (%.1f%%) | Old: %d (%.1f%%)\n",
                           expected, nNew, accuracy, nOld, 100.0 * nOld / expected);
                } else {
                    printf("  New: %d steps | Old: %d steps\n", nNew, nOld);
                }
                printf("  New: MeanFreq=%.2fHz  FreqCV=%.3f  MeanDur=%.0fms  DurCV=%.3f  MeanAmp=%.2f  MeanSL=%.3fm\n",
                       newMeanF, newCV, newMeanDur, newDurCV, newMeanAmp, newMeanSL);
                printf("  Old: MeanFreq=%.2fHz  FreqCV=%.3f  MeanAmp=%.2f  MeanSL=%.3fm\n",
                       oldMeanF, oldCV, oldMeanAmp, oldMeanSL);
                printf("  FreqCV improvement: %.1f%%\n",
                       (oldCV > 0.001) ? (1.0 - newCV / oldCV) * 100.0 : 0.0);

                // Per-step frequency distribution (New)
                if (sim.steps.size() >= 5) {
                    std::vector<double> freqs;
                    for (size_t i = 2; i < sim.steps.size(); ++i) {
                        if (sim.steps[i].freqHz > 0.0) freqs.push_back(sim.steps[i].freqHz);
                    }
                    if (!freqs.empty()) {
                        std::sort(freqs.begin(), freqs.end());
                        double p5  = freqs[(int)(freqs.size() * 0.05)];
                        double p25 = freqs[(int)(freqs.size() * 0.25)];
                        double p50 = freqs[(int)(freqs.size() * 0.50)];
                        double p75 = freqs[(int)(freqs.size() * 0.75)];
                        double p95 = freqs[(int)(freqs.size() * 0.95)];
                        printf("  New Freq distribution: P5=%.2f  P25=%.2f  P50=%.2f  P75=%.2f  P95=%.2f Hz\n",
                               p5, p25, p50, p75, p95);
                    }
                }

                // Per-step amplitude distribution (New)
                if (sim.steps.size() >= 3) {
                    std::vector<double> amps;
                    for (auto& s : sim.steps) amps.push_back(s.ampZ);
                    std::sort(amps.begin(), amps.end());
                    printf("  New Amp range: min=%.2f  max=%.2f\n", amps.front(), amps.back());
                }
                printf("\n");
            }
        } else {
            // ===== SUMMARY TABLE (original --batch) =====
            printf("\n%-40s | %5s %5s %8s %8s | %5s %5s %8s %8s | %s\n",
                   "Dataset", "NStep", "NFreq", "NMeanSL", "NFreqCV",
                   "OStep", "OFreq", "OMeanSL", "OFreqCV", "Verdict");
            printf("%-40s-+-%5s-%5s-%8s-%8s-+-%5s-%5s-%8s-%8s-+-%s\n",
                   "----------------------------------------",
                   "-----", "-----", "--------", "--------",
                   "-----", "-----", "--------", "--------", "-------");

            int passCount = 0, totalCount = 0;
            for (auto& path : dataFiles) {
                SimResult sim = runSimulation(path);
                double newCV = computeFreqCV(sim.steps);
                double oldCV = computeFreqCV(sim.oldSteps);
                double newMeanSL = computeMeanStepLen(sim.steps);
                double oldMeanSL = computeMeanStepLen(sim.oldSteps);
                double newMeanF = computeMeanFreq(sim.steps);
                double oldMeanF = computeMeanFreq(sim.oldSteps);

                std::string display = baseName(path);
                if (display.size() > 40) display = display.substr(0, 37) + "...";

                const char* verdict = (newCV <= oldCV + 0.005) ? "OK" : "REGRESS";
                if (newCV <= oldCV + 0.005) ++passCount;
                ++totalCount;

                printf("%-40s | %5d %5.2f %8.3f %8.3f | %5d %5.2f %8.3f %8.3f | %s\n",
                       display.c_str(),
                       (int)sim.steps.size(), newMeanF, newMeanSL, newCV,
                       (int)sim.oldSteps.size(), oldMeanF, oldMeanSL, oldCV,
                       verdict);
            }
            printf("\n=== Result: %d/%d passed (New FreqCV <= Old FreqCV + 0.005) ===\n",
                   passCount, totalCount);
        }
        return 0;
    }

    // --- Select PDR Data File ---
    std::string dataPath;
    if (argc >= 2) {
        dataPath = argv[1];
    } else {
        dataPath = nativeOpenFileDialog("Select PDR Data File", pdrDataDir);
    }
    if (dataPath.empty()) { printf("No PDR data selected.\n"); return 1; }

    // --- Select Map File ---
    std::string mapPath;
    if (argc >= 3) {
        mapPath = argv[2];
    } else {
        mapPath = nativeOpenFileDialog("Select Map File (Cancel for blank)", mapDataDir);
    }

    cv::Mat mapImage = loadMapFile(mapPath);
    if (mapImage.empty()) {
        printf("No map loaded. Creating blank canvas.\n");
        mapImage = makeBlankMap();
    }

    // --- Default Start Point (center of map, click to change anytime) ---
    cv::Point startPt(mapImage.cols / 2, mapImage.rows / 2);

    // --- Run Simulation ---
    SimResult sim = runSimulation(dataPath);
    if (sim.trace.empty()) { printf("No valid data.\n"); return 1; }

    // --- Visualization & Control Loop ---
    int sampleIdx = 0;
    int stride = 25;
    bool autoPlay = false;
    bool showAnalysis = false;
    bool showStepAnalysis = false;
    double mapScale = 10.0;

    MapViewState mapState;
    mapState.rotate90 = 3;
    mapState.flipY = true;

    cv::namedWindow("PDR Dashboard", cv::WINDOW_NORMAL);
    cv::namedWindow("World Map View", cv::WINDOW_NORMAL);

    MapViewClick mapViewClick{&mapState, &startPt, false};
    cv::setMouseCallback("World Map View", onMouseWorldMap, &mapViewClick);

    // Dashboard click state for waveform inspect
    DashboardClickState dashClick;
    dashClick.trace = &sim.trace;
    dashClick.currentSampleIdx = sampleIdx;
    // pdrPath width = (int)(rightH * 1.2) where rightH = 220+220+500 = 940
    dashClick.pdrPathWidth = (int)(940 * 1.2);
    cv::setMouseCallback("PDR Dashboard", onMouseDashboard, &dashClick);

    // Step Analysis click state for scale toggle
    StepAnalysisClick saClick;

    showAtSample(sim.steps, sim.oldSteps, sim.trace, sampleIdx, stride,
                 sim.dataFileName, mapImage, startPt, mapScale, mapState,
                 &dashClick.wfClick);
    showControlsWindow(mapState, stride);

    printf("\n=== Controls ===\n");
    printf("  [Space]: Play/Pause\n");
    printf("  [A]/[D]: Prev/Next Frame\n");
    printf("  [W]/[S]: Stride Up/Down\n");
    printf("  [I/J/K/L]: Move Start Point\n");
    printf("  [U]/[O]: Rotate PDR Path (1 deg)\n");
    printf("  [R]: Rotate Map 90 deg\n");
    printf("  [H]: Flip Horizontal  [V]: Flip Vertical\n");
    printf("  [F]: Open new PDR Data File\n");
    printf("  [M]: Open new Map File\n");
    printf("  [T]: Toggle Analysis Window\n");
    printf("  [B]: Toggle Step Analysis Window\n");
    printf("  [Q]: Quit\n");

    while (true) {
        int waitMs = autoPlay ? 33 : 30;
        int key = cv::waitKey(waitMs);

        bool update = false;

        // Check toggle button clicks on World Map View
        if (mapViewClick.changed) {
            mapViewClick.changed = false;
            update = true;
        }

        // Check waveform click on Dashboard
        if (dashClick.changed) {
            dashClick.changed = false;
            update = true;
        }

        int charKey = key & 0xFF;

        if (charKey == 'q' || charKey == 27) break;

        // --- File Reload ---
        if (charKey == 'f') {
            std::string lastDir = dirName(dataPath);
            std::string newPath = nativeOpenFileDialog("Select PDR Data File", lastDir);
            if (!newPath.empty()) {
                SimResult newSim = runSimulation(newPath);
                if (!newSim.trace.empty()) {
                    sim = std::move(newSim);
                    dataPath = newPath;
                    sampleIdx = 0;
                    autoPlay = false;
                    dashClick.trace = &sim.trace;
                    dashClick.wfClick.active = false;
                    update = true;
                    printf("Loaded: %s\n", sim.dataFileName.c_str());
                } else {
                    printf("Failed to load: %s\n", baseName(newPath).c_str());
                }
            }
        }

        if (charKey == 'm') {
            std::string lastDir = mapPath.empty() ? mapDataDir : dirName(mapPath);
            std::string newMap = nativeOpenFileDialog("Select Map File", lastDir);
            if (!newMap.empty()) {
                cv::Mat newImg = loadMapFile(newMap);
                if (!newImg.empty()) {
                    mapImage = newImg;
                    mapPath = newMap;
                    startPt = {mapImage.cols/2, mapImage.rows/2};
                    mapState.moveX = 0;
                    mapState.moveY = 0;
                    update = true;
                    printf("Map loaded: %s\n", baseName(newMap).c_str());
                }
            }
        }

        // --- PDR Playback Control ---
        if (charKey == 'd') { sampleIdx = std::min(sampleIdx + stride, (int)sim.trace.size() - 1); update = true; }
        if (charKey == 'a') { sampleIdx = std::max(sampleIdx - stride, 0); update = true; }
        if (charKey == 'w') { stride = std::min(stride * 2, 500); printf("Stride: %d\n", stride); update = true; }
        if (charKey == 's') { stride = std::max(stride / 2, 1); printf("Stride: %d\n", stride); update = true; }
        if (charKey == 32) autoPlay = !autoPlay;

        // --- Map Control ---
        if (charKey == 'r') { mapState.rotate90 = (mapState.rotate90 + 1) % 4; update = true; }
        if (charKey == 'h') { mapState.flipX = !mapState.flipX; update = true; }
        if (charKey == 'v') { mapState.flipY = !mapState.flipY; update = true; }

        // --- Path Visibility Toggle ---
        if (charKey == '1') { mapState.showNewPath = !mapState.showNewPath; update = true; }
        if (charKey == '2') { mapState.showOldPath = !mapState.showOldPath; update = true; }

        // --- PDR Rotation (1 deg) ---
        if (charKey == 'u') { mapState.pdrRotateDeg -= 1.0; update = true; }
        if (charKey == 'o') { mapState.pdrRotateDeg += 1.0; update = true; }

        // --- Toggle Analysis Window ---
        if (charKey == 't') {
            showAnalysis = !showAnalysis;
            if (showAnalysis) {
                cv::namedWindow("Analysis", cv::WINDOW_NORMAL);
                update = true;
            } else {
                cv::destroyWindow("Analysis");
            }
        }

        // --- Toggle Step Analysis Window ---
        if (charKey == 'b') {
            showStepAnalysis = !showStepAnalysis;
            if (showStepAnalysis) {
                cv::namedWindow("Step Analysis", cv::WINDOW_NORMAL);
                cv::setMouseCallback("Step Analysis", onMouseStepAnalysis, &saClick);
                update = true;
            } else {
                cv::destroyWindow("Step Analysis");
            }
        }

        // --- Scale Toggle from Step Analysis click ---
        if (saClick.scaleToggled) {
            saClick.scaleToggled = false;
            int newFemale = sim.isFemale ? 0 : 1;
            printf("  >> Scale toggled: %s\n", newFemale ? "Female x0.90" : "Male x1.00");
            SimResult newSim = runSimulation(dataPath, newFemale);
            if (!newSim.trace.empty()) {
                sim = std::move(newSim);
                sampleIdx = std::min(sampleIdx, (int)sim.trace.size() - 1);
                dashClick.trace = &sim.trace;
                update = true;
            }
        }

        // --- Move Start Point (I/J/K/L) ---
        int moveStep = 5;
        if (charKey == 'i') { mapState.moveY -= moveStep; update = true; }
        if (charKey == 'k') { mapState.moveY += moveStep; update = true; }
        if (charKey == 'j') { mapState.moveX -= moveStep; update = true; }
        if (charKey == 'l') { mapState.moveX += moveStep; update = true; }

        // --- Arrow Keys ---
        if (key == 65362 || key == 2490368) { mapState.moveY -= moveStep; update = true; }
        if (key == 65364 || key == 2621440) { mapState.moveY += moveStep; update = true; }
        if (key == 65361 || key == 2424832) { mapState.moveX -= moveStep; update = true; }
        if (key == 65363 || key == 2555904) { mapState.moveX += moveStep; update = true; }

        // --- Auto Play ---
        if (autoPlay) {
            sampleIdx = std::min(sampleIdx + stride, (int)sim.trace.size() - 1);
            update = true;
            if (sampleIdx >= (int)sim.trace.size() - 1) autoPlay = false;
        }

        if (update) {
            dashClick.currentSampleIdx = sampleIdx;
            showAtSample(sim.steps, sim.oldSteps, sim.trace, sampleIdx, stride,
                         sim.dataFileName, mapImage, startPt, mapScale, mapState,
                         &dashClick.wfClick);
            showControlsWindow(mapState, stride);
            if (showAnalysis) {
                showAnalysisWindow(sim.steps, sim.oldSteps, sim.trace, sampleIdx);
            }
            if (showStepAnalysis) {
                showStepAnalysisWindow(sim.steps, sim.oldSteps, sampleIdx, sim.isFemale);
            }
        }
    }

    cv::destroyAllWindows();
    return 0;
}
