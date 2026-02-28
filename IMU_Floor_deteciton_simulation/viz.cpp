#include "viz.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// ============================================================
//  Constants
// ============================================================
static const int MAIN_W = 1000;
static const int ML = 50;       // left margin
static const int MR = 10;       // right margin
static const int SCROLL_WIN = 500;  // visible samples in waveform

static const cv::Scalar BG_COLOR(30, 30, 30);
static const cv::Scalar GRID_COLOR(60, 60, 60);
static const cv::Scalar TEXT_COLOR(210, 210, 210);
static const cv::Scalar TEXT_DIM(130, 130, 130);
static const cv::Scalar STEP_MARKER_COLOR(255, 200, 80);

// ============================================================
//  Value extractors
// ============================================================
static double getPureX(const FloorTraceData& d)  { return d.pureX; }
static double getPureY(const FloorTraceData& d)  { return d.pureY; }
static double getLocalForwardY(const FloorTraceData& d) { return d.localForwardY; }
static double getLocalLateralX(const FloorTraceData& d) { return d.localLateralX; }
static double getPureZ(const FloorTraceData& d)  { return d.pureZ; }
static double getNegPureY(const FloorTraceData& d) { return -d.pureY; }
static double getEnvZ(const FloorTraceData& d) { return d.envZ; }
static double getEnvH(const FloorTraceData& d) { return d.envH; }
static double getStairScore(const FloorTraceData& d) { return d.stairScore; }
static double getEnergyFracZ(const FloorTraceData& d) { return d.energyFracZ; }
static double getProcRmsZ(const FloorTraceData& d) { return d.procRmsZ; }
static double getProcStepFreq(const FloorTraceData& d) { return d.procStepFreq * 2.0; }
static double getRmsZdivFreq(const FloorTraceData& d) { return d.rmsZdivFreq; }
static double getStepVarH(const FloorTraceData& d) { return d.stepVarH; }
static double getStepFreqHz(const FloorTraceData& d) { return d.stepFreqHz; }
static double getVarYZRatio(const FloorTraceData& d) { return d.varYZRatio; }
static double getDmagRatio(const FloorTraceData& d) { return d.dmagRatio; }
static double getZfRatio(const FloorTraceData& d) { return d.zfRatio; }
static double getFreqRatio(const FloorTraceData& d) { return d.freqRatio; }
static double getDmag(const FloorTraceData& d) { return d.dmag; }
static double getRecentStepVarYZ(const FloorTraceData& d) { return d.recentStepVarYZ; }
static double getGlobalZ(const FloorTraceData& d) { return d.globalZ; }
static double getGyroAngle(const FloorTraceData& d) { return d.gyroAngleDeg; }

// ============================================================
//  Helper: draw floor-label background bands
// ============================================================
static void drawFloorBands(cv::Mat& canvas,
                           const std::vector<FloorTraceData>& trace,
                           int startIdx, int count, int xOff,
                           double ppx, int yTop, int yBot,
                           int canvasW) {
    if (count <= 0) return;
    cv::Mat overlay = canvas.clone();
    for (int i = 0; i < count; ++i) {
        int idx = startIdx + i;
        if (idx < 0 || idx >= (int)trace.size()) continue;
        int x  = ML + (int)((xOff + i) * ppx);
        int x2 = ML + (int)((xOff + i + 1) * ppx);
        if (x2 <= x) x2 = x + 1;
        if (x >= canvasW - MR || x2 <= ML) continue;
        x  = std::max(x, ML);
        x2 = std::min(x2, canvasW - MR);
        cv::rectangle(overlay, {x, yTop}, {x2, yBot},
                      floorLabelColor(trace[idx].floorLabel), cv::FILLED);
    }
    cv::addWeighted(overlay, 0.25, canvas, 0.75, 0.0, canvas);
}

// ============================================================
//  Helper: FSM prediction background bands
// ============================================================
static void drawFsmBands(cv::Mat& canvas,
                         const std::vector<FloorTraceData>& trace,
                         int startIdx, int count, int xOff,
                         double ppx, int yTop, int yBot,
                         int canvasW) {
    if (count <= 0) return;
    cv::Mat overlay = canvas.clone();
    for (int i = 0; i < count; ++i) {
        int idx = startIdx + i;
        if (idx < 0 || idx >= (int)trace.size()) continue;
        int x  = ML + (int)((xOff + i) * ppx);
        int x2 = ML + (int)((xOff + i + 1) * ppx);
        if (x2 <= x) x2 = x + 1;
        if (x >= canvasW - MR || x2 <= ML) continue;
        x  = std::max(x, ML);
        x2 = std::min(x2, canvasW - MR);
        cv::Scalar col = floorLabelColor(trace[idx].fsmLabel);
        cv::rectangle(overlay, {x, yTop}, {x2, yBot}, col, cv::FILLED);
    }
    cv::addWeighted(overlay, 0.2, canvas, 0.8, 0.0, canvas);
}

// ============================================================
//  Helper: current-sample cursor (yellow vertical line)
// ============================================================
static void drawCursor(cv::Mat& canvas, int sampleIdx, int yTop, int yBot, int canvasW) {
    int start = std::max(0, sampleIdx - SCROLL_WIN + 1);
    int xOff  = SCROLL_WIN - (sampleIdx - start + 1);
    double ppx = (double)(canvasW - ML - MR) / SCROLL_WIN;
    int curX = ML + (int)((xOff + (sampleIdx - start)) * ppx);
    cv::line(canvas, {curX, yTop}, {curX, yBot}, cv::Scalar(0, 255, 255), 1);
}

// ============================================================
//  Generic waveform drawer (multi-line support)
// ============================================================
struct WaveLine {
    double (*getValue)(const FloorTraceData&);
    cv::Scalar color;
    int thickness;
    const char* legend;
    bool dashed;  // dashed line style
};

// Helper to draw a dashed line
static void drawDashedLine(cv::Mat& canvas, cv::Point p1, cv::Point p2,
                           const cv::Scalar& color, int thickness, int dashLen = 6) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist < 1.0) return;
    double ux = dx / dist, uy = dy / dist;
    double drawn = 0.0;
    bool on = true;
    while (drawn < dist) {
        double seg = std::min((double)dashLen, dist - drawn);
        if (on) {
            cv::Point a(p1.x + (int)(drawn * ux), p1.y + (int)(drawn * uy));
            cv::Point b(p1.x + (int)((drawn + seg) * ux), p1.y + (int)((drawn + seg) * uy));
            cv::line(canvas, a, b, color, thickness, cv::LINE_AA);
        }
        drawn += seg;
        on = !on;
    }
}

static void drawMultiWaveform(cv::Mat& canvas,
                              const std::vector<FloorTraceData>& trace,
                              int sampleIdx,
                              int yOffset, int height,
                              const char* title,
                              const WaveLine* lines, int lineCount,
                              int canvasW,
                              bool floorBands = true,
                              bool stepMarkers = false,
                              double fixedMin = 1e30, double fixedMax = -1e30) {
    int start = std::max(0, sampleIdx - SCROLL_WIN + 1);
    int count = sampleIdx - start + 1;
    int xOff  = SCROLL_WIN - count;
    double ppx = (double)(canvasW - ML - MR) / SCROLL_WIN;

    // Floor label bands
    if (floorBands)
        drawFloorBands(canvas, trace, start, count, xOff, ppx, yOffset, yOffset + height, canvasW);

    // Y-range
    double minV = fixedMin, maxV = fixedMax;
    if (minV > maxV) {
        // auto-range from entire dataset
        minV = 1e9; maxV = -1e9;
        for (int i = 0; i < (int)trace.size(); ++i) {
            for (int li = 0; li < lineCount; ++li) {
                double v = lines[li].getValue(trace[i]);
                if (v < minV) minV = v;
                if (v > maxV) maxV = v;
            }
        }
    }
    double range = maxV - minV;
    if (range < 0.1) { range = 1.0; minV -= 0.5; maxV += 0.5; }
    double pad = range * 0.1;
    minV -= pad; maxV += pad;
    range = maxV - minV;

    auto toY = [&](double v) -> int {
        return yOffset + height - (int)((v - minV) / range * (height - 20)) - 10;
    };

    // Grid: zero line
    if (minV <= 0.0 && maxV >= 0.0) {
        int y0 = toY(0.0);
        cv::line(canvas, {ML, y0}, {canvasW - MR, y0}, GRID_COLOR, 1);
        cv::putText(canvas, "0", {5, y0 + 4}, cv::FONT_HERSHEY_PLAIN, 0.9, TEXT_DIM, 1);
    }

    // Y-axis range labels
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2f", maxV);
        cv::putText(canvas, buf, {2, yOffset + 15}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_DIM, 1);
        snprintf(buf, sizeof(buf), "%.2f", minV);
        cv::putText(canvas, buf, {2, yOffset + height - 5}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_DIM, 1);
    }

    // Title
    cv::putText(canvas, title, {ML + 5, yOffset + 15},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1);

    // Draw each line
    for (int li = 0; li < lineCount; ++li) {
        cv::Point prev(-1, -1);
        for (int i = 0; i < count; ++i) {
            int idx = start + i;
            if (idx >= (int)trace.size()) break;
            double v = lines[li].getValue(trace[idx]);
            int x = ML + (int)((xOff + i) * ppx);
            int y = toY(v);
            cv::Point pt(x, y);
            if (prev.x >= 0) {
                if (lines[li].dashed) {
                    drawDashedLine(canvas, prev, pt, lines[li].color, lines[li].thickness);
                } else {
                    cv::line(canvas, prev, pt, lines[li].color, lines[li].thickness, cv::LINE_AA);
                }
            }
            prev = pt;
        }
    }

    // Step markers
    if (stepMarkers) {
        for (int i = 0; i < count; ++i) {
            int idx = start + i;
            if (idx >= (int)trace.size()) break;
            if (trace[idx].isStep) {
                int x = ML + (int)((xOff + i) * ppx);
                cv::line(canvas, {x, yOffset + 2}, {x, yOffset + height - 2},
                         STEP_MARKER_COLOR, 1);
            }
        }
    }

    // Legend (top-right)
    {
        int lx = canvasW - MR - 10;
        int ly = yOffset + 15;
        for (int li = lineCount - 1; li >= 0; --li) {
            if (!lines[li].legend) continue;
            int textW = (int)strlen(lines[li].legend) * 7 + 20;
            lx -= textW;
            if (lines[li].dashed) {
                drawDashedLine(canvas, {lx, ly - 4}, {lx + 14, ly - 4}, lines[li].color, 2, 4);
            } else {
                cv::line(canvas, {lx, ly - 4}, {lx + 14, ly - 4}, lines[li].color, 2);
            }
            cv::putText(canvas, lines[li].legend, {lx + 17, ly},
                        cv::FONT_HERSHEY_PLAIN, 0.85, TEXT_COLOR, 1);
        }
    }

    // Border
    cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
}

// ============================================================
//  Helper: draw threshold line
// ============================================================
static void drawThresholdLine(cv::Mat& canvas,
                              double threshVal, double minV, double maxV,
                              int yOffset, int height,
                              int canvasW, const cv::Scalar& color) {
    double range = maxV - minV;
    if (range < 0.01) return;
    int y = yOffset + height - (int)((threshVal - minV) / range * (height - 20)) - 10;
    if (y >= yOffset && y <= yOffset + height) {
        drawDashedLine(canvas, {ML, y}, {canvasW - MR, y}, color, 1, 8);
        char buf[16];
        snprintf(buf, sizeof(buf), "%.2f", threshVal);
        cv::putText(canvas, buf, {canvasW - MR - 40, y - 3},
                    cv::FONT_HERSHEY_PLAIN, 0.75, color, 1);
    }
}

// ============================================================
//  Helper: draw GT vs Pred comparison strip
// ============================================================
static void drawPredictionStrip(cv::Mat& canvas,
                                const std::vector<FloorTraceData>& trace,
                                int sampleIdx,
                                int yOffset, int canvasW) {
    int stripH = 18;
    int start = std::max(0, sampleIdx - SCROLL_WIN + 1);
    int count = sampleIdx - start + 1;
    int xOff  = SCROLL_WIN - count;
    double ppx = (double)(canvasW - ML - MR) / SCROLL_WIN;

    // GT strip
    cv::putText(canvas, "GT", {8, yOffset + 13}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_DIM, 1);
    for (int i = 0; i < count; ++i) {
        int idx = start + i;
        if (idx >= (int)trace.size()) break;
        int x  = ML + (int)((xOff + i) * ppx);
        int x2 = ML + (int)((xOff + i + 1) * ppx);
        if (x2 <= x) x2 = x + 1;
        x = std::max(x, ML); x2 = std::min(x2, canvasW - MR);
        cv::rectangle(canvas, {x, yOffset}, {x2, yOffset + stripH},
                      floorLabelColor(trace[idx].floorLabel), cv::FILLED);
    }

    // Pred strip
    int predY = yOffset + stripH + 2;
    cv::putText(canvas, "Pred", {2, predY + 13}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_DIM, 1);
    for (int i = 0; i < count; ++i) {
        int idx = start + i;
        if (idx >= (int)trace.size()) break;
        int x  = ML + (int)((xOff + i) * ppx);
        int x2 = ML + (int)((xOff + i + 1) * ppx);
        if (x2 <= x) x2 = x + 1;
        x = std::max(x, ML); x2 = std::min(x2, canvasW - MR);
        cv::rectangle(canvas, {x, predY}, {x2, predY + stripH},
                      floorLabelColor(trace[idx].fsmLabel), cv::FILLED);
    }

    cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, predY + stripH}, GRID_COLOR, 1);
}

// ============================================================
//  Helper: draw full-data GT minimap (shows entire dataset GT at a glance)
// ============================================================
static void drawGTMinimap(cv::Mat& canvas,
                          const std::vector<FloorTraceData>& trace,
                          int sampleIdx, int totalSamples,
                          int yOffset, int canvasW,
                          const std::vector<std::string>& gtLabels = {}) {
    int mapH = 12;
    int plotW = canvasW - ML - MR;
    int total = std::max(totalSamples, (int)trace.size());
    if (total <= 0) return;

    cv::putText(canvas, "Map", {6, yOffset + 10}, cv::FONT_HERSHEY_PLAIN, 0.7, TEXT_DIM, 1);

    // Draw GT color for each pixel column
    // Use gtLabels (full data) if available, otherwise fallback to trace
    for (int px = 0; px < plotW; ++px) {
        int idx = (int)((double)px / plotW * total);
        std::string label;
        if (!gtLabels.empty() && idx < (int)gtLabels.size()) {
            label = gtLabels[idx];
        } else if (idx < (int)trace.size()) {
            label = trace[idx].floorLabel;
        } else {
            // Not yet fed and no GT available — draw dark gray
            cv::line(canvas, {ML + px, yOffset}, {ML + px, yOffset + mapH},
                     cv::Scalar(45, 45, 45), 1);
            continue;
        }
        cv::line(canvas, {ML + px, yOffset}, {ML + px, yOffset + mapH},
                 floorLabelColor(label), 1);
    }

    // Current position marker (yellow triangle)
    int curX = ML + (int)((double)sampleIdx / total * plotW);
    curX = std::max(ML, std::min(curX, ML + plotW - 1));
    cv::line(canvas, {curX, yOffset}, {curX, yOffset + mapH}, cv::Scalar(0, 255, 255), 2);

    // Visible window indicator (white bracket)
    int winStart = std::max(0, sampleIdx - SCROLL_WIN + 1);
    int x1 = ML + (int)((double)winStart / total * plotW);
    int x2 = ML + (int)((double)(sampleIdx + 1) / total * plotW);
    x1 = std::max(ML, x1);
    x2 = std::min(ML + plotW, x2);
    cv::rectangle(canvas, {x1, yOffset}, {x2, yOffset + mapH}, cv::Scalar(255, 255, 255), 1);

    // Legend (right side)
    int lx = canvasW - MR - 180;
    int ly = yOffset + 10;
    cv::rectangle(canvas, {lx, ly - 7}, {lx + 8, ly + 1}, floorLabelColor("ground"), cv::FILLED);
    cv::putText(canvas, "gnd", {lx + 10, ly}, cv::FONT_HERSHEY_PLAIN, 0.7, TEXT_DIM, 1);
    lx += 35;
    cv::rectangle(canvas, {lx, ly - 7}, {lx + 8, ly + 1}, floorLabelColor("up"), cv::FILLED);
    cv::putText(canvas, "up", {lx + 10, ly}, cv::FONT_HERSHEY_PLAIN, 0.7, TEXT_DIM, 1);
    lx += 28;
    cv::rectangle(canvas, {lx, ly - 7}, {lx + 8, ly + 1}, floorLabelColor("down"), cv::FILLED);
    cv::putText(canvas, "dn", {lx + 10, ly}, cv::FONT_HERSHEY_PLAIN, 0.7, TEXT_DIM, 1);

    cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + mapH}, GRID_COLOR, 1);
}

// ============================================================
//  Helper: info bar
// ============================================================
static void drawInfoBar(cv::Mat& canvas,
                        const std::vector<FloorTraceData>& trace,
                        int sampleIdx,
                        const std::string& dataFileName,
                        int yOffset, int height,
                        int canvasW,
                        bool isLiveMode, bool isRecording) {
    if (sampleIdx < 0 || sampleIdx >= (int)trace.size()) return;
    const FloorTraceData& cur = trace[sampleIdx];
    char buf[256];
    int y = yOffset + 16;

    // Line 1: file + time + sample + steps
    snprintf(buf, sizeof(buf), "%s  |  t=%.3fs  sample %d/%d  |  Steps: %d%s",
             dataFileName.c_str(), cur.time_s, sampleIdx + 1, (int)trace.size(),
             cur.totalSteps, cur.isStep ? "  [STEP]" : "");
    cv::putText(canvas, buf, {ML + 5, y}, cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cur.isStep ? STEP_MARKER_COLOR : TEXT_COLOR, 1);

    // Line 2: GT + FSM prediction + features
    y += 16;
    snprintf(buf, sizeof(buf), "GT: %s", cur.floorLabel.c_str());
    cv::putText(canvas, buf, {ML + 5, y}, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                floorLabelColor(cur.floorLabel), 1);

    snprintf(buf, sizeof(buf), "FSM: %s (conf:%.2f)", cur.fsmLabel.c_str(), cur.stairConfidence);
    cv::putText(canvas, buf, {ML + 130, y}, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                floorLabelColor(cur.fsmLabel), 1);

    snprintf(buf, sizeof(buf), "vYZ:%.2f dm:%.2f zf:%.2f fq:%.2f | sVarYZ:%.3f dmag:%.2f",
             cur.varYZRatio, cur.dmagRatio, cur.zfRatio, cur.freqRatio,
             cur.recentStepVarYZ, cur.dmag);
    cv::putText(canvas, buf, {ML + 340, y}, cv::FONT_HERSHEY_SIMPLEX, 0.33, TEXT_DIM, 1);

    // Mode indicators (right side)
    if (isLiveMode) {
        cv::putText(canvas, "LIVE", {canvasW - MR - 100, yOffset + 16},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);
    } else {
        cv::putText(canvas, "PAUSE", {canvasW - MR - 100, yOffset + 16},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 200, 255), 1);
    }

    if (isRecording) {
        // Red blinking REC indicator
        cv::circle(canvas, {canvasW - MR - 30, yOffset + 12}, 5, cv::Scalar(0, 0, 255), cv::FILLED);
        cv::putText(canvas, "REC", {canvasW - MR - 22, yOffset + 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 255), 1);
    }

    cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
}

// ============================================================
//  Window 1: "Floor Detection" (MAIN_W x 700)
//
//  Row 0 (150px): pureZ(black) + -pureY(magenta) + floor GT bands
//  Row 1 (150px): varYZRatio(cyan) + zfRatio(magenta) + freqRatio(gray)
//                 + baseline(1.0) + thresholds (0.6, 1.3)
//  Row 2 (150px): FSM Pred background + pureZ waveform + step markers
//  Row 3 (44px):  GT vs Pred color strip (2 rows)
//  Row 3b(18px):  GT minimap (전체 데이터 GT 분포)
//  Row 4 (50px):  Info bar + REC indicator
//  Row 5 (26px):  Control hints
// ============================================================
void showMainWindow(const std::vector<FloorTraceData>& trace, int sampleIdx,
                    const std::string& dataFileName, bool isLiveMode, bool isRecording,
                    int totalSamples, const std::vector<std::string>& gtLabels) {
    const int ROW0A_H = 100;  // pureZ
    const int ROW0B_H = 100;  // pureY
    const int ROW1_H = 150;
    const int ROW2_H = 150;
    const int ROW3_H = 44;
    const int ROW3B_H = 18;  // GT minimap
    const int ROW4_H = 50;
    const int ROW5_H = 26;
    const int TOTAL_H = ROW0A_H + ROW0B_H + ROW1_H + ROW2_H + ROW3_H + ROW3B_H + ROW4_H + ROW5_H;

    cv::Mat canvas(TOTAL_H, MAIN_W, CV_8UC3, BG_COLOR);

    int y0 = 0;

    // ── Row 0a: pureZ + floor GT bands ──
    {
        WaveLine lines[] = {
            {getPureZ, cv::Scalar(200, 200, 200), 2, "pureZ", false},
        };
        drawMultiWaveform(canvas, trace, sampleIdx, y0, ROW0A_H,
                          "pureZ", lines, 1, MAIN_W,
                          true, false);
    }
    y0 += ROW0A_H;

    // ── Row 0b: localForwardY + floor GT bands ──
    {
        WaveLine lines[] = {
            {getLocalForwardY, cv::Scalar(255, 0, 255), 2, "localFwdY", false},
        };
        drawMultiWaveform(canvas, trace, sampleIdx, y0, ROW0B_H,
                          "localForwardY (body-forward)", lines, 1, MAIN_W,
                          true, false);
    }
    y0 += ROW0B_H;

    // ── Row 1: Envelope (envZ, envH, stairScore) ──
    //   stairScore = envZ - envH: > 0 = stair (수직 우세), ≈ 0 = ground
    {
        WaveLine lines[] = {
            {getEnvZ,       cv::Scalar(100, 200, 255), 2, "envZ",       false},
            {getEnvH,       cv::Scalar(255, 150, 100), 2, "envH",       false},
            {getStairScore, cv::Scalar(0, 255, 0),     2, "stairScore", false},
        };
        drawMultiWaveform(canvas, trace, sampleIdx, y0, ROW1_H,
                          "Envelope: envZ / envH / stairScore", lines, 3, MAIN_W,
                          true, true);
    }
    y0 += ROW1_H;

    // ── Row 2: GT background + pureZ + step markers ──
    {
        // pureZ waveform with GT floor bands as background
        WaveLine lines[] = {
            {getPureZ, cv::Scalar(200, 200, 200), 2, "pureZ", false},
        };
        drawMultiWaveform(canvas, trace, sampleIdx, y0, ROW2_H,
                          "GT + pureZ + steps", lines, 1, MAIN_W,
                          true, true);  // GT bands enabled
    }
    y0 += ROW2_H;

    // ── Row 3: GT vs Pred strip ──
    drawPredictionStrip(canvas, trace, sampleIdx, y0, MAIN_W);
    y0 += ROW3_H;

    // ── Row 3b: GT minimap (전체 데이터 GT 분포) ──
    drawGTMinimap(canvas, trace, sampleIdx, totalSamples, y0, MAIN_W, gtLabels);
    y0 += ROW3B_H;

    // ── Row 4: Info bar ──
    drawInfoBar(canvas, trace, sampleIdx, dataFileName, y0, ROW4_H, MAIN_W,
                isLiveMode, isRecording);
    y0 += ROW4_H;

    // ── Row 5: Control hints ──
    {
        const char* hints = "Space:Live/Pause  A/D:Scrub  W/S:Stride  1/2/5/0:Speed  R:Record  F:File  Q:Quit";
        cv::putText(canvas, hints, {ML + 5, y0 + 17},
                    cv::FONT_HERSHEY_SIMPLEX, 0.33, TEXT_DIM, 1);
    }

    // Cursor across rows 0a-2
    drawCursor(canvas, sampleIdx, 0, ROW0A_H + ROW0B_H + ROW1_H + ROW2_H, MAIN_W);

    cv::imshow("Floor Detection", canvas);
}

// ============================================================
//  Window 2: "Analysis" (500x500)
//
//  Top (250px): 2D scatter: pureY(X) vs pureZ(Y), recent step trajectory
//  Middle (200px): Step Start Points scatter (누적, GT label 색상)
//  Status (50px): features + FSM
// ============================================================
// Per-segment colors (스텝 구간별 구분 색상)
static const cv::Scalar SEG_COLORS[] = {
    cv::Scalar(255, 180, 60),   // cyan-ish
    cv::Scalar(60, 180, 255),   // orange-ish
    cv::Scalar(180, 255, 60),   // lime
    cv::Scalar(255, 60, 180),   // pink
};
static const int NUM_SEG_COLORS = 4;

static void drawYZTrajectory(cv::Mat& canvas,
                             const std::vector<FloorTraceData>& trace,
                             int sampleIdx,
                             int yOffset, int height, int canvasW) {
    // Step boundary 위치 수집: sampleIdx 이전에서 최근 3개 step 찾기
    // → 최대 3개 구간 (step0~step1, step1~step2, step2~현재)
    std::vector<int> stepPositions;
    for (int i = sampleIdx; i >= 0; --i) {
        if (trace[i].isStep) {
            stepPositions.push_back(i);
            if ((int)stepPositions.size() >= 3) break;
        }
    }
    std::reverse(stepPositions.begin(), stepPositions.end());

    // 범위: 가장 오래된 step부터 현재 sampleIdx까지
    int rangeStart = stepPositions.empty() ? std::max(0, sampleIdx - 100) : stepPositions[0];
    if (rangeStart < 0) rangeStart = 0;

    // 각 구간의 시작/끝 인덱스 계산
    struct Segment { int begin; int end; int segIdx; };
    std::vector<Segment> segments;
    {
        int prevStart = rangeStart;
        int segIdx = 0;
        for (int sp : stepPositions) {
            if (sp > prevStart) {
                segments.push_back({prevStart, sp - 1, segIdx});
                segIdx++;
            }
            prevStart = sp;
        }
        // 마지막 step ~ 현재
        segments.push_back({prevStart, sampleIdx, segIdx});
    }

    // Y/Z 범위 계산
    double minY = 1e9, maxY = -1e9;
    double minZ = 1e9, maxZ = -1e9;
    for (int i = rangeStart; i <= sampleIdx && i < (int)trace.size(); ++i) {
        double py = trace[i].localForwardY, pz = trace[i].pureZ;
        if (py < minY) minY = py; if (py > maxY) maxY = py;
        if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;
    }
    double padY = (maxY - minY) * 0.1 + 0.5;
    double padZ = (maxZ - minZ) * 0.1 + 0.5;
    minY -= padY; maxY += padY;
    minZ -= padZ; maxZ += padZ;
    double rangeY = maxY - minY;
    double rangeZ = maxZ - minZ;
    if (rangeY < 0.1) rangeY = 1.0;
    if (rangeZ < 0.1) rangeZ = 1.0;

    int plotW = canvasW - ML - MR;
    int plotH = height - 20;

    auto toX = [&](double v) -> int {
        return ML + (int)((v - minY) / rangeY * plotW);
    };
    auto toYp = [&](double v) -> int {
        return yOffset + height - 10 - (int)((v - minZ) / rangeZ * plotH);
    };

    // Grid: axes
    if (minY <= 0.0 && maxY >= 0.0) {
        int x0 = toX(0.0);
        cv::line(canvas, {x0, yOffset}, {x0, yOffset + height}, GRID_COLOR, 1);
    }
    if (minZ <= 0.0 && maxZ >= 0.0) {
        int y0 = toYp(0.0);
        cv::line(canvas, {ML, y0}, {canvasW - MR, y0}, GRID_COLOR, 1);
    }

    // Title
    cv::putText(canvas, "2D Scatter: localFwdY(X) vs pureZ(Y)  [per step segment]",
                {ML + 5, yOffset + 15}, cv::FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1);
    cv::putText(canvas, "fwdY ->", {canvasW - MR - 55, yOffset + height - 5},
                cv::FONT_HERSHEY_PLAIN, 0.75, TEXT_DIM, 1);
    cv::putText(canvas, "pureZ", {2, yOffset + height / 2},
                cv::FONT_HERSHEY_PLAIN, 0.75, TEXT_DIM, 1);

    // 각 구간별로 다른 색으로 궤적 그리기
    for (const auto& seg : segments) {
        cv::Scalar segCol = SEG_COLORS[seg.segIdx % NUM_SEG_COLORS];
        // 과거 구간은 좀 어둡게
        bool isLatest = (seg.end == sampleIdx) ||
                        (&seg == &segments.back());
        double alpha = isLatest ? 1.0 : 0.5;
        cv::Scalar drawCol(segCol[0] * alpha, segCol[1] * alpha, segCol[2] * alpha);

        cv::Point prev(-1, -1);
        for (int i = seg.begin; i <= seg.end && i < (int)trace.size(); ++i) {
            int x = toX(trace[i].localForwardY);
            int y = toYp(trace[i].pureZ);

            cv::circle(canvas, {x, y}, isLatest ? 2 : 1, drawCol, cv::FILLED);

            cv::Point pt(x, y);
            if (prev.x >= 0) {
                cv::line(canvas, prev, pt, drawCol, 1, cv::LINE_AA);
            }
            prev = pt;
        }

        // Step boundary marker (구간 시작점)
        if (seg.begin < (int)trace.size()) {
            int x = toX(trace[seg.begin].localForwardY);
            int y = toYp(trace[seg.begin].pureZ);
            cv::circle(canvas, {x, y}, 5, STEP_MARKER_COLOR, 2);
            // 구간 번호 표시
            char label[8];
            snprintf(label, sizeof(label), "S%d", seg.segIdx);
            cv::putText(canvas, label, {x + 6, y - 4},
                        cv::FONT_HERSHEY_PLAIN, 0.75, STEP_MARKER_COLOR, 1);
        }
    }

    // Current point highlighted
    if (sampleIdx >= 0 && sampleIdx < (int)trace.size()) {
        int x = toX(trace[sampleIdx].localForwardY);
        int y = toYp(trace[sampleIdx].pureZ);
        cv::circle(canvas, {x, y}, 4, cv::Scalar(0, 255, 255), 2);
    }

    // GT label of current segment (우하단)
    if (sampleIdx >= 0 && sampleIdx < (int)trace.size()) {
        char buf[64];
        snprintf(buf, sizeof(buf), "GT: %s", trace[sampleIdx].floorLabel.c_str());
        cv::putText(canvas, buf, {ML + 5, yOffset + height - 8},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    floorLabelColor(trace[sampleIdx].floorLabel), 1);
    }

    // Segment legend (우상단)
    {
        int lx = canvasW - MR - 10;
        int ly = yOffset + 15;
        for (int si = (int)segments.size() - 1; si >= 0; --si) {
            cv::Scalar col = SEG_COLORS[segments[si].segIdx % NUM_SEG_COLORS];
            char label[8];
            snprintf(label, sizeof(label), "S%d", segments[si].segIdx);
            int tw = (int)strlen(label) * 7 + 20;
            lx -= tw;
            cv::line(canvas, {lx, ly - 4}, {lx + 14, ly - 4}, col, 2);
            cv::putText(canvas, label, {lx + 17, ly},
                        cv::FONT_HERSHEY_PLAIN, 0.85, TEXT_COLOR, 1);
        }
    }

    cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
}

// ============================================================
//  Step Start Scatter: 모든 스텝 시작점을 GT label 색상으로 누적 표시
//  X축: pureY (at step), Y축: pureZ (at step)
// ============================================================
static void drawStepStartScatter(cv::Mat& canvas,
                                  const std::vector<FloorTraceData>& trace,
                                  int sampleIdx,
                                  int yOffset, int height, int canvasW) {
    // 모든 스텝 위치 수집 (0 ~ sampleIdx)
    struct StepPoint { double py; double pz; std::string label; int idx; };
    std::vector<StepPoint> steps;
    for (int i = 0; i <= sampleIdx && i < (int)trace.size(); ++i) {
        if (trace[i].isStep) {
            steps.push_back({trace[i].localForwardY, trace[i].pureZ, trace[i].floorLabel, i});
        }
    }

    // Y/Z 범위 계산 (모든 스텝 기준)
    double minY = 1e9, maxY = -1e9;
    double minZ = 1e9, maxZ = -1e9;
    for (const auto& sp : steps) {
        if (sp.py < minY) minY = sp.py; if (sp.py > maxY) maxY = sp.py;
        if (sp.pz < minZ) minZ = sp.pz; if (sp.pz > maxZ) maxZ = sp.pz;
    }
    if (steps.empty() || minY >= maxY) { minY = -2.0; maxY = 2.0; }
    if (minZ >= maxZ) { minZ = -2.0; maxZ = 2.0; }
    double padY = (maxY - minY) * 0.15 + 0.3;
    double padZ = (maxZ - minZ) * 0.15 + 0.3;
    minY -= padY; maxY += padY;
    minZ -= padZ; maxZ += padZ;
    double rangeY = maxY - minY;
    double rangeZ = maxZ - minZ;
    if (rangeY < 0.1) rangeY = 1.0;
    if (rangeZ < 0.1) rangeZ = 1.0;

    int plotW = canvasW - ML - MR;
    int plotH = height - 20;

    auto toX = [&](double v) -> int {
        return ML + (int)((v - minY) / rangeY * plotW);
    };
    auto toYp = [&](double v) -> int {
        return yOffset + height - 10 - (int)((v - minZ) / rangeZ * plotH);
    };

    // Grid: axes
    if (minY <= 0.0 && maxY >= 0.0) {
        int x0 = toX(0.0);
        cv::line(canvas, {x0, yOffset}, {x0, yOffset + height}, GRID_COLOR, 1);
    }
    if (minZ <= 0.0 && maxZ >= 0.0) {
        int y0 = toYp(0.0);
        cv::line(canvas, {ML, y0}, {canvasW - MR, y0}, GRID_COLOR, 1);
    }

    // Title
    cv::putText(canvas, "Step Start Points: localFwdY(X) vs pureZ(Y)  [colored by GT]",
                {ML + 5, yOffset + 15}, cv::FONT_HERSHEY_SIMPLEX, 0.33, TEXT_COLOR, 1);
    cv::putText(canvas, "fwdY ->", {canvasW - MR - 55, yOffset + height - 5},
                cv::FONT_HERSHEY_PLAIN, 0.75, TEXT_DIM, 1);
    cv::putText(canvas, "pureZ", {2, yOffset + height / 2},
                cv::FONT_HERSHEY_PLAIN, 0.75, TEXT_DIM, 1);

    // 스텝 시작점 그리기 — GT label 색상
    int gndCnt = 0, upCnt = 0, dnCnt = 0;
    for (const auto& sp : steps) {
        int x = toX(sp.py);
        int y = toYp(sp.pz);
        cv::Scalar col = floorLabelColor(sp.label);

        // 과거 스텝은 좀 작게, 최근 스텝은 크게
        bool isRecent = (sampleIdx - sp.idx) < 200;
        int radius = isRecent ? 4 : 2;
        double alpha = isRecent ? 1.0 : 0.5;
        cv::Scalar drawCol(col[0] * alpha, col[1] * alpha, col[2] * alpha);

        cv::circle(canvas, {x, y}, radius, drawCol, cv::FILLED);
        if (isRecent) {
            cv::circle(canvas, {x, y}, radius + 1, drawCol, 1);
        }

        if (sp.label == "ground") gndCnt++;
        else if (sp.label == "up") upCnt++;
        else dnCnt++;
    }

    // 가장 최근 스텝 강조 (노란 ring)
    if (!steps.empty()) {
        const auto& last = steps.back();
        int x = toX(last.py);
        int y = toYp(last.pz);
        cv::circle(canvas, {x, y}, 6, cv::Scalar(0, 255, 255), 2);
    }

    // Legend (우상단) — 카운트 포함
    {
        int lx = canvasW - MR - 10;
        int ly = yOffset + 15;
        char lbl[32];

        // down
        if (dnCnt > 0) {
            snprintf(lbl, sizeof(lbl), "dn:%d", dnCnt);
            int tw = (int)strlen(lbl) * 7 + 18;
            lx -= tw;
            cv::circle(canvas, {lx + 4, ly - 4}, 4, floorLabelColor("down"), cv::FILLED);
            cv::putText(canvas, lbl, {lx + 12, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_COLOR, 1);
        }
        // up
        if (upCnt > 0) {
            snprintf(lbl, sizeof(lbl), "up:%d", upCnt);
            int tw = (int)strlen(lbl) * 7 + 18;
            lx -= tw;
            cv::circle(canvas, {lx + 4, ly - 4}, 4, floorLabelColor("up"), cv::FILLED);
            cv::putText(canvas, lbl, {lx + 12, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_COLOR, 1);
        }
        // ground
        {
            snprintf(lbl, sizeof(lbl), "gnd:%d", gndCnt);
            int tw = (int)strlen(lbl) * 7 + 18;
            lx -= tw;
            cv::circle(canvas, {lx + 4, ly - 4}, 4, floorLabelColor("ground"), cv::FILLED);
            cv::putText(canvas, lbl, {lx + 12, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_COLOR, 1);
        }
    }

    cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
}

void showAnalysisWindow(const std::vector<FloorTraceData>& trace, int sampleIdx) {
    const int AW = 500;
    const int SCATTER_H = 250;
    const int STEP_START_H = 200;
    const int STATUS_H = 50;
    const int TOTAL_H = SCATTER_H + STEP_START_H + STATUS_H;

    cv::Mat canvas(TOTAL_H, AW, CV_8UC3, BG_COLOR);

    if (trace.empty() || sampleIdx < 0 || sampleIdx >= (int)trace.size()) {
        cv::imshow("Analysis", canvas);
        return;
    }

    // Top: 2D scatter (per-step trajectory)
    drawYZTrajectory(canvas, trace, sampleIdx, 0, SCATTER_H, AW);

    // Middle: Step Start Points scatter (누적, GT label 색상)
    drawStepStartScatter(canvas, trace, sampleIdx, SCATTER_H, STEP_START_H, AW);

    // Bottom: status
    {
        int y0 = SCATTER_H + STEP_START_H;
        const FloorTraceData& cur = trace[sampleIdx];
        char buf[128];

        snprintf(buf, sizeof(buf), "vYZr:%.2f  dmr:%.2f  zfr:%.2f  fqr:%.2f  |  sVarYZ:%.3f  dmag:%.2f",
                 cur.varYZRatio, cur.dmagRatio, cur.zfRatio, cur.freqRatio,
                 cur.recentStepVarYZ, cur.dmag);
        cv::putText(canvas, buf, {10, y0 + 20},
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1);

        snprintf(buf, sizeof(buf), "arm: %s   FSM: %s   conf: %.2f",
                 cur.armSpinFlag ? "YES" : "no", cur.fsmLabel.c_str(),
                 cur.stairConfidence);
        cv::putText(canvas, buf, {10, y0 + 38},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38,
                    floorLabelColor(cur.fsmLabel), 1);

        cv::rectangle(canvas, {0, y0}, {AW, TOTAL_H}, GRID_COLOR, 1);
    }

    cv::imshow("Analysis", canvas);
}

// ============================================================
//  Step Energy: rmsZ/freq chart (step-indexed)
//  rmsZdivFreq per step — stair indicator
// ============================================================
static void drawRmsZdivFreq(cv::Mat& canvas,
                             const std::vector<FloorTraceData>& trace,
                             int sampleIdx,
                             int yOffset, int height, int canvasW) {
    struct StepPt { double val; std::string label; int traceIdx; };
    std::vector<StepPt> steps;
    for (int i = 0; i <= sampleIdx && i < (int)trace.size(); ++i) {
        if (trace[i].isStep) {
            steps.push_back({trace[i].rmsZdivFreq, trace[i].floorLabel, i});
        }
    }

    const int SCROLL = 50;
    int startIdx = 0;
    if ((int)steps.size() > SCROLL) startIdx = (int)steps.size() - SCROLL;
    int visCount = (int)steps.size() - startIdx;

    int plotW = canvasW - ML - MR;
    int plotH = height - 30;
    int plotTop = yOffset + 20;

    // Auto Y range from visible steps (default 0.0~3.0)
    double minV = 1e9, maxV = -1e9;
    for (int i = startIdx; i < (int)steps.size(); ++i) {
        if (steps[i].val < minV) minV = steps[i].val;
        if (steps[i].val > maxV) maxV = steps[i].val;
    }
    if (steps.empty() || minV >= maxV) { minV = 0.0; maxV = 3.0; }
    double pad = (maxV - minV) * 0.12 + 0.1;
    if (minV > 0.0) minV = std::max(0.0, minV - pad);
    else minV -= pad;
    maxV += pad;
    double rangeV = maxV - minV;
    if (rangeV < 0.1) rangeV = 3.0;

    auto toY = [&](double v) -> int {
        return plotTop + plotH - (int)((v - minV) / rangeV * plotH);
    };
    auto toX = [&](int stepOff) -> int {
        int maxOff = std::max(visCount - 1, 1);
        return ML + (int)((double)stepOff / maxOff * plotW);
    };

    // Grid: horizontal lines (auto)
    {
        double step = rangeV / 4.0;
        for (int g = 0; g <= 4; ++g) {
            double v = minV + step * g;
            int y = toY(v);
            cv::line(canvas, {ML, y}, {canvasW - MR, y}, GRID_COLOR, 1);
            char lbl[8];
            snprintf(lbl, sizeof(lbl), "%.2f", v);
            cv::putText(canvas, lbl, {2, y + 4}, cv::FONT_HERSHEY_PLAIN, 0.7, TEXT_DIM, 1);
        }
    }

    // Threshold line at 0.90 baseline (orange dashed)
    if (minV <= 0.90 && maxV >= 0.90) {
        int yTh = toY(0.90);
        for (int x = ML; x < canvasW - MR; x += 8) {
            cv::line(canvas, {x, yTh}, {std::min(x + 4, canvasW - MR), yTh},
                     cv::Scalar(0, 140, 255), 1);
        }
        cv::putText(canvas, "0.90", {canvasW - MR - 30, yTh - 3},
                    cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0, 140, 255), 1);
    }

    // Title
    cv::putText(canvas, "rmsZ / freq  [per step]",
                {ML + 5, yOffset + 14}, cv::FONT_HERSHEY_SIMPLEX, 0.33, TEXT_COLOR, 1);

    if (steps.empty()) {
        cv::putText(canvas, "No steps yet", {canvasW / 2 - 40, yOffset + height / 2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1);
        cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
        return;
    }

    // 5-step rolling average
    std::vector<double> rollingAvg(steps.size(), 0.0);
    {
        double sum = 0.0;
        int win = 5;
        for (int i = 0; i < (int)steps.size(); ++i) {
            sum += steps[i].val;
            if (i >= win) sum -= steps[i - win].val;
            int cnt = std::min(i + 1, win);
            rollingAvg[i] = sum / cnt;
        }
    }

    // Draw rolling average line (thick white)
    {
        cv::Point prev(-1, -1);
        for (int i = startIdx; i < (int)steps.size(); ++i) {
            int off = i - startIdx;
            int x = toX(off);
            int y = toY(rollingAvg[i]);
            cv::Point pt(x, y);
            if (prev.x >= 0) {
                cv::line(canvas, prev, pt, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
            }
            prev = pt;
        }
    }

    // Draw individual step dots (GT color)
    int gndCnt = 0, upCnt = 0;
    for (int i = startIdx; i < (int)steps.size(); ++i) {
        int off = i - startIdx;
        int x = toX(off);
        int y = toY(steps[i].val);
        cv::Scalar col = floorLabelColor(steps[i].label);

        bool isRecent = (i >= (int)steps.size() - 5);
        int radius = isRecent ? 4 : 3;
        cv::circle(canvas, {x, y}, radius, col, cv::FILLED);

        if (steps[i].label == "ground") gndCnt++;
        else upCnt++;
    }

    // Highlight latest step (yellow ring)
    if (!steps.empty()) {
        int off = (int)steps.size() - 1 - startIdx;
        if (off >= 0) {
            int x = toX(off);
            int y = toY(steps.back().val);
            cv::circle(canvas, {x, y}, 6, cv::Scalar(0, 255, 255), 2);
        }
    }

    // Legend (top-right)
    {
        char lbl[32];
        int lx = canvasW - MR - 10;
        int ly = yOffset + 14;

        snprintf(lbl, sizeof(lbl), "up:%d", upCnt);
        int tw1 = (int)strlen(lbl) * 7 + 18;
        lx -= tw1;
        cv::circle(canvas, {lx + 4, ly - 4}, 3, floorLabelColor("up"), cv::FILLED);
        cv::putText(canvas, lbl, {lx + 10, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_COLOR, 1);

        snprintf(lbl, sizeof(lbl), "gnd:%d", gndCnt);
        int tw2 = (int)strlen(lbl) * 7 + 18;
        lx -= tw2;
        cv::circle(canvas, {lx + 4, ly - 4}, 3, floorLabelColor("ground"), cv::FILLED);
        cv::putText(canvas, lbl, {lx + 10, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_COLOR, 1);

        lx -= 60;
        cv::line(canvas, {lx, ly - 4}, {lx + 14, ly - 4}, cv::Scalar(220, 220, 220), 2);
        cv::putText(canvas, "avg5", {lx + 17, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_COLOR, 1);
    }

    cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
}

// ============================================================
//  Step Features: stairScore (envZ - envH) per step
// ============================================================
static void drawStairScore(cv::Mat& canvas,
                            const std::vector<FloorTraceData>& trace,
                            int sampleIdx,
                            int yOffset, int height, int canvasW) {
    struct StepPt { double val; std::string label; int traceIdx; };
    std::vector<StepPt> steps;
    for (int i = 0; i <= sampleIdx && i < (int)trace.size(); ++i) {
        if (trace[i].isStep) {
            steps.push_back({trace[i].stairScore, trace[i].floorLabel, i});
        }
    }

    const int SCROLL = 50;
    int startIdx = 0;
    if ((int)steps.size() > SCROLL) startIdx = (int)steps.size() - SCROLL;
    int visCount = (int)steps.size() - startIdx;

    int plotW = canvasW - ML - MR;
    int plotH = height - 30;
    int plotTop = yOffset + 20;

    // Auto Y range (default -0.5 ~ 0.5)
    double minV = 1e9, maxV = -1e9;
    for (int i = startIdx; i < (int)steps.size(); ++i) {
        if (steps[i].val < minV) minV = steps[i].val;
        if (steps[i].val > maxV) maxV = steps[i].val;
    }
    if (steps.empty() || minV >= maxV) { minV = -0.5; maxV = 0.5; }
    double pad = (maxV - minV) * 0.12 + 0.05;
    minV -= pad; maxV += pad;
    double rangeV = maxV - minV;
    if (rangeV < 0.1) rangeV = 1.0;

    auto toY = [&](double v) -> int {
        return plotTop + plotH - (int)((v - minV) / rangeV * plotH);
    };
    auto toX = [&](int stepOff) -> int {
        int maxOff = std::max(visCount - 1, 1);
        return ML + (int)((double)stepOff / maxOff * plotW);
    };

    // Grid
    {
        double step = rangeV / 4.0;
        for (int g = 0; g <= 4; ++g) {
            double v = minV + step * g;
            int y = toY(v);
            cv::line(canvas, {ML, y}, {canvasW - MR, y}, GRID_COLOR, 1);
            char lbl[8];
            snprintf(lbl, sizeof(lbl), "%.2f", v);
            cv::putText(canvas, lbl, {2, y + 4}, cv::FONT_HERSHEY_PLAIN, 0.7, TEXT_DIM, 1);
        }
    }

    // Zero line
    if (minV <= 0.0 && maxV >= 0.0) {
        int y0 = toY(0.0);
        cv::line(canvas, {ML, y0}, {canvasW - MR, y0}, cv::Scalar(80, 80, 80), 1);
    }

    // Threshold line at 0.15 (orange dashed)
    if (minV <= 0.15 && maxV >= 0.15) {
        int yTh = toY(0.15);
        for (int x = ML; x < canvasW - MR; x += 8) {
            cv::line(canvas, {x, yTh}, {std::min(x + 4, canvasW - MR), yTh},
                     cv::Scalar(0, 140, 255), 1);
        }
        cv::putText(canvas, "0.15", {canvasW - MR - 30, yTh - 3},
                    cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0, 140, 255), 1);
    }

    // Title
    cv::putText(canvas, "stairScore (envZ - envH)  [per step]",
                {ML + 5, yOffset + 14}, cv::FONT_HERSHEY_SIMPLEX, 0.33, TEXT_COLOR, 1);

    if (steps.empty()) {
        cv::putText(canvas, "No steps yet", {canvasW / 2 - 40, yOffset + height / 2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1);
        cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
        return;
    }

    // Draw connecting lines (dim) + dots (GT color)
    cv::Point prev(-1, -1);
    for (int i = startIdx; i < (int)steps.size(); ++i) {
        int off = i - startIdx;
        int x = toX(off);
        int y = toY(steps[i].val);
        cv::Scalar col = floorLabelColor(steps[i].label);

        cv::Point pt(x, y);
        if (prev.x >= 0) {
            cv::line(canvas, prev, pt, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
        }
        prev = pt;

        bool isRecent = (i >= (int)steps.size() - 5);
        int radius = isRecent ? 4 : 3;
        cv::circle(canvas, {x, y}, radius, col, cv::FILLED);
    }

    // Highlight latest step (yellow ring)
    if (!steps.empty()) {
        int off = (int)steps.size() - 1 - startIdx;
        if (off >= 0) {
            int x = toX(off);
            int y = toY(steps.back().val);
            cv::circle(canvas, {x, y}, 6, cv::Scalar(0, 255, 255), 2);
        }
    }

    cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
}

// ============================================================
//  Generic per-step chart (reusable for any step-indexed metric)
//  - Extracts value via getter, draws dots (GT-colored) + connecting line
//  - Optional threshold line (orange dashed)
//  - Auto Y range with fallback defaults
// ============================================================
static void drawStepMetric(cv::Mat& canvas,
                            const std::vector<FloorTraceData>& trace,
                            int sampleIdx,
                            int yOffset, int height, int canvasW,
                            const char* title,
                            double (*getValue)(const FloorTraceData&),
                            double defaultMin, double defaultMax,
                            double threshVal = 1e30,
                            bool threshAbove = true) {
    struct StepPt { double val; std::string label; int traceIdx; };
    std::vector<StepPt> steps;
    for (int i = 0; i <= sampleIdx && i < (int)trace.size(); ++i) {
        if (trace[i].isStep) {
            steps.push_back({getValue(trace[i]), trace[i].floorLabel, i});
        }
    }

    const int SCROLL = 50;
    int startIdx = 0;
    if ((int)steps.size() > SCROLL) startIdx = (int)steps.size() - SCROLL;
    int visCount = (int)steps.size() - startIdx;

    int plotW = canvasW - ML - MR;
    int plotH = height - 30;
    int plotTop = yOffset + 20;

    // Auto Y range
    double minV = 1e9, maxV = -1e9;
    for (int i = startIdx; i < (int)steps.size(); ++i) {
        if (steps[i].val < minV) minV = steps[i].val;
        if (steps[i].val > maxV) maxV = steps[i].val;
    }
    if (steps.empty() || minV >= maxV) { minV = defaultMin; maxV = defaultMax; }
    double pad = (maxV - minV) * 0.12 + 0.05;
    minV -= pad; maxV += pad;
    // Include threshold in range if needed
    if (threshVal < 1e20) {
        if (threshVal < minV) minV = threshVal - pad;
        if (threshVal > maxV) maxV = threshVal + pad;
    }
    double rangeV = maxV - minV;
    if (rangeV < 0.1) rangeV = 1.0;

    auto toY = [&](double v) -> int {
        return plotTop + plotH - (int)((v - minV) / rangeV * plotH);
    };
    auto toX = [&](int stepOff) -> int {
        int maxOff = std::max(visCount - 1, 1);
        return ML + (int)((double)stepOff / maxOff * plotW);
    };

    // Grid
    {
        double step = rangeV / 4.0;
        for (int g = 0; g <= 4; ++g) {
            double v = minV + step * g;
            int y = toY(v);
            cv::line(canvas, {ML, y}, {canvasW - MR, y}, GRID_COLOR, 1);
            char lbl[16];
            snprintf(lbl, sizeof(lbl), "%.2f", v);
            cv::putText(canvas, lbl, {2, y + 4}, cv::FONT_HERSHEY_PLAIN, 0.7, TEXT_DIM, 1);
        }
    }

    // Zero line
    if (minV <= 0.0 && maxV >= 0.0) {
        int y0 = toY(0.0);
        cv::line(canvas, {ML, y0}, {canvasW - MR, y0}, cv::Scalar(80, 80, 80), 1);
    }

    // Threshold line (orange dashed)
    if (threshVal < 1e20 && minV <= threshVal && maxV >= threshVal) {
        int yTh = toY(threshVal);
        for (int x = ML; x < canvasW - MR; x += 8) {
            cv::line(canvas, {x, yTh}, {std::min(x + 4, canvasW - MR), yTh},
                     cv::Scalar(0, 140, 255), 1);
        }
        char lbl[16];
        snprintf(lbl, sizeof(lbl), "%.2f", threshVal);
        cv::putText(canvas, lbl, {canvasW - MR - 35, yTh - 3},
                    cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0, 140, 255), 1);
    }

    // Title
    cv::putText(canvas, title,
                {ML + 5, yOffset + 14}, cv::FONT_HERSHEY_SIMPLEX, 0.33, TEXT_COLOR, 1);

    if (steps.empty()) {
        cv::putText(canvas, "No steps yet", {canvasW / 2 - 40, yOffset + height / 2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1);
        cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
        return;
    }

    // Draw connecting lines (dim) + dots (GT color)
    cv::Point prev(-1, -1);
    for (int i = startIdx; i < (int)steps.size(); ++i) {
        int off = i - startIdx;
        int x = toX(off);
        int y = toY(steps[i].val);
        cv::Scalar col = floorLabelColor(steps[i].label);

        cv::Point pt(x, y);
        if (prev.x >= 0)
            cv::line(canvas, prev, pt, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
        prev = pt;

        bool isRecent = (i >= (int)steps.size() - 5);
        cv::circle(canvas, {x, y}, isRecent ? 4 : 3, col, cv::FILLED);
    }

    // Highlight latest step (yellow ring)
    if (!steps.empty()) {
        int off = (int)steps.size() - 1 - startIdx;
        if (off >= 0) {
            int x = toX(off);
            int y = toY(steps.back().val);
            cv::circle(canvas, {x, y}, 6, cv::Scalar(0, 255, 255), 2);
        }
    }

    cv::rectangle(canvas, {ML, yOffset}, {canvasW - MR, yOffset + height}, GRID_COLOR, 1);
}

// Step metric getters
static double getStepTotalScore(const FloorTraceData& d)  { return d.totalStairScore; }
static double getStepDirR1(const FloorTraceData& d)       { return d.dirR1; }

// ============================================================
//  Window 3: Step Features — Weighted Scoring 전체 시각화
//
//  Row 0: stairScore (envZ - envH)       — 물리 궤적 (40점)
//  Row 1: varYZRatio (궤적 집중도)        — 30점
//  Row 2: dirR1 (방향 일관성)             — 20점
//  Row 3: totalStairScore (합산 점수)     — 커트라인 60
//  Row 4: Status bar (점수 분해 + FSM)
// ============================================================
void showStepEnergyWindow(const std::vector<FloorTraceData>& trace, int sampleIdx) {
    const int SW = 700;
    const int ROW_H = 130;
    const int ROWS = 4;
    const int BAR_H = 60;
    const int TOTAL_H = ROW_H * ROWS + BAR_H;

    cv::Mat canvas(TOTAL_H, SW, CV_8UC3, BG_COLOR);

    if (trace.empty() || sampleIdx < 0 || sampleIdx >= (int)trace.size()) {
        cv::imshow("Step Features", canvas);
        return;
    }

    int y0 = 0;

    // Row 0: stairScore (envZ - envH)  threshold 0.15
    drawStepMetric(canvas, trace, sampleIdx, y0, ROW_H, SW,
                   "stairScore (envZ-envH) [40pt if >0.15]",
                   getStairScore, -0.5, 0.5, 0.15);
    y0 += ROW_H;

    // Row 1: varYZRatio  threshold 0.85 (below = stair)
    drawStepMetric(canvas, trace, sampleIdx, y0, ROW_H, SW,
                   "varYZRatio (traj concentration) [30pt if <0.85]",
                   getVarYZRatio, 0.0, 2.0, 0.85);
    y0 += ROW_H;

    // Row 2: dirR1  threshold 0.85 (above = stair)
    drawStepMetric(canvas, trace, sampleIdx, y0, ROW_H, SW,
                   "dirR1 (direction consistency) [20pt if >0.85]",
                   getStepDirR1, 0.0, 1.0, 0.85);
    y0 += ROW_H;

    // Row 3: totalStairScore  threshold 60
    drawStepMetric(canvas, trace, sampleIdx, y0, ROW_H, SW,
                   "Total Score [stair if >=60]",
                   getStepTotalScore, -30.0, 100.0, 60.0);
    y0 += ROW_H;

    // Row 4: Status bar — 점수 분해 + FSM
    {
        const FloorTraceData& cur = trace[sampleIdx];

        // 각 항목 점수 재계산 (표시용)
        int ptEnv = 0;
        if (cur.stairScore > 0.15)      ptEnv = 40;
        else if (cur.stairScore > 0.05) ptEnv = 20;

        int ptVar = 0;
        if (cur.varYZRatio < 0.6)       ptVar = 30;
        else if (cur.varYZRatio < 0.85) ptVar = 15;

        int ptDir = 0;
        if (cur.dirR1 > 0.90)      ptDir = 20;
        else if (cur.dirR1 > 0.80) ptDir = 10;

        int ptH = (cur.stepVarH < 1.5) ? 10 : 0;

        int penalty = 0;
        if (cur.stepVarH > 2.5)    penalty -= 30;
        if (cur.rmsZdivFreq < 0.8) penalty -= 20;

        char buf[256];
        snprintf(buf, sizeof(buf),
                 "env:%d  varYZ:%d  dirR1:%d  varH:%d  penalty:%d  => TOTAL:%.0f/100",
                 ptEnv, ptVar, ptDir, ptH, penalty, cur.totalStairScore);
        cv::putText(canvas, buf, {10, y0 + 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1);

        snprintf(buf, sizeof(buf),
                 "stairScore=%.3f  varYZr=%.2f  dirR1=%.2f  stepVarH=%.2f  rmsZ/f=%.2f",
                 cur.stairScore, cur.varYZRatio, cur.dirR1, cur.stepVarH, cur.rmsZdivFreq);
        cv::putText(canvas, buf, {10, y0 + 34},
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1);

        snprintf(buf, sizeof(buf), "FSM: %s   GT: %s   step#%d",
                 cur.fsmLabel.c_str(), cur.floorLabel.c_str(), cur.totalSteps);
        cv::putText(canvas, buf, {10, y0 + 52},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38,
                    floorLabelColor(cur.fsmLabel), 1);

        // FSM color dot
        cv::circle(canvas, {SW - 60, y0 + 30}, 12,
                   floorLabelColor(cur.fsmLabel), cv::FILLED);
        cv::putText(canvas, cur.fsmLabel.c_str(), {SW - 55, y0 + 52},
                    cv::FONT_HERSHEY_PLAIN, 0.8, TEXT_COLOR, 1);

        cv::rectangle(canvas, {0, y0}, {SW, TOTAL_H}, GRID_COLOR, 1);
    }

    cv::imshow("Step Features", canvas);
}
