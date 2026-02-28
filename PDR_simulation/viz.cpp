#include "viz.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// World Map canvas offset (for mouse coordinate correction)
static int s_worldMapOffX = 0;
static int s_worldMapOffY = 0;

cv::Point getWorldMapCanvasOffset() {
    return {s_worldMapOffX, s_worldMapOffY};
}

// ============================================================
//  1. 분석용 PDR Path (좌측 화면 - Raw Data 분석용)
// ============================================================
static cv::Mat drawPDRPath(const std::vector<StepResult>& steps,
                           const std::vector<StepResult>& oldSteps,
                           int upToStep, int upToOldStep, int targetH) {
    int targetW = (int)(targetH * 1.2);
    cv::Mat canvas(targetH, targetW, CV_8UC3, cv::Scalar(255, 255, 255));

    if ((upToStep <= 0 || steps.empty()) && (upToOldStep <= 0 || oldSteps.empty())) {
        cv::putText(canvas, "No Data", {20, targetH/2}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,0), 2);
        return canvas;
    }

    // Collect all points for bounding box
    double minX = 0.0, maxX = 0.0, minY = 0.0, maxY = 0.0;

    std::vector<cv::Point2d> newPts;
    newPts.push_back({0.0, 0.0});
    for (int i = 0; i < upToStep && i < (int)steps.size(); ++i) {
        newPts.push_back({steps[i].x, steps[i].y});
        minX = std::min(minX, steps[i].x); maxX = std::max(maxX, steps[i].x);
        minY = std::min(minY, steps[i].y); maxY = std::max(maxY, steps[i].y);
    }

    std::vector<cv::Point2d> oldPts;
    oldPts.push_back({0.0, 0.0});
    for (int i = 0; i < upToOldStep && i < (int)oldSteps.size(); ++i) {
        oldPts.push_back({oldSteps[i].x, oldSteps[i].y});
        minX = std::min(minX, oldSteps[i].x); maxX = std::max(maxX, oldSteps[i].x);
        minY = std::min(minY, oldSteps[i].y); maxY = std::max(maxY, oldSteps[i].y);
    }

    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    if (rangeX < 0.1) rangeX = 1.0;
    if (rangeY < 0.1) rangeY = 1.0;

    int padding = 50;
    int availW = targetW - 2 * padding;
    int availH = targetH - 2 * padding;

    double scaleX = availW / rangeX;
    double scaleY = availH / rangeY;
    double scale = std::min(scaleX, scaleY);

    double contentW = rangeX * scale;
    double contentH = rangeY * scale;
    int offX = padding + (int)(availW - contentW) / 2;
    int offY = padding + (int)(availH - contentH) / 2;

    auto toScreen = [&](double wx, double wy) -> cv::Point {
        int sx = (int)((wx - minX) * scale + offX);
        int sy = (int)((maxY - wy) * scale + offY);
        return {sx, sy};
    };

    // Draw Old path (blue, behind)
    for (size_t i = 1; i < oldPts.size(); ++i) {
        cv::line(canvas, toScreen(oldPts[i-1].x, oldPts[i-1].y),
                 toScreen(oldPts[i].x, oldPts[i].y),
                 cv::Scalar(255, 150, 50), 2, cv::LINE_AA); // Blue (BGR)
    }

    // Draw New path (red, on top)
    for (size_t i = 1; i < newPts.size(); ++i) {
        cv::line(canvas, toScreen(newPts[i-1].x, newPts[i-1].y),
                 toScreen(newPts[i].x, newPts[i].y),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA); // Red (BGR)
    }

    // Start point
    cv::circle(canvas, toScreen(0.0, 0.0), 6, cv::Scalar(0, 200, 0), cv::FILLED);

    // Current positions
    if (newPts.size() > 1) {
        cv::circle(canvas, toScreen(newPts.back().x, newPts.back().y), 7, cv::Scalar(0, 0, 255), cv::FILLED);
    }
    if (oldPts.size() > 1) {
        cv::circle(canvas, toScreen(oldPts.back().x, oldPts.back().y), 7, cv::Scalar(255, 150, 50), cv::FILLED);
    }

    // Title & Legend
    std::string title = "Local Path (New:" + std::to_string(upToStep) + " Old:" + std::to_string(upToOldStep) + " steps)";
    cv::putText(canvas, title, {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 50, 50), 2);

    // Legend
    int ly = targetH - 35;
    cv::line(canvas, {15, ly}, {40, ly}, cv::Scalar(0, 0, 255), 3);
    cv::putText(canvas, "New", {45, ly + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);
    cv::line(canvas, {90, ly}, {115, ly}, cv::Scalar(255, 150, 50), 3);
    cv::putText(canvas, "Old", {120, ly + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 150, 50), 1);

    return canvas;
}

// ============================================================
//  2. 실제 지도 뷰 (지도 변환 + PDR 회전 적용)
// ============================================================
static void drawWorldMapWindow(const std::vector<StepResult>& steps,
                               const std::vector<StepResult>& oldSteps,
                               int upToStep,
                               int upToOldStep,
                               const cv::Mat& rawMapImage,
                               cv::Point initialStartPos,
                               double scale,
                               const MapViewState& viewState) {

    // 1. 지도 변환 (회전 -> 반전 순서)
    cv::Mat mapImg = rawMapImage.clone();

    // 회전 (90도 단위)
    if (viewState.rotate90 == 1) cv::rotate(mapImg, mapImg, cv::ROTATE_90_CLOCKWISE);
    else if (viewState.rotate90 == 2) cv::rotate(mapImg, mapImg, cv::ROTATE_180);
    else if (viewState.rotate90 == 3) cv::rotate(mapImg, mapImg, cv::ROTATE_90_COUNTERCLOCKWISE);

    // 반전 (Flip)
    if (viewState.flipX && viewState.flipY) cv::flip(mapImg, mapImg, -1);
    else if (viewState.flipX) cv::flip(mapImg, mapImg, 1);
    else if (viewState.flipY) cv::flip(mapImg, mapImg, 0);

    // 2. 시작점 (맵 이미지 기준 좌표)
    cv::Point baseStart = initialStartPos + cv::Point(viewState.moveX, viewState.moveY);

    // 3. 경로 바운딩 박스 계산 (맵 이미지 좌표계)
    double rad = viewState.pdrRotateDeg * M_PI / 180.0;
    double cosT = cos(rad);
    double sinT = sin(rad);

    auto rawToMap = [&](double px, double py) -> cv::Point {
        double rx = px * cosT - py * sinT;
        double ry = px * sinT + py * cosT;
        int mx = baseStart.x + (int)(rx * scale);
        int my = baseStart.y - (int)(ry * scale);
        return {mx, my};
    };

    int minPx = 0, minPy = 0;
    int maxPx = mapImg.cols, maxPy = mapImg.rows;

    auto expandBounds = [&](cv::Point p) {
        minPx = std::min(minPx, p.x);
        minPy = std::min(minPy, p.y);
        maxPx = std::max(maxPx, p.x);
        maxPy = std::max(maxPy, p.y);
    };

    if (viewState.showNewPath) {
        for (int i = 0; i < upToStep && i < (int)steps.size(); ++i)
            expandBounds(rawToMap(steps[i].x, steps[i].y));
    }
    if (viewState.showOldPath) {
        for (int i = 0; i < upToOldStep && i < (int)oldSteps.size(); ++i)
            expandBounds(rawToMap(oldSteps[i].x, oldSteps[i].y));
    }

    // 4. 확장 캔버스 생성
    const int PAD = 40;
    minPx -= PAD; minPy -= PAD;
    maxPx += PAD; maxPy += PAD;

    int canvasW = maxPx - minPx;
    int canvasH = maxPy - minPy;
    int offX = -minPx;  // 맵 이미지 좌표 → 캔버스 좌표 오프셋
    int offY = -minPy;

    cv::Mat canvas(canvasH, canvasW, CV_8UC3, cv::Scalar(240, 240, 240));
    mapImg.copyTo(canvas(cv::Rect(offX, offY, mapImg.cols, mapImg.rows)));

    // Cache offset for mouse callback coordinate correction
    s_worldMapOffX = offX;
    s_worldMapOffY = offY;

    // 5. 오프셋 적용된 시작점 & 좌표 변환
    cv::Point currentStartPos = baseStart + cv::Point(offX, offY);

    auto toMap = [&](double px, double py) -> cv::Point {
        double rx = px * cosT - py * sinT;
        double ry = px * sinT + py * cosT;
        int mx = currentStartPos.x + (int)(rx * scale);
        int my = currentStartPos.y - (int)(ry * scale);
        return {mx, my};
    };

    // 시작점 표시
    cv::circle(canvas, currentStartPos, 5, cv::Scalar(0, 200, 0), cv::FILLED);
    cv::circle(canvas, currentStartPos, 10, cv::Scalar(0, 150, 0), 2);
    cv::putText(canvas, "Start", currentStartPos + cv::Point(12, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 100, 0), 2);

    // 5a. Old 경로 (파란색, 뒤에 그림)
    if (viewState.showOldPath && upToOldStep > 0 && !oldSteps.empty()) {
        cv::Point prevPt = currentStartPos;
        for (int i = 0; i < upToOldStep && i < (int)oldSteps.size(); ++i) {
            cv::Point curPt = toMap(oldSteps[i].x, oldSteps[i].y);
            cv::line(canvas, prevPt, curPt, cv::Scalar(255, 150, 50), 2, cv::LINE_AA);
            prevPt = curPt;
        }
        cv::circle(canvas, prevPt, 5, cv::Scalar(255, 150, 50), cv::FILLED);
    }

    // 5b. New 경로 (빨간색, 위에 그림)
    if (viewState.showNewPath && upToStep > 0 && !steps.empty()) {
        cv::Point prevPt = currentStartPos;
        for (int i = 0; i < upToStep && i < (int)steps.size(); ++i) {
            cv::Point curPt = toMap(steps[i].x, steps[i].y);
            cv::line(canvas, prevPt, curPt, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            prevPt = curPt;
        }

        cv::circle(canvas, prevPt, 6, cv::Scalar(255, 0, 0), cv::FILLED);
        cv::circle(canvas, prevPt, 10, cv::Scalar(255, 0, 0), 2);

        char buf[64];
        snprintf(buf, sizeof(buf), "(%.1fm, %.1fm)", steps[std::min(upToStep-1, (int)steps.size()-1)].x, steps[std::min(upToStep-1, (int)steps.size()-1)].y);
        cv::putText(canvas, buf, prevPt + cv::Point(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
    }

    // --- Toggle Buttons (top-left) ---
    // Button geometry: [New ●] [Old ●]
    const int btnW = 80, btnH = 28, btnY = 8, btnGap = 6;
    int btnX0 = 8;

    // New button
    cv::Rect newBtnRect(btnX0, btnY, btnW, btnH);
    cv::Scalar newBtnBg = viewState.showNewPath ? cv::Scalar(0, 0, 180) : cv::Scalar(80, 80, 80);
    cv::rectangle(canvas, newBtnRect, newBtnBg, cv::FILLED);
    cv::rectangle(canvas, newBtnRect, cv::Scalar(200, 200, 200), 1);
    cv::circle(canvas, {btnX0 + 16, btnY + btnH/2}, 5,
               viewState.showNewPath ? cv::Scalar(100, 100, 255) : cv::Scalar(50, 50, 50), cv::FILLED);
    cv::putText(canvas, "New", {btnX0 + 28, btnY + btnH/2 + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(255, 255, 255), 1);

    // Old button
    int btnX1 = btnX0 + btnW + btnGap;
    cv::Rect oldBtnRect(btnX1, btnY, btnW, btnH);
    cv::Scalar oldBtnBg = viewState.showOldPath ? cv::Scalar(180, 100, 30) : cv::Scalar(80, 80, 80);
    cv::rectangle(canvas, oldBtnRect, oldBtnBg, cv::FILLED);
    cv::rectangle(canvas, oldBtnRect, cv::Scalar(200, 200, 200), 1);
    cv::circle(canvas, {btnX1 + 16, btnY + btnH/2}, 5,
               viewState.showOldPath ? cv::Scalar(255, 180, 80) : cv::Scalar(50, 50, 50), cv::FILLED);
    cv::putText(canvas, "Old", {btnX1 + 28, btnY + btnH/2 + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(255, 255, 255), 1);

    // Legend (bottom)
    int ly = canvas.rows - 30;
    if (viewState.showNewPath) {
        cv::line(canvas, {10, ly}, {35, ly}, cv::Scalar(0, 0, 255), 3);
        cv::putText(canvas, "New", {40, ly + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);
    }
    if (viewState.showOldPath) {
        cv::line(canvas, {80, ly}, {105, ly}, cv::Scalar(255, 150, 50), 3);
        cv::putText(canvas, "Old", {110, ly + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 150, 50), 1);
    }

    // 5. 상태 정보 (간략 표시)
    int infoY = 30;
    auto drawInfo = [&](const std::string& text) {
        cv::putText(canvas, text, {canvas.cols - 200, infoY}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
        infoY += 18;
    };

    drawInfo("Path Rot: " + std::to_string((int)viewState.pdrRotateDeg) + " deg");
    drawInfo("Map Rot: " + std::to_string(viewState.rotate90 * 90) + " deg");
    if(viewState.flipX) drawInfo("Flip X: ON");
    if(viewState.flipY) drawInfo("Flip Y: ON");

    // 6. 축척 바 표시
    int barMeters = 10;
    int barLenPx = (int)(barMeters * scale);

    cv::Point barStart(canvas.cols - 150, canvas.rows - 50);
    cv::line(canvas, barStart, barStart + cv::Point(barLenPx, 0), cv::Scalar(0,0,0), 3);
    cv::putText(canvas, "10m", barStart + cv::Point(barLenPx/2 - 15, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 2);

    cv::imshow("World Map View", canvas);
}

// ============================================================
//  3. 파형 및 정보창 (상세 정보 + 평균값 추가됨)
// ============================================================
static const int SCROLL_WIN = 400;

static cv::Mat drawAccelWaveform(const std::vector<SampleData>& trace,
                                  int sampleIdx, int upToStep,
                                  const std::string& label,
                                  cv::Scalar markerColor,
                                  bool useOldSteps,
                                  const WaveformClick* click = nullptr) {
    const int W = 800, H = 220;
    const int ML = 40, TOP = 25, BOT = H - 15;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(30, 30, 30));

    double lo = -5.0, hi = 5.0;
    for (double v = -4; v <= 4; v += 2.0) {
        int py = BOT - (int)((v - lo) / (hi - lo) * (BOT - TOP));
        cv::line(img, {ML, py}, {W - 5, py}, cv::Scalar(55, 55, 55), 1);
    }
    int zy = BOT - (int)((0 - lo) / (hi - lo) * (BOT - TOP));
    cv::line(img, {ML, zy}, {W - 5, zy}, cv::Scalar(90, 90, 90), 1);

    int start = std::max(0, sampleIdx - SCROLL_WIN + 1);
    int count = sampleIdx - start + 1;
    int xOff  = SCROLL_WIN - count;
    double ppx = (double)(W - ML - 5) / SCROLL_WIN;

    auto toY = [&](double v) { return BOT - (int)((v - lo) / (hi - lo) * (BOT - TOP)); };
    auto toX = [&](int localI) { return ML + (int)((xOff + localI) * ppx); };

    for (int i = 1; i < count; ++i) {
        int si = start + i;
        cv::line(img, {toX(i - 1), toY(trace[si-1].globalZ)}, {toX(i), toY(trace[si].globalZ)}, cv::Scalar(255, 160, 40), 1, cv::LINE_AA);
        cv::line(img, {toX(i - 1), toY(trace[si-1].filteredZ)}, {toX(i), toY(trace[si].filteredZ)}, cv::Scalar(40, 140, 255), 2, cv::LINE_AA);
    }

    // --- Dynamic Threshold Lines ---
    if (count > 1) {
        // Helper lambda: compute old-style thresholds from recent filteredZ
        // Old algorithm: always adaptive, dynamicZDiff = max(1.2, effectiveRange*0.45)
        const int AMP_WIN = 25;
        auto computeOldThresh = [&](int si) -> std::array<double,3> {
            int wStart = std::max(0, si - AMP_WIN + 1);
            double mn = 1e9, mx = -1e9;
            for (int j = wStart; j <= si; ++j) {
                double v = trace[j].filteredZ;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            double longRange = mx - mn;
            double shortRange = longRange;
            if (si - wStart + 1 > 10) {
                int sStart = si - 9;
                double smn = 1e9, smx = -1e9;
                for (int j = sStart; j <= si; ++j) {
                    double v = trace[j].filteredZ;
                    if (v < smn) smn = v;
                    if (v > smx) smx = v;
                }
                shortRange = smx - smn;
            }
            double effRange = std::min(longRange, shortRange);
            double zDiff = std::max(1.2, effRange * 0.45);
            return {zDiff, zDiff * 0.4, -(zDiff * 0.4)};
        };

        for (int i = 1; i < count; ++i) {
            int si = start + i;
            double up0, up1, dn0, dn1;
            if (!useOldSteps) {
                // New: use actual cached values
                up0 = trace[si-1].dynamicUpThresh;
                up1 = trace[si].dynamicUpThresh;
                dn0 = trace[si-1].dynamicDownThresh;
                dn1 = trace[si].dynamicDownThresh;
            } else {
                // Old: recompute from trace
                auto t0 = computeOldThresh(si - 1);
                auto t1 = computeOldThresh(si);
                up0 = t0[1]; up1 = t1[1];
                dn0 = t0[2]; dn1 = t1[2];
            }
            int x0 = toX(i - 1), x1 = toX(i);
            if (i % 3 != 0) {
                cv::line(img, {x0, toY(up0)}, {x1, toY(up1)},
                         cv::Scalar(180, 0, 180), 1, cv::LINE_AA);
                cv::line(img, {x0, toY(dn0)}, {x1, toY(dn1)},
                         cv::Scalar(180, 0, 180), 1, cv::LINE_AA);
            }
        }

        // Current threshold labels
        double curZDiff, curUp, curDown;
        if (!useOldSteps) {
            curZDiff = trace[sampleIdx].dynamicZDiff;
            curUp    = trace[sampleIdx].dynamicUpThresh;
            curDown  = trace[sampleIdx].dynamicDownThresh;
        } else {
            auto t = computeOldThresh(sampleIdx);
            curZDiff = t[0]; curUp = t[1]; curDown = t[2];
        }
        char dzBuf[64];
        snprintf(dzBuf, sizeof(dzBuf), "ZDiff=%.2f", curZDiff);
        int labelY = toY(curUp) - 4;
        if (labelY < TOP + 10) labelY = toY(curUp) + 14;
        cv::putText(img, dzBuf, {ML + 5, labelY},
                    cv::FONT_HERSHEY_PLAIN, 0.75, cv::Scalar(200, 80, 200), 1);

        snprintf(dzBuf, sizeof(dzBuf), "Up=%.2f", curUp);
        cv::putText(img, dzBuf, {ML + 90, labelY},
                    cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(180, 0, 180), 1);
        snprintf(dzBuf, sizeof(dzBuf), "Dn=%.2f", curDown);
        int dnLabelY = toY(curDown) + 14;
        if (dnLabelY > BOT - 4) dnLabelY = toY(curDown) - 4;
        cv::putText(img, dzBuf, {ML + 90, dnLabelY},
                    cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(180, 0, 180), 1);
    }

    // Step Markers
    for (int i = 0; i < count; ++i) {
        int si = start + i;
        if (useOldSteps) {
            if (!trace[si].isOldStep || trace[si].oldStepIdx < 0) continue;
            if (trace[si].oldStepIdx >= upToStep) continue;
        } else {
            if (!trace[si].isStep || trace[si].stepIdx < 0) continue;
            if (trace[si].stepIdx >= upToStep) continue;
        }
        int px = toX(i);
        cv::line(img, {px, TOP}, {px, BOT}, markerColor, 1);
    }

    // --- Real-time Peak/Valley Tracking ---
    {
        // Search backward from sampleIdx to find last confirmed step
        int lastStepSample = start;
        for (int si = sampleIdx - 1; si >= 0; --si) {
            bool isStepHere = useOldSteps
                ? (trace[si].isOldStep && trace[si].oldStepIdx >= 0 && trace[si].oldStepIdx < upToStep)
                : (trace[si].isStep && trace[si].stepIdx >= 0 && trace[si].stepIdx < upToStep);
            if (isStepHere) {
                lastStepSample = si;
                break;
            }
        }

        // Track running peak/valley of filteredZ from lastStepSample to sampleIdx
        double runPeak = -1e9, runValley = 1e9;
        int peakSample = lastStepSample, valleySample = lastStepSample;
        for (int si = lastStepSample; si <= sampleIdx; ++si) {
            double fz = trace[si].filteredZ;
            if (fz > runPeak)   { runPeak = fz;   peakSample = si; }
            if (fz < runValley) { runValley = fz;  valleySample = si; }
        }

        if (runPeak > -1e8 && runValley < 1e8) {
            int peakY   = toY(runPeak);
            int valleyY = toY(runValley);

            // Green dashed line at peak
            for (int dx = ML; dx < W - 5; dx += 8)
                cv::line(img, {dx, peakY}, {std::min(dx + 4, W - 5), peakY},
                         cv::Scalar(0, 200, 0), 1);
            // Yellow dashed line at valley
            for (int dx = ML; dx < W - 5; dx += 8)
                cv::line(img, {dx, valleyY}, {std::min(dx + 4, W - 5), valleyY},
                         cv::Scalar(0, 200, 200), 1);

            // Filled circle markers at peak/valley sample positions
            if (peakSample >= start && peakSample <= sampleIdx) {
                int pxP = toX(peakSample - start);
                cv::circle(img, {pxP, peakY}, 4, cv::Scalar(0, 255, 0), cv::FILLED);
            }
            if (valleySample >= start && valleySample <= sampleIdx) {
                int pxV = toX(valleySample - start);
                cv::circle(img, {pxV, valleyY}, 4, cv::Scalar(0, 255, 255), cv::FILLED);
            }

            // Amplitude bracket on right edge
            double ampZ = runPeak - runValley;
            int brX = W - 8;
            cv::line(img, {brX - 3, peakY}, {brX + 3, peakY}, cv::Scalar(160, 160, 160), 1);
            cv::line(img, {brX, peakY}, {brX, valleyY}, cv::Scalar(160, 160, 160), 1);
            cv::line(img, {brX - 3, valleyY}, {brX + 3, valleyY}, cv::Scalar(160, 160, 160), 1);

            // P / V / A labels near right margin
            char pvBuf[32];
            snprintf(pvBuf, sizeof(pvBuf), "P=%.2f", runPeak);
            cv::putText(img, pvBuf, {W - 95, peakY - 4},
                        cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 230, 0), 1);

            snprintf(pvBuf, sizeof(pvBuf), "V=%.2f", runValley);
            cv::putText(img, pvBuf, {W - 95, valleyY + 12},
                        cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 230, 230), 1);

            int midY = (peakY + valleyY) / 2;
            snprintf(pvBuf, sizeof(pvBuf), "A=%.2f", ampZ);
            cv::rectangle(img, {W - 97, midY - 8}, {W - 35, midY + 6},
                          cv::Scalar(30, 30, 30), cv::FILLED);
            cv::putText(img, pvBuf, {W - 95, midY + 4},
                        cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(200, 200, 200), 1);
        }
    }

    // Title (top-left)
    cv::putText(img, label, {ML + 5, TOP - 3}, cv::FONT_HERSHEY_SIMPLEX, 0.5, markerColor, 2);

    // Time label (top-right)
    char tbuf[32];
    snprintf(tbuf, sizeof(tbuf), "t=%.2fs", trace[sampleIdx].time_s);
    cv::putText(img, tbuf, {W - 120, TOP - 3}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(210, 210, 210), 1);

    // --- Click Inspect Overlay ---
    if (click && click->active && click->sampleIdx >= start && click->sampleIdx <= sampleIdx) {
        int localI = click->sampleIdx - start;
        int px = toX(localI);
        int gyVal = toY(click->value);
        int fyVal = toY(click->filteredValue);

        // Vertical line at clicked sample
        cv::line(img, {px, TOP}, {px, BOT}, cv::Scalar(0, 255, 255), 1);

        // Horizontal dashed line at globalZ value
        for (int dx = ML; dx < W - 5; dx += 8) {
            cv::line(img, {dx, gyVal}, {std::min(dx + 4, W - 5), gyVal}, cv::Scalar(0, 200, 200), 1);
        }

        // Small circles at intersection points
        cv::circle(img, {px, gyVal}, 4, cv::Scalar(255, 160, 40), cv::FILLED);
        cv::circle(img, {px, fyVal}, 4, cv::Scalar(40, 140, 255), cv::FILLED);

        // Value label
        char clickBuf[128];
        snprintf(clickBuf, sizeof(clickBuf), "t=%.3fs  Z=%.3f  filt=%.3f",
                 click->time_s, click->value, click->filteredValue);
        int labelX = px + 8;
        int labelY = std::min(gyVal, fyVal) - 8;
        if (labelX + 280 > W) labelX = px - 285;
        if (labelY < TOP + 12) labelY = std::max(gyVal, fyVal) + 18;
        // Background rect for readability
        int textW = 270, textH = 16;
        cv::rectangle(img, {labelX - 2, labelY - 12}, {labelX + textW, labelY + textH - 8},
                      cv::Scalar(20, 20, 20), cv::FILLED);
        cv::putText(img, clickBuf, {labelX, labelY},
                    cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(0, 255, 255), 1);
    }

    return img;
}

static cv::Mat drawStepInfo(const std::vector<StepResult>& steps,
                             const std::vector<StepResult>& oldSteps,
                             const std::vector<SampleData>& trace,
                             int sampleIdx, int upToStep, int upToOldStep,
                             int stride, const std::string& dataFileName) {
    const int W = 800, H = 500;
    const int COL_L = 15;          // left column x
    const int COL_R = W / 2 + 10;  // right column x
    const int ROW_H = 20;          // row height

    cv::Mat img(H, W, CV_8UC3, cv::Scalar(40, 40, 40));

    char buf[256];

    auto put = [&](int col_x, int row, const std::string& text, cv::Scalar col = cv::Scalar(210,210,210)) {
        cv::putText(img, text, {col_x, 24 + row * ROW_H}, cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1);
    };

    // --- Header (full width) ---
    put(COL_L, 0, "File: " + dataFileName, cv::Scalar(50, 230, 230));

    snprintf(buf, sizeof(buf), "Sample: %d / %d  (stride: %d)   Time: %.3f s",
             sampleIdx, (int)trace.size() - 1, stride, trace[sampleIdx].time_s);
    put(COL_L, 1, buf, cv::Scalar(180, 180, 180));

    // --- Column divider ---
    int divX = W / 2 - 2;
    cv::line(img, {divX, 2 * ROW_H + 10}, {divX, H - 10}, cv::Scalar(80, 80, 80), 1);

    // --- New algorithm averages ---
    double newTotalLen = 0.0, newTotalFreq = 0.0, newTotalAmp = 0.0;
    int newCount = 0;
    for (int i = 0; i < upToStep && i < (int)steps.size(); ++i) {
        newTotalLen += steps[i].stepLen;
        newTotalFreq += steps[i].freqHz;
        newTotalAmp += steps[i].ampZ;
        newCount++;
    }

    // --- Old algorithm averages ---
    double oldTotalLen = 0.0, oldTotalFreq = 0.0, oldTotalAmp = 0.0;
    int oldCount = 0;
    for (int i = 0; i < upToOldStep && i < (int)oldSteps.size(); ++i) {
        oldTotalLen += oldSteps[i].stepLen;
        oldTotalFreq += oldSteps[i].freqHz;
        oldTotalAmp += oldSteps[i].ampZ;
        oldCount++;
    }

    // ========== LEFT COLUMN: New Algorithm ==========
    int row = 2;
    put(COL_L, row, "=== NEW Algorithm ===", cv::Scalar(80, 220, 80));

    if (upToStep > 0 && !steps.empty()) {
        int si = std::min(upToStep - 1, (int)steps.size() - 1);
        const StepResult& s = steps[si];

        snprintf(buf, sizeof(buf), "Step: #%d / %d", s.stepNum, (int)steps.size());
        put(COL_L, ++row, buf, cv::Scalar(80, 220, 80));

        snprintf(buf, sizeof(buf), "StepLen:  %.3f m (raw: %.3f)", s.stepLen, s.rawStepLen);
        put(COL_L, ++row, buf);

        snprintf(buf, sizeof(buf), "Direction: %.1f deg", s.directionDeg);
        put(COL_L, ++row, buf);

        snprintf(buf, sizeof(buf), "Position: (%.2f, %.2f)", s.x, s.y);
        put(COL_L, ++row, buf);

        snprintf(buf, sizeof(buf), "Amplitude: %.3f", s.ampZ);
        put(COL_L, ++row, buf);

        snprintf(buf, sizeof(buf), "Peak: %.3f  Valley: %.3f", s.peak, s.valley);
        put(COL_L, ++row, buf);

        snprintf(buf, sizeof(buf), "Frequency: %.2f Hz", s.freqHz);
        put(COL_L, ++row, buf);

        if (newCount > 0) {
            ++row;
            snprintf(buf, sizeof(buf), "Avg SL:   %.3f m", newTotalLen / newCount);
            put(COL_L, row, buf, cv::Scalar(255, 200, 100));
            snprintf(buf, sizeof(buf), "Avg Freq: %.2f Hz", newTotalFreq / newCount);
            put(COL_L, ++row, buf, cv::Scalar(255, 200, 100));
            snprintf(buf, sizeof(buf), "Avg Amp:  %.3f", newTotalAmp / newCount);
            put(COL_L, ++row, buf, cv::Scalar(255, 200, 100));
        }
    } else {
        put(COL_L, ++row, "No steps yet.", cv::Scalar(150, 150, 150));
    }

    // ========== RIGHT COLUMN: Old Algorithm ==========
    row = 2;
    put(COL_R, row, "=== OLD Algorithm ===", cv::Scalar(255, 150, 50));

    if (upToOldStep > 0 && !oldSteps.empty()) {
        int oi = std::min(upToOldStep - 1, (int)oldSteps.size() - 1);
        const StepResult& o = oldSteps[oi];

        snprintf(buf, sizeof(buf), "Step: #%d / %d", o.stepNum, (int)oldSteps.size());
        put(COL_R, ++row, buf, cv::Scalar(255, 150, 50));

        snprintf(buf, sizeof(buf), "StepLen:  %.3f m (raw: %.3f)", o.stepLen, o.rawStepLen);
        put(COL_R, ++row, buf);

        snprintf(buf, sizeof(buf), "Direction: %.1f deg", o.directionDeg);
        put(COL_R, ++row, buf);

        snprintf(buf, sizeof(buf), "Position: (%.2f, %.2f)", o.x, o.y);
        put(COL_R, ++row, buf);

        snprintf(buf, sizeof(buf), "Amplitude: %.3f", o.ampZ);
        put(COL_R, ++row, buf);

        snprintf(buf, sizeof(buf), "Peak: %.3f  Valley: %.3f", o.peak, o.valley);
        put(COL_R, ++row, buf);

        snprintf(buf, sizeof(buf), "Frequency: %.2f Hz", o.freqHz);
        put(COL_R, ++row, buf);

        if (oldCount > 0) {
            ++row;
            snprintf(buf, sizeof(buf), "Avg SL:   %.3f m", oldTotalLen / oldCount);
            put(COL_R, row, buf, cv::Scalar(255, 200, 100));
            snprintf(buf, sizeof(buf), "Avg Freq: %.2f Hz", oldTotalFreq / oldCount);
            put(COL_R, ++row, buf, cv::Scalar(255, 200, 100));
            snprintf(buf, sizeof(buf), "Avg Amp:  %.3f", oldTotalAmp / oldCount);
            put(COL_R, ++row, buf, cv::Scalar(255, 200, 100));
        }
    } else {
        put(COL_R, ++row, "No steps yet.", cv::Scalar(150, 150, 150));
    }

    // ========== DIFF ROW (centered) ==========
    int diffRow = 14;
    cv::line(img, {COL_L, diffRow * ROW_H + 6}, {W - COL_L, diffRow * ROW_H + 6}, cv::Scalar(80, 80, 80), 1);

    if (newCount > 0 && oldCount > 0) {
        double dAvgSL = (newTotalLen / newCount) - (oldTotalLen / oldCount);
        double dAvgFreq = (newTotalFreq / newCount) - (oldTotalFreq / oldCount);
        double dAvgAmp = (newTotalAmp / newCount) - (oldTotalAmp / oldCount);

        int si = std::min(upToStep - 1, (int)steps.size() - 1);
        int oi = std::min(upToOldStep - 1, (int)oldSteps.size() - 1);
        double dLatest = steps[si].stepLen - oldSteps[oi].stepLen;

        put(COL_L, diffRow, "DIFF (New - Old)", cv::Scalar(150, 255, 255));

        snprintf(buf, sizeof(buf), "Latest SL: %+.3fm", dLatest);
        put(COL_L + 160, diffRow, buf, cv::Scalar(200, 200, 200));

        snprintf(buf, sizeof(buf), "Avg SL: %+.3fm", dAvgSL);
        put(COL_L + 340, diffRow, buf, cv::Scalar(200, 200, 200));

        snprintf(buf, sizeof(buf), "Avg Freq: %+.2fHz", dAvgFreq);
        put(COL_L + 160, diffRow + 1, buf, cv::Scalar(200, 200, 200));

        snprintf(buf, sizeof(buf), "Avg Amp: %+.3f", dAvgAmp);
        put(COL_L + 340, diffRow + 1, buf, cv::Scalar(200, 200, 200));
    }

    // ========== Step Length History Chart (overlay both) ==========
    int chartTop = (diffRow + 1) * ROW_H + 28;
    int chartBot = H - 15;
    int chartL = COL_L;
    int chartR = W - COL_L;

    if (chartBot - chartTop < 30) return img;

    cv::line(img, {chartL, chartBot}, {chartR, chartBot}, cv::Scalar(80, 80, 80), 1);

    // Find common max for Y scale
    double maxL = 0.5;
    for (int i = 0; i < upToStep && i < (int)steps.size(); ++i)
        maxL = std::max(maxL, steps[i].stepLen);
    for (int i = 0; i < upToOldStep && i < (int)oldSteps.size(); ++i)
        maxL = std::max(maxL, oldSteps[i].stepLen);
    maxL *= 1.1;

    int nNew = std::min(upToStep, (int)steps.size());
    int nOld = std::min(upToOldStep, (int)oldSteps.size());
    int maxN = std::max(nNew, nOld);
    int showStart = std::max(0, maxN - 50);
    int showCount = maxN - showStart;

    if (showCount > 0) {
        double bw = (double)(chartR - chartL) / showCount;
        double halfBw = bw / 2.0;

        // Draw old bars (left half of each slot, blue)
        for (int i = 0; i < showCount; ++i) {
            int idx = showStart + i;
            if (idx >= nOld) continue;
            int bx = chartL + (int)(i * bw);
            int bh = (int)(oldSteps[idx].stepLen / maxL * (chartBot - chartTop));
            cv::rectangle(img, {bx, chartBot - bh}, {bx + std::max((int)halfBw - 1, 1), chartBot},
                          cv::Scalar(255, 150, 50), cv::FILLED);
        }

        // Draw new bars (right half of each slot, red)
        for (int i = 0; i < showCount; ++i) {
            int idx = showStart + i;
            if (idx >= nNew) continue;
            int bx = chartL + (int)(i * bw + halfBw);
            int bh = (int)(steps[idx].stepLen / maxL * (chartBot - chartTop));
            cv::Scalar col = (idx == nNew - 1) ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 0, 220);
            cv::rectangle(img, {bx, chartBot - bh}, {bx + std::max((int)halfBw - 1, 1), chartBot},
                          col, cv::FILLED);
        }

        // Chart legend
        int legendY = chartTop - 5;
        cv::line(img, {chartL, legendY}, {chartL + 20, legendY}, cv::Scalar(0, 0, 220), 3);
        cv::putText(img, "New", {chartL + 25, legendY + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 0, 220), 1);
        cv::line(img, {chartL + 70, legendY}, {chartL + 90, legendY}, cv::Scalar(255, 150, 50), 3);
        cv::putText(img, "Old", {chartL + 95, legendY + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(255, 150, 50), 1);

        cv::putText(img, "Step Length History (Last 50)", {chartL + 160, legendY + 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(120, 120, 120), 1);

        // Y-axis labels
        for (int tick = 0; tick <= 4; ++tick) {
            double val = maxL * tick / 4.0;
            int ty = chartBot - (int)((chartBot - chartTop) * tick / 4.0);
            cv::line(img, {chartL - 3, ty}, {chartL, ty}, cv::Scalar(100, 100, 100), 1);
            snprintf(buf, sizeof(buf), "%.2f", val);
            cv::putText(img, buf, {chartR + 3, ty + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(100, 100, 100), 1);
        }
    }

    return img;
}

// ============================================================
//  4. 메인 호출 함수 (showAtSample)
// ============================================================
void showAtSample(const std::vector<StepResult>& steps,
                  const std::vector<StepResult>& oldSteps,
                  const std::vector<SampleData>& trace,
                  int sampleIdx,
                  int stride,
                  const std::string& dataFileName,
                  const cv::Mat& mapImage,
                  cv::Point initialStartPos,
                  double mapScalePxPerM,
                  const MapViewState& viewState,
                  const WaveformClick* wfClick) {

    int upToStep = 0;
    for (int i = 0; i < (int)steps.size(); ++i) {
        if (steps[i].sampleIdx <= sampleIdx) upToStep = i + 1;
        else break;
    }

    int upToOldStep = 0;
    for (int i = 0; i < (int)oldSteps.size(); ++i) {
        if (oldSteps[i].sampleIdx <= sampleIdx) upToOldStep = i + 1;
        else break;
    }

    // --- Window 1: Dashboard (2 Waveforms + Info + Local Path) ---
    cv::Mat newWav = drawAccelWaveform(trace, sampleIdx, upToStep,
                                        "NEW Step Detection", cv::Scalar(80, 80, 255), false, wfClick);
    cv::Mat oldWav = drawAccelWaveform(trace, sampleIdx, upToOldStep,
                                        "OLD Step Detection", cv::Scalar(255, 150, 50), true, wfClick);
    cv::Mat stepInfo = drawStepInfo(steps, oldSteps, trace, sampleIdx, upToStep, upToOldStep, stride, dataFileName);

    int rightH = newWav.rows + oldWav.rows + stepInfo.rows;
    int rightW = std::max({newWav.cols, oldWav.cols, stepInfo.cols});
    cv::Mat rightCol(rightH, rightW, CV_8UC3, cv::Scalar(40, 40, 40));
    int yOff = 0;
    newWav.copyTo(rightCol(cv::Rect(0, yOff, newWav.cols, newWav.rows)));   yOff += newWav.rows;
    oldWav.copyTo(rightCol(cv::Rect(0, yOff, oldWav.cols, oldWav.rows)));   yOff += oldWav.rows;
    stepInfo.copyTo(rightCol(cv::Rect(0, yOff, stepInfo.cols, stepInfo.rows)));

    cv::Mat pdrPath = drawPDRPath(steps, oldSteps, upToStep, upToOldStep, rightH);

    int totalW = pdrPath.cols + rightCol.cols;
    cv::Mat dashboard(rightH, totalW, CV_8UC3, cv::Scalar(40, 40, 40));
    pdrPath.copyTo(dashboard(cv::Rect(0, 0, pdrPath.cols, pdrPath.rows)));
    rightCol.copyTo(dashboard(cv::Rect(pdrPath.cols, 0, rightCol.cols, rightCol.rows)));

    cv::imshow("PDR Dashboard", dashboard);

    // --- Window 2: World Map (Dynamic View) ---
    drawWorldMapWindow(steps, oldSteps, upToStep, upToOldStep, mapImage, initialStartPos, mapScalePxPerM, viewState);
}

// ============================================================
//  5. Analysis Window — 설계 의도 검증용 5개 차트
// ============================================================

// --- Helper: Scatter Axes (Chart 2, 3) ---
struct ScatterAxes {
    int W, H;
    int left, right, top, bot;
    double minX, maxX, minY, maxY;

    ScatterAxes(int w, int h, int pad = 50)
        : W(w), H(h), left(pad), right(w - 15), top(25), bot(h - 30),
          minX(0), maxX(1), minY(0), maxY(1) {}

    void fitRange(double xLo, double xHi, double yLo, double yHi) {
        if (xHi - xLo < 1e-6) { xLo -= 0.5; xHi += 0.5; }
        if (yHi - yLo < 1e-6) { yLo -= 0.5; yHi += 0.5; }
        double xMargin = (xHi - xLo) * 0.08;
        double yMargin = (yHi - yLo) * 0.08;
        minX = xLo - xMargin; maxX = xHi + xMargin;
        minY = yLo - yMargin; maxY = yHi + yMargin;
    }

    cv::Point toScreen(double x, double y) const {
        int sx = left + (int)((x - minX) / (maxX - minX) * (right - left));
        int sy = bot  - (int)((y - minY) / (maxY - minY) * (bot - top));
        return {sx, sy};
    }

    void drawAxes(cv::Mat& img) const {
        cv::line(img, {left, bot}, {right, bot}, cv::Scalar(100,100,100), 1);
        cv::line(img, {left, top}, {left, bot}, cv::Scalar(100,100,100), 1);
        // X ticks
        for (int i = 0; i <= 4; ++i) {
            double v = minX + (maxX - minX) * i / 4.0;
            int sx = left + (right - left) * i / 4;
            cv::line(img, {sx, bot}, {sx, bot + 4}, cv::Scalar(80,80,80), 1);
            char buf[16]; snprintf(buf, sizeof(buf), "%.2f", v);
            cv::putText(img, buf, {sx - 15, bot + 16}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(140,140,140), 1);
        }
        // Y ticks
        for (int i = 0; i <= 4; ++i) {
            double v = minY + (maxY - minY) * i / 4.0;
            int sy = bot - (bot - top) * i / 4;
            cv::line(img, {left - 4, sy}, {left, sy}, cv::Scalar(80,80,80), 1);
            char buf[16]; snprintf(buf, sizeof(buf), "%.2f", v);
            cv::putText(img, buf, {2, sy + 4}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(140,140,140), 1);
        }
    }
};

// --- Helper: Time-Series Axes (Chart 4, 5) ---
struct TimeSeriesAxes {
    int W, H;
    int left, right, top, bot;
    int stepStart, stepEnd; // x range in step number
    double minY, maxY;

    TimeSeriesAxes(int w, int h, int pad = 50)
        : W(w), H(h), left(pad), right(w - 15), top(25), bot(h - 30),
          stepStart(0), stepEnd(1), minY(0), maxY(1) {}

    void setXRange(int s, int e) { stepStart = s; stepEnd = std::max(s + 1, e); }
    void setYRange(double lo, double hi) {
        if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
        double margin = (hi - lo) * 0.08;
        minY = lo - margin; maxY = hi + margin;
    }

    cv::Point toScreen(int step, double y) const {
        int sx = left + (int)((double)(step - stepStart) / (stepEnd - stepStart) * (right - left));
        int sy = bot  - (int)((y - minY) / (maxY - minY) * (bot - top));
        return {sx, sy};
    }

    void drawAxes(cv::Mat& img) const {
        cv::line(img, {left, bot}, {right, bot}, cv::Scalar(100,100,100), 1);
        cv::line(img, {left, top}, {left, bot}, cv::Scalar(100,100,100), 1);
        // X ticks
        int range = stepEnd - stepStart;
        int tickCount = std::min(range, 5);
        for (int i = 0; i <= tickCount; ++i) {
            int step = stepStart + range * i / std::max(tickCount, 1);
            int sx = left + (right - left) * i / std::max(tickCount, 1);
            cv::line(img, {sx, bot}, {sx, bot + 4}, cv::Scalar(80,80,80), 1);
            cv::putText(img, std::to_string(step), {sx - 8, bot + 16}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(140,140,140), 1);
        }
        // Y ticks
        for (int i = 0; i <= 4; ++i) {
            double v = minY + (maxY - minY) * i / 4.0;
            int sy = bot - (bot - top) * i / 4;
            cv::line(img, {left - 4, sy}, {left, sy}, cv::Scalar(80,80,80), 1);
            char buf[16]; snprintf(buf, sizeof(buf), "%.2f", v);
            cv::putText(img, buf, {2, sy + 4}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(140,140,140), 1);
        }
    }
};

// --- Helper: Draw filled triangle (peak marker ▲) ---
static void drawTriangleUp(cv::Mat& img, cv::Point center, int size, cv::Scalar color, bool filled) {
    cv::Point pts[3] = {
        {center.x, center.y - size},
        {center.x - size, center.y + size},
        {center.x + size, center.y + size}
    };
    if (filled) {
        cv::fillConvexPoly(img, pts, 3, color, cv::LINE_AA);
    } else {
        const cv::Point* ppt[1] = { pts };
        int npt[1] = { 3 };
        cv::polylines(img, ppt, npt, 1, true, color, 1, cv::LINE_AA);
    }
}

// --- Helper: Draw filled triangle (valley marker ▼) ---
static void drawTriangleDown(cv::Mat& img, cv::Point center, int size, cv::Scalar color, bool filled) {
    cv::Point pts[3] = {
        {center.x, center.y + size},
        {center.x - size, center.y - size},
        {center.x + size, center.y - size}
    };
    if (filled) {
        cv::fillConvexPoly(img, pts, 3, color, cv::LINE_AA);
    } else {
        const cv::Point* ppt[1] = { pts };
        int npt[1] = { 3 };
        cv::polylines(img, ppt, npt, 1, true, color, 1, cv::LINE_AA);
    }
}

// ----- Chart 1: Peak/Valley 파형 (600x300) -----
static cv::Mat drawPeakValleyChart(const std::vector<StepResult>& steps,
                                    const std::vector<StepResult>& oldSteps,
                                    const std::vector<SampleData>& trace,
                                    int sampleIdx, int upToStep, int upToOldStep) {
    const int W = 600, H = 300;
    const int ML = 40, TOP = 30, BOT = H - 20;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(30, 30, 30));

    // Title
    cv::putText(img, "Peak / Valley Detection", {ML + 5, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(210,210,210), 1);

    if (trace.empty()) {
        cv::putText(img, "No Data", {W/2-40, H/2}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(150,150,150), 1);
        return img;
    }

    double lo = -5.0, hi = 5.0;
    // Grid lines
    for (double v = -4; v <= 4; v += 2.0) {
        int py = BOT - (int)((v - lo) / (hi - lo) * (BOT - TOP));
        cv::line(img, {ML, py}, {W - 5, py}, cv::Scalar(45, 45, 45), 1);
    }
    int zy = BOT - (int)((0 - lo) / (hi - lo) * (BOT - TOP));
    cv::line(img, {ML, zy}, {W - 5, zy}, cv::Scalar(70, 70, 70), 1);

    int winSize = 400;
    int start = std::max(0, sampleIdx - winSize + 1);
    int count = sampleIdx - start + 1;
    int xOff = winSize - count;
    double ppx = (double)(W - ML - 5) / winSize;

    auto toY = [&](double v) { return BOT - (int)((v - lo) / (hi - lo) * (BOT - TOP)); };
    auto toX = [&](int localI) { return ML + (int)((xOff + localI) * ppx); };

    // Draw waveforms
    for (int i = 1; i < count; ++i) {
        int si = start + i;
        cv::line(img, {toX(i-1), toY(trace[si-1].globalZ)}, {toX(i), toY(trace[si].globalZ)},
                 cv::Scalar(40, 160, 255), 1, cv::LINE_AA);  // globalZ 주황
        cv::line(img, {toX(i-1), toY(trace[si-1].filteredZ)}, {toX(i), toY(trace[si].filteredZ)},
                 cv::Scalar(255, 140, 40), 2, cv::LINE_AA);  // filteredZ 파랑
    }

    // Helper: search backwards from step detection sample to find actual peak/valley position
    auto findPeakSample = [&](int stepSampleIdx, double peakVal, int prevStepSampleIdx) -> int {
        int searchStart = std::max(prevStepSampleIdx, start);
        int bestIdx = stepSampleIdx;
        double bestDist = 1e9;
        for (int s = stepSampleIdx; s >= searchStart; --s) {
            double dist = std::abs(trace[s].filteredZ - peakVal);
            if (dist < bestDist) { bestDist = dist; bestIdx = s; }
        }
        return bestIdx;
    };
    auto findValleySample = [&](int stepSampleIdx, double valleyVal, int prevStepSampleIdx) -> int {
        int searchStart = std::max(prevStepSampleIdx, start);
        int bestIdx = stepSampleIdx;
        double bestDist = 1e9;
        for (int s = stepSampleIdx; s >= searchStart; --s) {
            double dist = std::abs(trace[s].filteredZ - valleyVal);
            if (dist < bestDist) { bestDist = dist; bestIdx = s; }
        }
        return bestIdx;
    };

    // New step markers: filled ▲ (peak) and ▼ (valley) at actual positions
    for (int i = 0; i < count; ++i) {
        int si = start + i;
        if (!trace[si].isStep || trace[si].stepIdx < 0 || trace[si].stepIdx >= upToStep) continue;
        int idx = trace[si].stepIdx;
        if (idx >= (int)steps.size()) continue;
        int prevSi = (idx > 0) ? steps[idx - 1].sampleIdx : 0;
        int peakSi = findPeakSample(si, steps[idx].peak, prevSi);
        int valleySi = findValleySample(si, steps[idx].valley, prevSi);
        int peakLocal = peakSi - start;
        int valleyLocal = valleySi - start;
        if (peakLocal >= 0 && peakLocal < count)
            drawTriangleUp(img, {toX(peakLocal), toY(steps[idx].peak)}, 6, cv::Scalar(0, 255, 0), true);
        if (valleyLocal >= 0 && valleyLocal < count)
            drawTriangleDown(img, {toX(valleyLocal), toY(steps[idx].valley)}, 6, cv::Scalar(0, 200, 255), true);
    }

    // Old step markers: hollow ▲ and ▼ at actual positions
    for (int i = 0; i < count; ++i) {
        int si = start + i;
        if (!trace[si].isOldStep || trace[si].oldStepIdx < 0 || trace[si].oldStepIdx >= upToOldStep) continue;
        int idx = trace[si].oldStepIdx;
        if (idx >= (int)oldSteps.size()) continue;
        int prevSi = (idx > 0) ? oldSteps[idx - 1].sampleIdx : 0;
        int peakSi = findPeakSample(si, oldSteps[idx].peak, prevSi);
        int valleySi = findValleySample(si, oldSteps[idx].valley, prevSi);
        int peakLocal = peakSi - start;
        int valleyLocal = valleySi - start;
        if (peakLocal >= 0 && peakLocal < count)
            drawTriangleUp(img, {toX(peakLocal), toY(oldSteps[idx].peak)}, 6, cv::Scalar(0, 255, 0), false);
        if (valleyLocal >= 0 && valleyLocal < count)
            drawTriangleDown(img, {toX(valleyLocal), toY(oldSteps[idx].valley)}, 6, cv::Scalar(0, 200, 255), false);
    }

    // Legend
    int ly = H - 12;
    cv::putText(img, "globalZ", {ML, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(40,160,255), 1);
    cv::putText(img, "filteredZ", {ML+70, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255,140,40), 1);
    drawTriangleUp(img, {ML+155, ly-4}, 5, cv::Scalar(0,255,0), true);
    cv::putText(img, "Peak", {ML+165, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0,255,0), 1);
    drawTriangleDown(img, {ML+210, ly-4}, 5, cv::Scalar(0,200,255), true);
    cv::putText(img, "Valley", {ML+220, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0,200,255), 1);
    cv::putText(img, "(hollow=Old)", {ML+280, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(120,120,120), 1);

    return img;
}

// ----- Chart 2: Amplitude vs Step Length Scatter (600x300) -----
static cv::Mat drawAmpVsStepLen(const std::vector<StepResult>& steps,
                                 const std::vector<StepResult>& oldSteps,
                                 int upToStep, int upToOldStep) {
    const int W = 600, H = 300;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(30, 30, 30));

    cv::putText(img, "Amplitude vs Step Length", {55, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(210,210,210), 1);

    int nNew = std::min(upToStep, (int)steps.size());
    int nOld = std::min(upToOldStep, (int)oldSteps.size());

    if (nNew == 0 && nOld == 0) {
        cv::putText(img, "No Data", {W/2-40, H/2}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(150,150,150), 1);
        return img;
    }

    ScatterAxes ax(W, H);
    double xLo = DBL_MAX, xHi = -DBL_MAX, yLo = DBL_MAX, yHi = -DBL_MAX;
    for (int i = 0; i < nNew; ++i) {
        xLo = std::min(xLo, steps[i].ampZ); xHi = std::max(xHi, steps[i].ampZ);
        yLo = std::min(yLo, steps[i].stepLen); yHi = std::max(yHi, steps[i].stepLen);
    }
    for (int i = 0; i < nOld; ++i) {
        xLo = std::min(xLo, oldSteps[i].ampZ); xHi = std::max(xHi, oldSteps[i].ampZ);
        yLo = std::min(yLo, oldSteps[i].stepLen); yHi = std::max(yHi, oldSteps[i].stepLen);
    }
    ax.fitRange(xLo, xHi, yLo, yHi);
    ax.drawAxes(img);

    // Old points (orange)
    for (int i = 0; i < nOld; ++i) {
        cv::Point p = ax.toScreen(oldSteps[i].ampZ, oldSteps[i].stepLen);
        int r = (i == nOld - 1) ? 7 : 4;
        cv::circle(img, p, r, cv::Scalar(255, 150, 50), cv::FILLED, cv::LINE_AA);
    }
    // New points (red)
    for (int i = 0; i < nNew; ++i) {
        cv::Point p = ax.toScreen(steps[i].ampZ, steps[i].stepLen);
        int r = (i == nNew - 1) ? 7 : 4;
        cv::circle(img, p, r, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }

    // Legend
    int ly = H - 12;
    cv::circle(img, {ax.left, ly - 3}, 4, cv::Scalar(0,0,255), cv::FILLED);
    cv::putText(img, "New", {ax.left + 8, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0,0,255), 1);
    cv::circle(img, {ax.left + 45, ly - 3}, 4, cv::Scalar(255,150,50), cv::FILLED);
    cv::putText(img, "Old", {ax.left + 53, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255,150,50), 1);

    // Axis labels
    cv::putText(img, "ampZ", {W/2, H - 2}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(140,140,140), 1);

    return img;
}

// ----- Chart 3: Frequency vs Step Length Scatter (600x300) -----
static cv::Mat drawFreqVsStepLen(const std::vector<StepResult>& steps,
                                  const std::vector<StepResult>& oldSteps,
                                  int upToStep, int upToOldStep) {
    const int W = 600, H = 300;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(30, 30, 30));

    cv::putText(img, "Frequency vs Step Length", {55, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(210,210,210), 1);

    int nNew = std::min(upToStep, (int)steps.size());
    int nOld = std::min(upToOldStep, (int)oldSteps.size());

    if (nNew == 0 && nOld == 0) {
        cv::putText(img, "No Data", {W/2-40, H/2}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(150,150,150), 1);
        return img;
    }

    ScatterAxes ax(W, H);
    double xLo = DBL_MAX, xHi = -DBL_MAX, yLo = DBL_MAX, yHi = -DBL_MAX;
    for (int i = 0; i < nNew; ++i) {
        xLo = std::min(xLo, steps[i].freqHz); xHi = std::max(xHi, steps[i].freqHz);
        yLo = std::min(yLo, steps[i].stepLen); yHi = std::max(yHi, steps[i].stepLen);
    }
    for (int i = 0; i < nOld; ++i) {
        xLo = std::min(xLo, oldSteps[i].freqHz); xHi = std::max(xHi, oldSteps[i].freqHz);
        yLo = std::min(yLo, oldSteps[i].stepLen); yHi = std::max(yHi, oldSteps[i].stepLen);
    }
    ax.fitRange(xLo, xHi, yLo, yHi);

    // Draw 1.5~2.2Hz normal walking band BEFORE axes
    {
        cv::Point topLeft = ax.toScreen(1.5, ax.maxY);
        cv::Point botRight = ax.toScreen(2.2, ax.minY);
        cv::rectangle(img, topLeft, botRight, cv::Scalar(45, 50, 45), cv::FILLED);
        cv::putText(img, "1.5-2.2Hz", {(topLeft.x + botRight.x)/2 - 30, ax.top + 14},
                    cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(80, 120, 80), 1);
    }

    ax.drawAxes(img);

    // Old points
    for (int i = 0; i < nOld; ++i) {
        cv::Point p = ax.toScreen(oldSteps[i].freqHz, oldSteps[i].stepLen);
        int r = (i == nOld - 1) ? 7 : 4;
        cv::circle(img, p, r, cv::Scalar(255, 150, 50), cv::FILLED, cv::LINE_AA);
    }
    // New points
    for (int i = 0; i < nNew; ++i) {
        cv::Point p = ax.toScreen(steps[i].freqHz, steps[i].stepLen);
        int r = (i == nNew - 1) ? 7 : 4;
        cv::circle(img, p, r, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }

    // Legend
    int ly = H - 12;
    cv::circle(img, {ax.left, ly - 3}, 4, cv::Scalar(0,0,255), cv::FILLED);
    cv::putText(img, "New", {ax.left + 8, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0,0,255), 1);
    cv::circle(img, {ax.left + 45, ly - 3}, 4, cv::Scalar(255,150,50), cv::FILLED);
    cv::putText(img, "Old", {ax.left + 53, ly}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255,150,50), 1);

    cv::putText(img, "freqHz", {W/2, H - 2}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(140,140,140), 1);

    return img;
}

// ----- Chart 4: Raw vs Smoothed Step Length (600x300) -----
static cv::Mat drawRawVsSmoothedSL(const std::vector<StepResult>& steps,
                                    const std::vector<StepResult>& oldSteps,
                                    int upToStep, int upToOldStep) {
    const int W = 600, H = 300;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(30, 30, 30));

    cv::putText(img, "Raw vs Smoothed Step Length", {55, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(210,210,210), 1);

    int nNew = std::min(upToStep, (int)steps.size());
    int nOld = std::min(upToOldStep, (int)oldSteps.size());

    if (nNew == 0 && nOld == 0) {
        cv::putText(img, "No Data", {W/2-40, H/2}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(150,150,150), 1);
        return img;
    }

    TimeSeriesAxes ax(W, H);
    int maxN = std::max(nNew, nOld);
    ax.setXRange(0, maxN);

    double yLo = DBL_MAX, yHi = -DBL_MAX;
    for (int i = 0; i < nNew; ++i) {
        yLo = std::min({yLo, steps[i].stepLen, steps[i].rawStepLen});
        yHi = std::max({yHi, steps[i].stepLen, steps[i].rawStepLen});
    }
    for (int i = 0; i < nOld; ++i) {
        yLo = std::min({yLo, oldSteps[i].stepLen, oldSteps[i].rawStepLen});
        yHi = std::max({yHi, oldSteps[i].stepLen, oldSteps[i].rawStepLen});
    }
    ax.setYRange(yLo, yHi);
    ax.drawAxes(img);

    // Old: raw thin orange (1px) + smoothed thick orange (2px)
    for (int i = 1; i < nOld; ++i) {
        cv::line(img, ax.toScreen(i-1, oldSteps[i-1].rawStepLen), ax.toScreen(i, oldSteps[i].rawStepLen),
                 cv::Scalar(255, 150, 50), 1, cv::LINE_AA);
        cv::line(img, ax.toScreen(i-1, oldSteps[i-1].stepLen), ax.toScreen(i, oldSteps[i].stepLen),
                 cv::Scalar(255, 150, 50), 2, cv::LINE_AA);
    }
    // New: raw thin red (1px) + smoothed thick red (2px)
    for (int i = 1; i < nNew; ++i) {
        cv::line(img, ax.toScreen(i-1, steps[i-1].rawStepLen), ax.toScreen(i, steps[i].rawStepLen),
                 cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::line(img, ax.toScreen(i-1, steps[i-1].stepLen), ax.toScreen(i, steps[i].stepLen),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // Legend
    int ly = H - 12;
    cv::line(img, {ax.left, ly}, {ax.left + 15, ly}, cv::Scalar(0,0,255), 1);
    cv::putText(img, "New raw", {ax.left + 18, ly + 2}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0,0,255), 1);
    cv::line(img, {ax.left + 75, ly}, {ax.left + 95, ly}, cv::Scalar(0,0,255), 2);
    cv::putText(img, "smooth", {ax.left + 98, ly + 2}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0,0,255), 1);
    cv::line(img, {ax.left + 150, ly}, {ax.left + 165, ly}, cv::Scalar(255,150,50), 1);
    cv::putText(img, "Old raw", {ax.left + 168, ly + 2}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(255,150,50), 1);
    cv::line(img, {ax.left + 225, ly}, {ax.left + 245, ly}, cv::Scalar(255,150,50), 2);
    cv::putText(img, "smooth", {ax.left + 248, ly + 2}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(255,150,50), 1);

    cv::putText(img, "step #", {W/2, H - 2}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(140,140,140), 1);

    return img;
}

// ----- Chart 5: Frequency & Amplitude Time Series (1200x300, full width) -----
static cv::Mat drawFreqAmpTimeSeries(const std::vector<StepResult>& steps,
                                      const std::vector<StepResult>& oldSteps,
                                      int upToStep, int upToOldStep) {
    const int W = 1200, H = 300;
    const int LEFT = 50, RIGHT_MARGIN = 55, TOP = 30, BOT = H - 30;
    const int RIGHT = W - RIGHT_MARGIN;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(30, 30, 30));

    cv::putText(img, "Frequency & Amplitude over Steps", {LEFT + 5, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(210,210,210), 1);

    int nNew = std::min(upToStep, (int)steps.size());
    int nOld = std::min(upToOldStep, (int)oldSteps.size());

    if (nNew == 0 && nOld == 0) {
        cv::putText(img, "No Data", {W/2-40, H/2}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(150,150,150), 1);
        return img;
    }

    int maxN = std::max(nNew, nOld);
    int stepStart = 0, stepEnd = std::max(1, maxN);

    // Y ranges: freq (left axis), amp (right axis)
    double freqLo = DBL_MAX, freqHi = -DBL_MAX;
    double ampLo = DBL_MAX, ampHi = -DBL_MAX;
    for (int i = 0; i < nNew; ++i) {
        freqLo = std::min(freqLo, steps[i].freqHz); freqHi = std::max(freqHi, steps[i].freqHz);
        ampLo = std::min(ampLo, steps[i].ampZ); ampHi = std::max(ampHi, steps[i].ampZ);
    }
    for (int i = 0; i < nOld; ++i) {
        freqLo = std::min(freqLo, oldSteps[i].freqHz); freqHi = std::max(freqHi, oldSteps[i].freqHz);
        ampLo = std::min(ampLo, oldSteps[i].ampZ); ampHi = std::max(ampHi, oldSteps[i].ampZ);
    }
    // Add margins
    if (freqHi - freqLo < 0.1) { freqLo -= 0.5; freqHi += 0.5; }
    double fMargin = (freqHi - freqLo) * 0.08;
    freqLo -= fMargin; freqHi += fMargin;
    if (ampHi - ampLo < 0.01) { ampLo -= 0.5; ampHi += 0.5; }
    double aMargin = (ampHi - ampLo) * 0.08;
    ampLo -= aMargin; ampHi += aMargin;

    auto toX = [&](int step) -> int {
        return LEFT + (int)((double)(step - stepStart) / (stepEnd - stepStart) * (RIGHT - LEFT));
    };
    auto freqToY = [&](double f) -> int {
        return BOT - (int)((f - freqLo) / (freqHi - freqLo) * (BOT - TOP));
    };
    auto ampToY = [&](double a) -> int {
        return BOT - (int)((a - ampLo) / (ampHi - ampLo) * (BOT - TOP));
    };

    // Draw 1.5~2.2Hz expected band
    {
        int y1 = freqToY(2.2);
        int y2 = freqToY(1.5);
        y1 = std::max(y1, TOP); y2 = std::min(y2, BOT);
        if (y2 > y1) {
            cv::rectangle(img, {LEFT, y1}, {RIGHT, y2}, cv::Scalar(45, 50, 45), cv::FILLED);
            cv::putText(img, "Expected", {RIGHT - 60, y1 + 12}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(80, 120, 80), 1);
        }
    }

    // Axes
    cv::line(img, {LEFT, BOT}, {RIGHT, BOT}, cv::Scalar(100,100,100), 1);
    cv::line(img, {LEFT, TOP}, {LEFT, BOT}, cv::Scalar(100,100,100), 1);
    cv::line(img, {RIGHT, TOP}, {RIGHT, BOT}, cv::Scalar(100,100,100), 1);

    // Left Y axis (freq) ticks
    for (int i = 0; i <= 4; ++i) {
        double v = freqLo + (freqHi - freqLo) * i / 4.0;
        int sy = BOT - (BOT - TOP) * i / 4;
        cv::line(img, {LEFT - 4, sy}, {LEFT, sy}, cv::Scalar(80,80,80), 1);
        char buf[16]; snprintf(buf, sizeof(buf), "%.1f", v);
        cv::putText(img, buf, {2, sy + 4}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(100, 200, 100), 1);
    }
    cv::putText(img, "Hz", {5, TOP - 5}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(100, 200, 100), 1);

    // Right Y axis (amp) ticks
    for (int i = 0; i <= 4; ++i) {
        double v = ampLo + (ampHi - ampLo) * i / 4.0;
        int sy = BOT - (BOT - TOP) * i / 4;
        cv::line(img, {RIGHT, sy}, {RIGHT + 4, sy}, cv::Scalar(80,80,80), 1);
        char buf[16]; snprintf(buf, sizeof(buf), "%.2f", v);
        cv::putText(img, buf, {RIGHT + 6, sy + 4}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(200, 150, 100), 1);
    }
    cv::putText(img, "amp", {RIGHT + 6, TOP - 5}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(200, 150, 100), 1);

    // X ticks
    int range = stepEnd - stepStart;
    int tickCount = std::min(range, 6);
    for (int i = 0; i <= tickCount; ++i) {
        int step = stepStart + range * i / std::max(tickCount, 1);
        int sx = toX(step);
        cv::line(img, {sx, BOT}, {sx, BOT + 4}, cv::Scalar(80,80,80), 1);
        cv::putText(img, std::to_string(step), {sx - 8, BOT + 16}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(140,140,140), 1);
    }

    // Old freq (thick orange), Old amp (thin orange dashed-like)
    for (int i = 1; i < nOld; ++i) {
        cv::line(img, {toX(i-1), freqToY(oldSteps[i-1].freqHz)}, {toX(i), freqToY(oldSteps[i].freqHz)},
                 cv::Scalar(255, 150, 50), 2, cv::LINE_AA);
        cv::line(img, {toX(i-1), ampToY(oldSteps[i-1].ampZ)}, {toX(i), ampToY(oldSteps[i].ampZ)},
                 cv::Scalar(255, 150, 50), 1, cv::LINE_AA);
    }

    // New freq (thick red), New amp (thin red)
    for (int i = 1; i < nNew; ++i) {
        cv::line(img, {toX(i-1), freqToY(steps[i-1].freqHz)}, {toX(i), freqToY(steps[i].freqHz)},
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::line(img, {toX(i-1), ampToY(steps[i-1].ampZ)}, {toX(i), ampToY(steps[i].ampZ)},
                 cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    // Legend
    int ly = H - 12;
    cv::line(img, {LEFT, ly}, {LEFT + 18, ly}, cv::Scalar(0,0,255), 2);
    cv::putText(img, "New freq", {LEFT + 22, ly + 2}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0,0,255), 1);
    cv::line(img, {LEFT + 85, ly}, {LEFT + 103, ly}, cv::Scalar(0,0,255), 1);
    cv::putText(img, "New amp", {LEFT + 107, ly + 2}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0,0,255), 1);
    cv::line(img, {LEFT + 170, ly}, {LEFT + 188, ly}, cv::Scalar(255,150,50), 2);
    cv::putText(img, "Old freq", {LEFT + 192, ly + 2}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(255,150,50), 1);
    cv::line(img, {LEFT + 255, ly}, {LEFT + 273, ly}, cv::Scalar(255,150,50), 1);
    cv::putText(img, "Old amp", {LEFT + 277, ly + 2}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(255,150,50), 1);

    return img;
}

// ----- showAnalysisWindow: 3행 합성 -----
void showAnalysisWindow(const std::vector<StepResult>& steps,
                        const std::vector<StepResult>& oldSteps,
                        const std::vector<SampleData>& trace,
                        int sampleIdx) {
    // Compute upToStep / upToOldStep (same logic as showAtSample)
    int upToStep = 0;
    for (int i = 0; i < (int)steps.size(); ++i) {
        if (steps[i].sampleIdx <= sampleIdx) upToStep = i + 1;
        else break;
    }
    int upToOldStep = 0;
    for (int i = 0; i < (int)oldSteps.size(); ++i) {
        if (oldSteps[i].sampleIdx <= sampleIdx) upToOldStep = i + 1;
        else break;
    }

    // Row 1: Chart1 (600x300) | Chart2 (600x300)
    cv::Mat c1 = drawPeakValleyChart(steps, oldSteps, trace, sampleIdx, upToStep, upToOldStep);
    cv::Mat c2 = drawAmpVsStepLen(steps, oldSteps, upToStep, upToOldStep);
    cv::Mat row1(300, 1200, CV_8UC3, cv::Scalar(30, 30, 30));
    c1.copyTo(row1(cv::Rect(0, 0, 600, 300)));
    c2.copyTo(row1(cv::Rect(600, 0, 600, 300)));

    // Row 2: Chart3 (600x300) | Chart4 (600x300)
    cv::Mat c3 = drawFreqVsStepLen(steps, oldSteps, upToStep, upToOldStep);
    cv::Mat c4 = drawRawVsSmoothedSL(steps, oldSteps, upToStep, upToOldStep);
    cv::Mat row2(300, 1200, CV_8UC3, cv::Scalar(30, 30, 30));
    c3.copyTo(row2(cv::Rect(0, 0, 600, 300)));
    c4.copyTo(row2(cv::Rect(600, 0, 600, 300)));

    // Row 3: Chart5 (1200x300)
    cv::Mat row3 = drawFreqAmpTimeSeries(steps, oldSteps, upToStep, upToOldStep);

    // Compose 3-row canvas (1200x900)
    cv::Mat canvas(900, 1200, CV_8UC3, cv::Scalar(30, 30, 30));
    row1.copyTo(canvas(cv::Rect(0, 0, 1200, 300)));
    row2.copyTo(canvas(cv::Rect(0, 300, 1200, 300)));
    row3.copyTo(canvas(cv::Rect(0, 600, 1200, 300)));

    cv::imshow("Analysis", canvas);
}

// ============================================================
//  6. Step Analysis Window — 보폭 계산 파이프라인 + 걸음별 테이블
// ============================================================

// Helper: draw a pipeline stage box
static void drawPipeBox(cv::Mat& img, int x, int y, int w, int h,
                        const char* label, const char* value,
                        cv::Scalar borderColor, cv::Scalar valueColor) {
    cv::rectangle(img, {x, y}, {x + w, y + h}, borderColor, 1);
    int labelX = x + 4;
    int labelY = y + 14;
    cv::putText(img, label, {labelX, labelY}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(180, 180, 180), 1);
    cv::putText(img, value, {labelX, labelY + 16}, cv::FONT_HERSHEY_PLAIN, 0.9, valueColor, 1);
}

void showStepAnalysisWindow(const std::vector<StepResult>& steps,
                            const std::vector<StepResult>& oldSteps,
                            int sampleIdx,
                            bool isFemale) {
    const int W = 1200, H = 720;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(30, 30, 30));

    // Compute upToStep / upToOldStep
    int upToStep = 0;
    for (int i = 0; i < (int)steps.size(); ++i) {
        if (steps[i].sampleIdx <= sampleIdx) upToStep = i + 1;
        else break;
    }
    int upToOldStep = 0;
    for (int i = 0; i < (int)oldSteps.size(); ++i) {
        if (oldSteps[i].sampleIdx <= sampleIdx) upToOldStep = i + 1;
        else break;
    }

    char buf[256];

    // ============================================================
    //  Top Section: Amplitude Pipeline (rows 1-2)
    // ============================================================
    cv::putText(img, "Step Length Pipeline", {15, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 220, 255), 2);

    // (Female indicator is shown as scale box in pipeline)

    const int BOX_W = 110, BOX_H = 38, ARROW_GAP = 22;
    const int NUM_AMP_STAGES = 3;  // ampZ, A^1/4, corrected
    const int NUM_FREQ_STAGES = 3; // dt(ms), freq(Hz), k(f)
    const int NUM_OUT_STAGES = 3;  // L0, Lfinal, Lsmooth

    // Layout: amplitude path (3 boxes) + freq path (3 boxes) converge → output (3 boxes)
    // Row arrangement per pipeline (New/Old):
    //   amp row:  [ampZ] → [A^1/4] → [corrected] ──┐
    //                                                ├→ [L0] → [Lfinal] → [Lsmooth]
    //   freq row: [dt(ms)] → [freq(Hz)] → [k(f)]  ──┘

    const int AMP_START_X = 15;
    const int MERGE_GAP = 30;
    const int MERGE_X = AMP_START_X + NUM_AMP_STAGES * (BOX_W + ARROW_GAP); // where paths merge
    const int OUT_START_X = MERGE_X + MERGE_GAP;

    // --- New Pipeline ---
    int newAmpY = 35;
    int newFreqY = newAmpY + BOX_H + 6;
    int newMergeY = newAmpY + BOX_H / 2 + (newFreqY + BOX_H / 2 - newAmpY - BOX_H / 2) / 2; // midpoint
    int newOutY = newMergeY - BOX_H / 2;

    cv::putText(img, "New", {AMP_START_X - 2, newAmpY - 3},
                cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(80, 80, 255), 1);

    struct PipeStage { const char* label; char val[24]; };

    // New amplitude stages
    PipeStage newAmp[3] = {{"ampZ", ""}, {"A^(1/4)", ""}, {"corrected", ""}};
    PipeStage newFreq[3] = {{"dt(ms)", ""}, {"freq(Hz)", ""}, {"k(f)", ""}};
    PipeStage newOut[3] = {{"L0", ""}, {"Lfinal", ""}, {"Lsmooth", ""}};

    if (upToStep > 0) {
        const StepResult& s = steps[std::min(upToStep - 1, (int)steps.size() - 1)];
        snprintf(newAmp[0].val, 24, "%.3f", s.ampZ);
        snprintf(newAmp[1].val, 24, "%.3f", s.fourthRoot);
        snprintf(newAmp[2].val, 24, "%.3f", s.correctedRoot);
        snprintf(newFreq[0].val, 24, "%.0f", s.durationMs);
        snprintf(newFreq[1].val, 24, "%.2f", s.freqHz);
        snprintf(newFreq[2].val, 24, "%.3f", s.kf);
        snprintf(newOut[0].val, 24, "%.3f", s.L0);
        snprintf(newOut[1].val, 24, "%.3f", s.Lfinal);
        snprintf(newOut[2].val, 24, "%.3f", s.stepLen);
    }

    cv::Scalar newBorder(80, 80, 200), newVal(100, 200, 255);

    // Draw amplitude path boxes
    for (int i = 0; i < 3; ++i) {
        int bx = AMP_START_X + i * (BOX_W + ARROW_GAP);
        drawPipeBox(img, bx, newAmpY, BOX_W, BOX_H, newAmp[i].label, newAmp[i].val, newBorder, newVal);
        if (i < 2) {
            cv::arrowedLine(img, {bx + BOX_W + 2, newAmpY + BOX_H/2},
                            {bx + BOX_W + ARROW_GAP - 2, newAmpY + BOX_H/2},
                            newBorder, 1, cv::LINE_AA, 0, 0.3);
        }
    }
    // Draw frequency path boxes
    for (int i = 0; i < 3; ++i) {
        int bx = AMP_START_X + i * (BOX_W + ARROW_GAP);
        drawPipeBox(img, bx, newFreqY, BOX_W, BOX_H, newFreq[i].label, newFreq[i].val, newBorder, newVal);
        if (i < 2) {
            cv::arrowedLine(img, {bx + BOX_W + 2, newFreqY + BOX_H/2},
                            {bx + BOX_W + ARROW_GAP - 2, newFreqY + BOX_H/2},
                            newBorder, 1, cv::LINE_AA, 0, 0.3);
        }
    }
    // Merge lines: amp path and freq path converge
    int lastAmpBoxRight = AMP_START_X + 2 * (BOX_W + ARROW_GAP) + BOX_W;
    int lastFreqBoxRight = lastAmpBoxRight;
    cv::line(img, {lastAmpBoxRight + 2, newAmpY + BOX_H/2},
             {MERGE_X + 10, newOutY + BOX_H/2}, newBorder, 1, cv::LINE_AA);
    cv::line(img, {lastFreqBoxRight + 2, newFreqY + BOX_H/2},
             {MERGE_X + 10, newOutY + BOX_H/2}, newBorder, 1, cv::LINE_AA);
    cv::arrowedLine(img, {MERGE_X + 10, newOutY + BOX_H/2},
                    {OUT_START_X - 2, newOutY + BOX_H/2}, newBorder, 1, cv::LINE_AA, 0, 0.3);

    // Scale box (ON/OFF) + output boxes: [scale] → [L0] → [Lfinal] → [Lsmooth]
    {
        int sbx = OUT_START_X;
        cv::Scalar sBorder = isFemale ? cv::Scalar(180, 50, 180) : cv::Scalar(70, 70, 70);
        cv::Scalar sFill   = isFemale ? cv::Scalar(60, 15, 60)   : cv::Scalar(35, 35, 35);
        cv::rectangle(img, {sbx, newOutY}, {sbx + BOX_W, newOutY + BOX_H}, sFill, cv::FILLED);
        cv::rectangle(img, {sbx, newOutY}, {sbx + BOX_W, newOutY + BOX_H}, sBorder, 1);
        cv::putText(img, "scale", {sbx + 4, newOutY + 14}, cv::FONT_HERSHEY_PLAIN, 0.8,
                    isFemale ? cv::Scalar(220, 160, 220) : cv::Scalar(100, 100, 100), 1);
        const char* sVal = isFemale ? "x0.90" : "x1.00";
        cv::putText(img, sVal, {sbx + 4, newOutY + 30}, cv::FONT_HERSHEY_PLAIN, 0.9,
                    isFemale ? cv::Scalar(255, 140, 255) : cv::Scalar(80, 80, 80), 1);
        // ON/OFF pill
        int pillX = sbx + BOX_W - 30, pillY = newOutY + 4;
        cv::Scalar pillBg = isFemale ? cv::Scalar(0, 160, 0) : cv::Scalar(60, 60, 60);
        cv::rectangle(img, {pillX, pillY}, {pillX + 26, pillY + 12}, pillBg, cv::FILLED);
        cv::rectangle(img, {pillX, pillY}, {pillX + 26, pillY + 12}, cv::Scalar(120,120,120), 1);
        cv::putText(img, isFemale ? "ON" : "OFF", {pillX + 2, pillY + 10},
                    cv::FONT_HERSHEY_PLAIN, 0.65,
                    isFemale ? cv::Scalar(200, 255, 200) : cv::Scalar(100, 100, 100), 1);
        // Arrow from scale box to L0
        cv::arrowedLine(img, {sbx + BOX_W + 2, newOutY + BOX_H/2},
                        {sbx + BOX_W + ARROW_GAP - 2, newOutY + BOX_H/2},
                        newBorder, 1, cv::LINE_AA, 0, 0.3);
    }

    // Output boxes: L0, Lfinal, Lsmooth (shifted by one box+gap)
    const int OUT_SHIFTED_X = OUT_START_X + BOX_W + ARROW_GAP;
    for (int i = 0; i < 3; ++i) {
        int bx = OUT_SHIFTED_X + i * (BOX_W + ARROW_GAP);
        drawPipeBox(img, bx, newOutY, BOX_W, BOX_H, newOut[i].label, newOut[i].val, newBorder, newVal);
        if (i < 2) {
            cv::arrowedLine(img, {bx + BOX_W + 2, newOutY + BOX_H/2},
                            {bx + BOX_W + ARROW_GAP - 2, newOutY + BOX_H/2},
                            newBorder, 1, cv::LINE_AA, 0, 0.3);
        }
    }

    // --- Old Pipeline (same layout, shifted down) ---
    int oldYOffset = newFreqY + BOX_H + 18;
    int oldAmpY = oldYOffset;
    int oldFreqY = oldAmpY + BOX_H + 6;
    int oldMergeY = oldAmpY + BOX_H / 2 + (oldFreqY + BOX_H / 2 - oldAmpY - BOX_H / 2) / 2;
    int oldOutY = oldMergeY - BOX_H / 2;

    cv::putText(img, "Old", {AMP_START_X - 2, oldAmpY - 3},
                cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255, 150, 50), 1);

    PipeStage oldAmp[3] = {{"ampZ", ""}, {"A^(1/4)", ""}, {"corrected", ""}};
    PipeStage oldFreq[3] = {{"dt(ms)", ""}, {"freq(Hz)", ""}, {"k(f)", ""}};
    PipeStage oldOut[3] = {{"L0", ""}, {"Lfinal", ""}, {"Lsmooth", ""}};

    if (upToOldStep > 0) {
        const StepResult& o = oldSteps[std::min(upToOldStep - 1, (int)oldSteps.size() - 1)];
        snprintf(oldAmp[0].val, 24, "%.3f", o.ampZ);
        snprintf(oldAmp[1].val, 24, "%.3f", o.fourthRoot);
        snprintf(oldAmp[2].val, 24, "%.3f", o.correctedRoot);
        snprintf(oldFreq[0].val, 24, "%.0f", o.durationMs);
        snprintf(oldFreq[1].val, 24, "%.2f", o.freqHz);
        snprintf(oldFreq[2].val, 24, "%.3f", o.kf);
        snprintf(oldOut[0].val, 24, "%.3f", o.L0);
        snprintf(oldOut[1].val, 24, "%.3f", o.Lfinal);
        snprintf(oldOut[2].val, 24, "%.3f", o.stepLen);
    }

    cv::Scalar oldBorder(200, 120, 40), oldVal(255, 180, 80);

    for (int i = 0; i < 3; ++i) {
        int bx = AMP_START_X + i * (BOX_W + ARROW_GAP);
        drawPipeBox(img, bx, oldAmpY, BOX_W, BOX_H, oldAmp[i].label, oldAmp[i].val, oldBorder, oldVal);
        if (i < 2) cv::arrowedLine(img, {bx+BOX_W+2, oldAmpY+BOX_H/2}, {bx+BOX_W+ARROW_GAP-2, oldAmpY+BOX_H/2}, oldBorder, 1, cv::LINE_AA, 0, 0.3);
    }
    for (int i = 0; i < 3; ++i) {
        int bx = AMP_START_X + i * (BOX_W + ARROW_GAP);
        drawPipeBox(img, bx, oldFreqY, BOX_W, BOX_H, oldFreq[i].label, oldFreq[i].val, oldBorder, oldVal);
        if (i < 2) cv::arrowedLine(img, {bx+BOX_W+2, oldFreqY+BOX_H/2}, {bx+BOX_W+ARROW_GAP-2, oldFreqY+BOX_H/2}, oldBorder, 1, cv::LINE_AA, 0, 0.3);
    }
    cv::line(img, {lastAmpBoxRight+2, oldAmpY+BOX_H/2}, {MERGE_X+10, oldOutY+BOX_H/2}, oldBorder, 1, cv::LINE_AA);
    cv::line(img, {lastFreqBoxRight+2, oldFreqY+BOX_H/2}, {MERGE_X+10, oldOutY+BOX_H/2}, oldBorder, 1, cv::LINE_AA);
    cv::arrowedLine(img, {MERGE_X+10, oldOutY+BOX_H/2}, {OUT_START_X-2, oldOutY+BOX_H/2}, oldBorder, 1, cv::LINE_AA, 0, 0.3);

    // Old scale box (same ON/OFF style)
    {
        int sbx = OUT_START_X;
        cv::Scalar sBorder = isFemale ? cv::Scalar(180, 50, 180) : cv::Scalar(70, 70, 70);
        cv::Scalar sFill   = isFemale ? cv::Scalar(60, 15, 60)   : cv::Scalar(35, 35, 35);
        cv::rectangle(img, {sbx, oldOutY}, {sbx + BOX_W, oldOutY + BOX_H}, sFill, cv::FILLED);
        cv::rectangle(img, {sbx, oldOutY}, {sbx + BOX_W, oldOutY + BOX_H}, sBorder, 1);
        cv::putText(img, "scale", {sbx + 4, oldOutY + 14}, cv::FONT_HERSHEY_PLAIN, 0.8,
                    isFemale ? cv::Scalar(220, 160, 220) : cv::Scalar(100, 100, 100), 1);
        const char* sVal = isFemale ? "x0.90" : "x1.00";
        cv::putText(img, sVal, {sbx + 4, oldOutY + 30}, cv::FONT_HERSHEY_PLAIN, 0.9,
                    isFemale ? cv::Scalar(255, 140, 255) : cv::Scalar(80, 80, 80), 1);
        int pillX = sbx + BOX_W - 30, pillY = oldOutY + 4;
        cv::Scalar pillBg = isFemale ? cv::Scalar(0, 160, 0) : cv::Scalar(60, 60, 60);
        cv::rectangle(img, {pillX, pillY}, {pillX + 26, pillY + 12}, pillBg, cv::FILLED);
        cv::rectangle(img, {pillX, pillY}, {pillX + 26, pillY + 12}, cv::Scalar(120,120,120), 1);
        cv::putText(img, isFemale ? "ON" : "OFF", {pillX + 2, pillY + 10},
                    cv::FONT_HERSHEY_PLAIN, 0.65,
                    isFemale ? cv::Scalar(200, 255, 200) : cv::Scalar(100, 100, 100), 1);
        cv::arrowedLine(img, {sbx + BOX_W + 2, oldOutY + BOX_H/2},
                        {sbx + BOX_W + ARROW_GAP - 2, oldOutY + BOX_H/2},
                        oldBorder, 1, cv::LINE_AA, 0, 0.3);
    }

    for (int i = 0; i < 3; ++i) {
        int bx = OUT_SHIFTED_X + i * (BOX_W + ARROW_GAP);
        drawPipeBox(img, bx, oldOutY, BOX_W, BOX_H, oldOut[i].label, oldOut[i].val, oldBorder, oldVal);
        if (i < 2) cv::arrowedLine(img, {bx+BOX_W+2, oldOutY+BOX_H/2}, {bx+BOX_W+ARROW_GAP-2, oldOutY+BOX_H/2}, oldBorder, 1, cv::LINE_AA, 0, 0.3);
    }

    // --- k(f) Breakdown Detail ---
    int detailY = oldFreqY + BOX_H + 14;
    cv::putText(img, "k(f) = base(0.317) + walk_sigmoid + run_sigmoid", {15, detailY},
                cv::FONT_HERSHEY_PLAIN, 0.85, cv::Scalar(140, 140, 140), 1);
    detailY += 16;

    if (upToStep > 0) {
        const StepResult& s = steps[std::min(upToStep - 1, (int)steps.size() - 1)];
        snprintf(buf, sizeof(buf), "New: k(f) = 0.317 + %.4f (walk) + %.4f (run) = %.4f   [f=%.2fHz, dt=%.0fms]",
                 s.kf_walk, s.kf_run, s.kf, s.freqHz, s.durationMs);
        cv::putText(img, buf, {25, detailY}, cv::FONT_HERSHEY_PLAIN, 0.85, cv::Scalar(100, 200, 255), 1);
    }
    detailY += 16;

    if (upToOldStep > 0) {
        const StepResult& o = oldSteps[std::min(upToOldStep - 1, (int)oldSteps.size() - 1)];
        snprintf(buf, sizeof(buf), "Old: k(f) = 0.317 + %.4f (walk) + %.4f (run) = %.4f   [f=%.2fHz, dt=%.0fms]",
                 o.kf_walk, o.kf_run, o.kf, o.freqHz, o.durationMs);
        cv::putText(img, buf, {25, detailY}, cv::FONT_HERSHEY_PLAIN, 0.85, cv::Scalar(255, 180, 80), 1);
    }
    detailY += 6;

    // ============================================================
    //  Bottom Section: Step Table
    // ============================================================
    int tableTop = detailY + 8;
    cv::line(img, {10, tableTop}, {W - 10, tableTop}, cv::Scalar(70, 70, 70), 1);
    tableTop += 5;

    // Column positions (12 columns with dt(ms) added)
    const int colX[] = {15, 60, 130, 200, 270, 350, 420, 490, 560, 640, 730, 830};
    const char* colHeaders[] = {
        "Step#", "Time(s)", "ampZ", "A^1/4", "corr", "dt(ms)", "freq", "k(f)", "L0", "Lfinal", "Lsmooth", "rawSL"
    };
    const int NUM_COLS = 12;
    const int ROW_H = 18;

    auto putCell = [&](int col, int row, const char* text, cv::Scalar color) {
        cv::putText(img, text, {colX[col], tableTop + 15 + row * ROW_H},
                    cv::FONT_HERSHEY_PLAIN, 0.8, color, 1);
    };

    // --- New Steps Header ---
    cv::putText(img, "NEW Steps", {15, tableTop + 12},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(80, 220, 80), 1);

    int headerRow = 1;
    for (int c = 0; c < NUM_COLS; ++c) {
        putCell(c, headerRow, colHeaders[c], cv::Scalar(160, 160, 160));
    }
    cv::line(img, {10, tableTop + 15 + headerRow * ROW_H + 3},
             {W - 10, tableTop + 15 + headerRow * ROW_H + 3}, cv::Scalar(60, 60, 60), 1);

    // New steps data (last 8)
    int nNew = std::min(upToStep, (int)steps.size());
    int newShowStart = std::max(0, nNew - 8);
    int dataRow = headerRow + 1;
    for (int i = newShowStart; i < nNew; ++i) {
        const StepResult& s = steps[i];
        bool isLatest = (i == nNew - 1);
        cv::Scalar rowColor = isLatest ? cv::Scalar(100, 255, 255) : cv::Scalar(180, 180, 220);

        if (isLatest) {
            cv::rectangle(img, {10, tableTop + 15 + dataRow * ROW_H - 12},
                          {W - 10, tableTop + 15 + dataRow * ROW_H + 5},
                          cv::Scalar(50, 50, 70), cv::FILLED);
        }

        snprintf(buf, sizeof(buf), "%d", s.stepNum);       putCell(0, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.2f", s.time_s);      putCell(1, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", s.ampZ);        putCell(2, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", s.fourthRoot);  putCell(3, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", s.correctedRoot); putCell(4, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.0f", s.durationMs);  putCell(5, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.2f", s.freqHz);      putCell(6, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", s.kf);          putCell(7, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", s.L0);          putCell(8, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", s.Lfinal);      putCell(9, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", s.stepLen);     putCell(10, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", s.rawStepLen);  putCell(11, dataRow, buf, rowColor);
        dataRow++;
    }

    // --- Old Steps Section ---
    dataRow += 1;
    cv::line(img, {10, tableTop + 15 + (dataRow - 1) * ROW_H + 3},
             {W - 10, tableTop + 15 + (dataRow - 1) * ROW_H + 3}, cv::Scalar(60, 60, 60), 1);
    cv::putText(img, "OLD Steps", {15, tableTop + 15 + dataRow * ROW_H},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 150, 50), 1);
    dataRow++;

    for (int c = 0; c < NUM_COLS; ++c) {
        putCell(c, dataRow, colHeaders[c], cv::Scalar(160, 160, 160));
    }
    cv::line(img, {10, tableTop + 15 + dataRow * ROW_H + 3},
             {W - 10, tableTop + 15 + dataRow * ROW_H + 3}, cv::Scalar(60, 60, 60), 1);
    dataRow++;

    // Old steps data (last 8)
    int nOld = std::min(upToOldStep, (int)oldSteps.size());
    int oldShowStart = std::max(0, nOld - 8);
    for (int i = oldShowStart; i < nOld; ++i) {
        if (tableTop + 15 + dataRow * ROW_H + 5 > H - 5) break;
        const StepResult& o = oldSteps[i];
        bool isLatest = (i == nOld - 1);
        cv::Scalar rowColor = isLatest ? cv::Scalar(100, 220, 255) : cv::Scalar(220, 180, 140);

        if (isLatest) {
            cv::rectangle(img, {10, tableTop + 15 + dataRow * ROW_H - 12},
                          {W - 10, tableTop + 15 + dataRow * ROW_H + 5},
                          cv::Scalar(60, 50, 40), cv::FILLED);
        }

        snprintf(buf, sizeof(buf), "%d", o.stepNum);       putCell(0, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.2f", o.time_s);      putCell(1, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", o.ampZ);        putCell(2, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", o.fourthRoot);  putCell(3, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", o.correctedRoot); putCell(4, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.0f", o.durationMs);  putCell(5, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.2f", o.freqHz);      putCell(6, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", o.kf);          putCell(7, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", o.L0);          putCell(8, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", o.Lfinal);      putCell(9, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", o.stepLen);     putCell(10, dataRow, buf, rowColor);
        snprintf(buf, sizeof(buf), "%.3f", o.rawStepLen);  putCell(11, dataRow, buf, rowColor);
        dataRow++;
    }

    cv::imshow("Step Analysis", img);
}

// ============================================================
//  7. Controls 요약 창
// ============================================================
void showControlsWindow(const MapViewState& viewState, int stride) {
    const int W = 310, H = 500;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(35, 35, 35));

    int y = 0;
    auto header = [&](const std::string& text) {
        y += 28;
        cv::putText(img, text, {12, y}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 220, 255), 2);
        y += 6;
    };
    auto row = [&](const std::string& key, const std::string& desc) {
        y += 20;
        cv::putText(img, key, {16, y}, cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(120, 255, 120), 1);
        cv::putText(img, desc, {100, y}, cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(200, 200, 200), 1);
    };
    auto separator = [&]() {
        y += 10;
        cv::line(img, {12, y}, {W - 12, y}, cv::Scalar(70, 70, 70), 1);
        y += 4;
    };

    header("Playback");
    row("Space",  "Play / Pause");
    row("A / D",  "Prev / Next Frame");
    row("W / S",  "Stride Up / Down");

    separator();
    header("Map View");
    row("1 / 2",   "Toggle New / Old Path");
    row("I/J/K/L", "Move Start Point");
    row("U / O",   "Rotate Path +/-1 deg");
    row("R",       "Rotate Map 90 deg");
    row("H",       "Flip Horizontal");
    row("V",       "Flip Vertical");

    separator();
    header("File / Analysis");
    row("F",  "Open PDR Data File");
    row("M",  "Open Map File");
    row("T",  "Toggle Analysis");
    row("B",  "Toggle Step Analysis");

    separator();
    row("Click",   "Inspect Waveform");
    row("Q / Esc", "Quit");

    // --- 현재 상태 ---
    separator();
    y += 6;
    char buf[128];
    cv::putText(img, "Current State", {12, y += 18}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(200, 200, 100), 1);

    snprintf(buf, sizeof(buf), "Stride: %d", stride);
    cv::putText(img, buf, {16, y += 20}, cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(180, 180, 180), 1);

    snprintf(buf, sizeof(buf), "Path Rot: %d deg", (int)viewState.pdrRotateDeg);
    cv::putText(img, buf, {16, y += 18}, cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(180, 180, 180), 1);

    snprintf(buf, sizeof(buf), "Map Rot: %d deg", viewState.rotate90 * 90);
    cv::putText(img, buf, {16, y += 18}, cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(180, 180, 180), 1);

    std::string flipStr;
    if (viewState.flipX) flipStr += "X ";
    if (viewState.flipY) flipStr += "Y ";
    if (flipStr.empty()) flipStr = "None";
    cv::putText(img, ("Flip: " + flipStr).c_str(), {16, y += 18}, cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(180, 180, 180), 1);

    std::string pathStr;
    if (viewState.showNewPath) pathStr += "New ";
    if (viewState.showOldPath) pathStr += "Old ";
    if (pathStr.empty()) pathStr = "None";
    cv::putText(img, ("Paths: " + pathStr).c_str(), {16, y += 18}, cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(180, 180, 180), 1);

    cv::imshow("Controls", img);
}