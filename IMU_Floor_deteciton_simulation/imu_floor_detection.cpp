#include "imu_floor_detection.h"
#include <algorithm>

IMUFloorDetection::IMUFloorDetection()
    : sampleCount_(0)
    , baseX_(0.0), baseY_(0.0), baseZ_(0.0), hpfInit_(false)
    , lastProcStepTime_(-1.0), procStepCount_(0)
    , lastEnergyFracZ_(0.0), lastProcRmsZ_(0.0), lastProcStepFreq_(0.0)
    , lastRmsZdivFreq_(0.0), lastStepVarH_(0.0)
    , ewmaVarH_(0.0), ewmaInit_(false), lastArmSpinFlag_(false)
    , envZ_(0.0), envH_(0.0)
    , fsmState_("ground"), fsmCandState_("ground"), fsmStreak_(0)
    , lastZPeakYEnergy_(0.0), lastZPeakYRatio_(0.0)
    , sumSqX_(0.0), sumSqY_(0.0), sumSqZ_(0.0), segCount_(0)
    , lastRmsX_(0.0), lastRmsY_(0.0), lastRmsZ_(0.0)
    , lastStepFreqHz_(0.0)
    , lastDmag_(0.0), lastRecentStepVarYZ_(99.0)
    , ewmaBaseVarYZ_(0.0), ewmaBaseDmag_(0.0), ewmaBaseZdivFreq_(0.0), ewmaBaseFreqHz_(0.0)
    , baselineInit_(false), baselineStepCount_(0)
    , lastVarYZRatio_(1.0), lastDmagRatio_(1.0), lastZfRatio_(1.0), lastFreqRatio_(1.0)
    , smoothVarYZRatio_(1.0), smoothZfRatio_(1.0)
    , lastStairConfidence_(0.0), lastTotalStairScore_(0.0)
{
}

void IMUFloorDetection::reset() {
    sampleCount_ = 0;

    baseX_ = baseY_ = baseZ_ = 0.0;
    hpfInit_ = false;

    stepBuf_.clear();
    lastProcStepTime_ = -1.0;
    procStepCount_ = 0;
    lastEnergyFracZ_ = lastProcRmsZ_ = lastProcStepFreq_ = 0.0;
    lastRmsZdivFreq_ = lastStepVarH_ = 0.0;

    ewmaVarH_ = 0.0;
    ewmaInit_ = false;
    lastArmSpinFlag_ = false;

    envZ_ = envH_ = 0.0;

    fsmState_ = "ground";
    fsmCandState_ = "ground";
    fsmStreak_ = 0;

    zPeakBuf_.clear();
    lastZPeakYEnergy_ = lastZPeakYRatio_ = 0.0;

    sumSqX_ = sumSqY_ = sumSqZ_ = 0.0;
    segCount_ = 0;
    lastRmsX_ = lastRmsY_ = lastRmsZ_ = 0.0;
    lastStepFreqHz_ = 0.0;

    recentStepStarts_.clear();
    lastDmag_ = 0.0;
    lastRecentStepVarYZ_ = 99.0;

    ewmaBaseVarYZ_ = ewmaBaseDmag_ = ewmaBaseZdivFreq_ = ewmaBaseFreqHz_ = 0.0;
    baselineInit_ = false;
    baselineStepCount_ = 0;
    lastVarYZRatio_ = lastDmagRatio_ = lastZfRatio_ = lastFreqRatio_ = 1.0;
    smoothVarYZRatio_ = smoothZfRatio_ = 1.0;
    lastStairConfidence_ = 0.0;
    lastTotalStairScore_ = 0.0;

    recentDirAngles_.clear();
    lastDirAngle1_ = 0.0;
    lastDirR1_ = 0.5;
    smoothDirR1_ = 0.5;
}

FloorDetectionResult IMUFloorDetection::update(const FloorDetectionInput& input) {
    sampleCount_++;

    // ── 1. HPF: 순수 동적 가속도 계산 (3축 모두) ──
    if (!hpfInit_) {
        baseX_ = input.globalX;
        baseY_ = input.globalY;
        baseZ_ = input.globalZ;
        hpfInit_ = true;
    }
    const double hpfAlpha = 0.995;
    baseX_ = hpfAlpha * baseX_ + (1.0 - hpfAlpha) * input.globalX;
    baseY_ = hpfAlpha * baseY_ + (1.0 - hpfAlpha) * input.globalY;
    baseZ_ = hpfAlpha * baseZ_ + (1.0 - hpfAlpha) * input.globalZ;

    double pureX = input.globalX - baseX_;
    double pureY = input.globalY - baseY_;
    double pureZ = input.globalZ - baseZ_;

    // ── 1a. 회전 변환: Global → Local (내 배꼽 앞방향) 보정 ──
    double thetaRad = input.gyroAngleDeg * (M_PI / 180.0);
    double localForwardY =  pureY * std::cos(thetaRad) + pureX * std::sin(thetaRad);
    double localLateralX = -pureY * std::sin(thetaRad) + pureX * std::cos(thetaRad);

    // ── 1b. Envelope LPF (매 샘플, 실시간) ──
    double absZ = std::abs(pureZ);
    double magH = std::sqrt(pureX * pureX + pureY * pureY);
    const double envAlpha = 0.95;
    envZ_ = envAlpha * envZ_ + (1.0 - envAlpha) * absZ;
    envH_ = envAlpha * envH_ + (1.0 - envAlpha) * magH;

    // ── 2. 스텝 버퍼에 push (local 보정 좌표 사용) ──
    stepBuf_.push_back({input.timeSec, localLateralX, localForwardY, pureZ});

    // ── 3. Z-peak 분석용 버퍼 (최근 2초 유지) ──
    zPeakBuf_.push_back({input.timeSec, localLateralX, localForwardY, pureZ});
    while (!zPeakBuf_.empty() && (input.timeSec - zPeakBuf_.front().t) > 2.0) {
        zPeakBuf_.pop_front();
    }

    // ── Legacy RMS 누적 ──
    sumSqX_ += input.globalX * input.globalX;
    sumSqY_ += input.globalY * input.globalY;
    sumSqZ_ += input.globalZ * input.globalZ;
    segCount_++;

    // ── 4. 스텝 감지 시: 피처 계산 ──
    if (input.isStep && stepBuf_.size() >= 2) {
        int N = (int)stepBuf_.size();
        double n = (double)N;

        // (a) rmsZ = sqrt(mean(pureZ^2))
        double sumPZ2 = 0.0;
        for (int i = 0; i < N; ++i) {
            sumPZ2 += stepBuf_[i].pZ * stepBuf_[i].pZ;
        }
        lastProcRmsZ_ = std::sqrt(sumPZ2 / n);

        // (b) varX, varY, varZ → energyFracZ = varZ / (varAll + eps)
        double sumX = 0.0, sumX2 = 0.0;
        double sumY = 0.0, sumY2 = 0.0;
        double sumZ = 0.0, sumZ2 = 0.0;
        for (int i = 0; i < N; ++i) {
            sumX += stepBuf_[i].pX; sumX2 += stepBuf_[i].pX * stepBuf_[i].pX;
            sumY += stepBuf_[i].pY; sumY2 += stepBuf_[i].pY * stepBuf_[i].pY;
            sumZ += stepBuf_[i].pZ; sumZ2 += stepBuf_[i].pZ * stepBuf_[i].pZ;
        }
        double varX = (sumX2 / n) - (sumX / n) * (sumX / n);
        double varY = (sumY2 / n) - (sumY / n) * (sumY / n);
        double varZ = (sumZ2 / n) - (sumZ / n) * (sumZ / n);
        if (varX < 0.0) varX = 0.0;
        if (varY < 0.0) varY = 0.0;
        if (varZ < 0.0) varZ = 0.0;
        double varAll = varX + varY + varZ;
        lastEnergyFracZ_ = varZ / (varAll + 1e-9);

        // (c) stepFreq = steps / timeDelta
        procStepCount_++;
        if (lastProcStepTime_ >= 0.0) {
            double dt = input.timeSec - lastProcStepTime_;
            if (dt > 0.01) {
                lastProcStepFreq_ = 1.0 / dt;  // Hz (steps per second)
            }
        }
        lastProcStepTime_ = input.timeSec;

        // (d) Arm-spin detection: EWMA of varH (horizontal variance)
        double varH = varX + varY;
        const double ewmaAlpha = 0.3;
        if (!ewmaInit_) {
            ewmaVarH_ = varH;
            ewmaInit_ = true;
        } else {
            ewmaVarH_ = ewmaAlpha * varH + (1.0 - ewmaAlpha) * ewmaVarH_;
        }
        // arm-spin 감지: 수평 분산이 수직 분산의 3배 이상이면 arm-spin
        lastArmSpinFlag_ = (varH > varZ * 3.0) && (ewmaVarH_ > 2.0);

        // (e-1) stepVarH 저장 (수평 에너지 — 보조 게이트)
        lastStepVarH_ = varH;

        // (e-2) Z-peak aligned Y analysis (FSM 보다 먼저 계산 — 보조 피처)
        if (!zPeakBuf_.empty()) {
            double maxAbsZ = 0.0;
            double peakTime = zPeakBuf_.back().t;
            for (const auto& s : zPeakBuf_) {
                double absZ = std::abs(s.pZ);
                if (absZ > maxAbsZ) {
                    maxAbsZ = absZ;
                    peakTime = s.t;
                }
            }

            double yEnergy = 0.0;
            double totalEnergy = 0.0;
            int cnt = 0;
            for (const auto& s : zPeakBuf_) {
                totalEnergy += s.pY * s.pY;
                if (std::abs(s.t - peakTime) <= 0.1) {
                    yEnergy += s.pY * s.pY;
                    cnt++;
                }
            }
            lastZPeakYEnergy_ = yEnergy;
            lastZPeakYRatio_ = (totalEnergy > 1e-9) ? (yEnergy / totalEnergy) : 0.0;
        }

        // Legacy RMS
        if (segCount_ > 0) {
            double ns = (double)segCount_;
            lastRmsX_ = std::sqrt(sumSqX_ / ns);
            lastRmsY_ = std::sqrt(sumSqY_ / ns);
            lastRmsZ_ = std::sqrt(sumSqZ_ / ns);
        }
        sumSqX_ = sumSqY_ = sumSqZ_ = 0.0;
        segCount_ = 0;

        // Peak-to-peak frequency (onHand step detection 기반)
        if (input.stepPeakToPeakMs > 1.0) {
            lastStepFreqHz_ = 1000.0 / input.stepPeakToPeakMs;
        }

        // (e-3) rmsZdivFreq = stepRmsZ / stepFreqHz
        if (lastStepFreqHz_ > 0.1) {
            lastRmsZdivFreq_ = lastProcRmsZ_ / lastStepFreqHz_;
        } else {
            lastRmsZdivFreq_ = 0.0;
        }

        // (e-4) 스텝 시작점 추적: dmag + recentStepVarYZ
        {
            StepStartPt curStart{localForwardY, pureZ};

            // dmag: 이전 스텝 시작점과의 YZ 거리
            if (!recentStepStarts_.empty()) {
                double dy = curStart.pY - recentStepStarts_.back().pY;
                double dz = curStart.pZ - recentStepStarts_.back().pZ;
                lastDmag_ = std::sqrt(dy * dy + dz * dz);
            }

            // 버퍼에 추가 (최근 5개 유지)
            recentStepStarts_.push_back(curStart);
            while (recentStepStarts_.size() > 5) {
                recentStepStarts_.pop_front();
            }

            // recentStepVarYZ: 최근 스텝 시작점의 var(pY) + var(pZ)
            if (recentStepStarts_.size() >= 3) {
                int m = (int)recentStepStarts_.size();
                double sY = 0, sY2 = 0, sZ = 0, sZ2 = 0;
                for (const auto& pt : recentStepStarts_) {
                    sY += pt.pY; sY2 += pt.pY * pt.pY;
                    sZ += pt.pZ; sZ2 += pt.pZ * pt.pZ;
                }
                double vY = (sY2 / m) - (sY / m) * (sY / m);
                double vZ = (sZ2 / m) - (sZ / m) * (sZ / m);
                if (vY < 0) vY = 0;
                if (vZ < 0) vZ = 0;
                lastRecentStepVarYZ_ = vY + vZ;
            }
        }

        // (e-4b) 궤적 방향 일관성 (Direction Consistency R)
        //   stepBuf_의 전반부 방향각을 최근 5스텝 rolling → Mean Resultant Length
        {
            int bufN = (int)stepBuf_.size();
            int mid = bufN / 2;
            if (bufN >= 4 && mid > 0) {
                double dy = stepBuf_[mid].pY - stepBuf_[0].pY;
                double dz = stepBuf_[mid].pZ - stepBuf_[0].pZ;
                if (std::abs(dy) > 1e-9 || std::abs(dz) > 1e-9) {
                    lastDirAngle1_ = std::atan2(dz, dy);
                    recentDirAngles_.push_back(lastDirAngle1_);
                    if (recentDirAngles_.size() > 5)
                        recentDirAngles_.pop_front();
                }
            }
            if (recentDirAngles_.size() >= 3) {
                double cs = 0.0, sn = 0.0;
                for (auto a : recentDirAngles_) {
                    cs += std::cos(a);
                    sn += std::sin(a);
                }
                int n = (int)recentDirAngles_.size();
                lastDirR1_ = std::sqrt((cs / n) * (cs / n) + (sn / n) * (sn / n));
            }
            const double dirSmooth = 0.30;
            smoothDirR1_ = dirSmooth * lastDirR1_ + (1.0 - dirSmooth) * smoothDirR1_;
        }

        // (e-5) Envelope + rmsZ/freq 기반 계단 감지
        //
        // 원리: stairScore = envZ - envH
        //   계단: 수직(Z) 흔들림이 수평(H) 흔들림보다 큼 → stairScore > 0
        //   평지: 수평 흔들림이 크거나 비슷 → stairScore ≈ 0 또는 음수
        //   → 1걸음 궤적 모양만으로 즉시 판별 가능
        //
        // rmsZdivFreq 보조: 계단에서 Z에너지 상승 + 주파수 하락 → 값 상승
        {
            // Baseline 통계는 분석/시각화용으로 계속 업데이트
            baselineStepCount_++;
            bool varYZValid = (recentStepStarts_.size() >= 3);
            bool isWarmup = (baselineStepCount_ <= 2);
            bool shouldUpdate = isWarmup || (fsmState_ == "ground");
            double alpha = isWarmup ? 0.50 : 0.10;

            if (!baselineInit_) {
                ewmaBaseVarYZ_    = varYZValid ? lastRecentStepVarYZ_ : 0.1;
                ewmaBaseDmag_     = std::max(lastDmag_, 0.01);
                ewmaBaseZdivFreq_ = std::max(lastRmsZdivFreq_, 0.1);
                ewmaBaseFreqHz_   = std::max(lastStepFreqHz_, 0.5);
                baselineInit_ = true;
            } else if (shouldUpdate) {
                if (varYZValid)
                    ewmaBaseVarYZ_ = alpha * lastRecentStepVarYZ_ + (1.0 - alpha) * ewmaBaseVarYZ_;
                ewmaBaseDmag_     = alpha * std::max(lastDmag_, 0.001) + (1.0 - alpha) * ewmaBaseDmag_;
                ewmaBaseZdivFreq_ = alpha * lastRmsZdivFreq_   + (1.0 - alpha) * ewmaBaseZdivFreq_;
                ewmaBaseFreqHz_   = alpha * lastStepFreqHz_     + (1.0 - alpha) * ewmaBaseFreqHz_;
            }

            // Self-normalizing ratios (시각화/분석용)
            lastVarYZRatio_ = (varYZValid && ewmaBaseVarYZ_ > 1e-6)
                            ? lastRecentStepVarYZ_ / ewmaBaseVarYZ_ : 1.0;
            lastDmagRatio_  = (ewmaBaseDmag_ > 1e-6) ? lastDmag_ / ewmaBaseDmag_ : 1.0;
            lastZfRatio_    = (ewmaBaseZdivFreq_ > 1e-6) ? lastRmsZdivFreq_ / ewmaBaseZdivFreq_ : 1.0;
            lastFreqRatio_  = (ewmaBaseFreqHz_ > 1e-6) ? lastStepFreqHz_ / ewmaBaseFreqHz_ : 1.0;
            const double smoothAlpha = 0.25;
            smoothVarYZRatio_ = smoothAlpha * lastVarYZRatio_ + (1.0 - smoothAlpha) * smoothVarYZRatio_;
            smoothZfRatio_    = smoothAlpha * lastZfRatio_    + (1.0 - smoothAlpha) * smoothZfRatio_;

            // ── ★ 가중치 점수 기반 계단 예측 (Weighted Scoring FSM) ★ ──

            double totalStairScore = 0.0;
            double stairScore = envZ_ - envH_;
            std::string newVote = "ground";

            if (!lastArmSpinFlag_) {
                // 1. 핵심 물리 지표 (배점 40): 위로 솟구치되 앞으로는 안 나가는가?
                if (stairScore > 0.15)      totalStairScore += 40.0;
                else if (stairScore > 0.05) totalStairScore += 20.0;

                // 2. 궤적 집중도 지표 (배점 30): 보폭이 일정하게 좁혀졌는가?
                if (recentStepStarts_.size() >= 3) {
                    if (lastVarYZRatio_ < 0.6)       totalStairScore += 30.0;
                    else if (lastVarYZRatio_ < 0.85)  totalStairScore += 15.0;
                }

                // 3. 방향 일관성 지표 (배점 20): 한 방향으로 올곧게 진행하는가?
                if (recentDirAngles_.size() >= 3) {
                    if (lastDirR1_ > 0.90)      totalStairScore += 20.0;
                    else if (lastDirR1_ > 0.80)  totalStairScore += 10.0;
                }

                // 4. 수평 안정성 지표 (배점 10): 몸을 불필요하게 흔들지 않는가?
                if (lastStepVarH_ < 1.5) totalStairScore += 10.0;

                // 치명적 페널티 (거친 보행 / 폰 흔들기 방어막)
                if (lastStepVarH_ > 2.5)
                    totalStairScore -= 30.0;
                if (lastRmsZdivFreq_ < 0.8)
                    totalStairScore -= 20.0;

                // 최종 판정 (커트라인 60점)
                if (totalStairScore >= 60.0) {
                    newVote = "stair";
                }
            }

            // Confidence: 총점을 100점 만점 기준으로 환산
            lastStairConfidence_ = std::max(0.0, std::min(totalStairScore / 100.0, 1.0));
            lastTotalStairScore_ = totalStairScore;

            // 초고속 2-step FSM hysteresis
            if (newVote == fsmCandState_) {
                fsmStreak_++;
            } else {
                fsmCandState_ = newVote;
                fsmStreak_ = 1;
            }
            if (fsmStreak_ >= 2) {
                fsmState_ = fsmCandState_;
            }
        }

        // 스텝 버퍼 클리어 (마지막 샘플 유지)
        PureSample last = stepBuf_.back();
        stepBuf_.clear();
        stepBuf_.push_back(last);
    }

    // ── 결과 구성 ──
    FloorDetectionResult result;
    result.predictedLabel = fsmState_;
    result.confidence = std::min(std::max(lastStairConfidence_, 0.0), 1.0);

    result.pureX = pureX;
    result.pureY = pureY;
    result.pureZ = pureZ;
    result.localForwardY = localForwardY;
    result.localLateralX = localLateralX;

    // 핵심 피처
    result.stepRmsZ = lastProcRmsZ_;
    result.stepFreqHz = lastStepFreqHz_;
    result.rmsZdivFreq = lastRmsZdivFreq_;
    result.stepVarH = lastStepVarH_;

    // 스텝 시작점 피처
    result.dmag = lastDmag_;
    result.recentStepVarYZ = lastRecentStepVarYZ_;

    // Self-normalizing ratios
    result.varYZRatio = lastVarYZRatio_;
    result.dmagRatio = lastDmagRatio_;
    result.zfRatio = lastZfRatio_;
    result.freqRatio = lastFreqRatio_;
    result.stairConfidence = lastStairConfidence_;
    result.totalStairScore = lastTotalStairScore_;

    // 궤적 방향 일관성
    result.dirAngle1 = lastDirAngle1_;
    result.dirR1 = lastDirR1_;

    // 보조 피처
    result.energyFracZ = lastEnergyFracZ_;
    result.zPeakYRatio = lastZPeakYRatio_;

    // Envelope (참고용)
    result.envZ = envZ_;
    result.envH = envH_;
    result.stairScore = envZ_ - envH_;

    // FSM
    result.fsmLabel = fsmState_;
    result.armSpinFlag = lastArmSpinFlag_;

    // legacy
    result.procStepFreq = lastProcStepFreq_;

    result.rmsX = lastRmsX_;
    result.rmsY = lastRmsY_;
    result.rmsZ = lastRmsZ_;
    result.stepFreqHz = lastStepFreqHz_;

    return result;
}
