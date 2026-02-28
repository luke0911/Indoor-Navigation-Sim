#ifndef SIM_DATA_LOGGER_H
#define SIM_DATA_LOGGER_H

// =========================================================
// SimDataLogger — PDR 시뮬레이션 재생용 센서 데이터 기록기
//
// 기록 양식: time(s)\tglobalZ\tgyroAngle
// 출력 파일: <basePath>/pdr_data/sim_data_YYMMDD-HHMMSS.txt
//
// PDR_EXT 모듈에 포함되어 있으므로 어떤 프로젝트든
// 이 모듈을 끼우면 자동으로 기록됩니다.
//
// 사용법 (C++, onHand.cpp 등에서 자동 호출):
//   SimDataLog_write(timestampMs, globalZ, gyroAngleDeg);
//
// 수동 제어 (JNI를 통해 Java/Kotlin에서도 호출 가능):
//   SimDataLog_start(basePath);   // 기록 시작
//   SimDataLog_stop();            // 기록 종료
//   SimDataLog_isEnabled();       // 상태 확인
// =========================================================

#include <cstdint>

// 매 센서 샘플마다 호출 — 첫 호출 시 auto-start 시도
void SimDataLog_write(int64_t timestampMs, double globalZ, double gyroAngleDeg);

// 수동 시작 (basePath = 앱의 files 디렉터리)
void SimDataLog_start(const char* basePath);

// 기록 종료 (flush + close)
void SimDataLog_stop();

// 현재 기록 중인지 확인
bool SimDataLog_isEnabled();

#endif // SIM_DATA_LOGGER_H
