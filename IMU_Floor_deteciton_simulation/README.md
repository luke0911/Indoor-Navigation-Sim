ㅂㅂ# IMU Floor Detection Simulation

IMU 센서 데이터를 활용한 **실시간 계단/평지 감지 알고리즘** 시뮬레이션 도구.

OpenCV 기반 시각화로 알고리즘 동작을 실시간으로 확인하고, 다양한 보행 패턴(일반/살살/거친 걸음)에서 감지 성능을 검증할 수 있다.

---

## 아키텍처

```
main.cpp          ─ 데이터 로딩, 스텝 감지, 시뮬레이션 루프
imu_floor_detection.h/cpp  ─ 계단 감지 알고리즘 (Android 이식 가능, OpenCV 무의존)
viz.h/cpp         ─ 3개 윈도우 시각화 (OpenCV)
include/src/mock/ ─ PDR 스텝 감지 공유 코드 (../PDR_simulation 심링크)
```

### 핵심 설계 원칙
- `imu_floor_detection.h/cpp`는 `<string>`, `<cmath>`, `<vector>`, `<deque>`만 사용 — Android NDK에 그대로 복사 가능
- 시각화(`viz.*`)와 감지 로직(`imu_floor_detection.*`)이 완전 분리

---

## 빌드 & 실행

### 요구사항
- C++17, CMake 3.14+
- OpenCV 4.x
- PDR_simulation 프로젝트가 같은 레벨(`../PDR_simulation`)에 존재

### 빌드
```bash
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### 실행 (GUI 모드)
```bash
./floor_sim ../data/Floor_sim_data/Floor_sim_data_sample.txt
```
파일 인자 없이 실행하면 macOS 파일 선택 다이얼로그가 열린다.

### 실행 (배치 모드)
```bash
./floor_sim ../data/Floor_sim_data/Floor_sim_data_일반보행.txt --batch
```
GUI 없이 전체 데이터를 처리하고 `data/output/` 에 TSV 결과를 저장한다.

---

## 데이터 포맷

탭 구분 텍스트(TSV), 헤더 필수:

```
time(s)  globalX  globalY  globalZ  magX  magY  magZ  gyroAngle  pitch  floorLabel
0.010    0.076    0.100    0.046    19.69 -6.32 -41.66  45.00    -3.90  ground
```

| 컬럼 | 설명 |
|------|------|
| `time(s)` | 타임스탬프 (초) |
| `globalX/Y/Z` | 글로벌 좌표계 가속도 (m/s^2) |
| `magX/Y/Z` | 자기장 (선택) |
| `gyroAngle` | 방위각 (도) |
| `pitch` | 기울기 (도, 선택 - 쿼터니언에서 자동 계산 가능) |
| `floorLabel` | GT 라벨: `ground`, `up`, `down` |

---

## 감지 알고리즘: Weighted Scoring FSM

가중치 점수 기반 휴리스틱. 각 피처가 배점만큼 점수를 내고, 총점이 커트라인(60점)을 넘으면 계단으로 판정한다.

### 신호 전처리

```
[Global Accel] → HPF(α=0.995) → pureX, pureY, pureZ
                                      │
                     gyroAngle ────────┤
                                      ▼
                              회전 변환 (Z축 회전)
                                      │
                              localForwardY (내 몸 앞방향)
                              localLateralX (내 몸 좌우)
                                      │
                              Envelope LPF(α=0.95)
                                      │
                              envZ = LPF(|pureZ|)
                              envH = LPF(magH)
```

**Local Forward 변환**: `gyroAngle`로 글로벌 X/Y를 몸 기준 좌표로 회전. 사용자가 어느 방향으로 걷든 일관된 전진 가속도를 제공한다.

### 점수 구성 (100점 만점)

| 피처 | 배점 | 의미 | 계단 특성 |
|------|------|------|-----------|
| **stairScore** (envZ - envH) | 40 | 수직 vs 수평 흔들림 비율 | 양수 (수직 우세) |
| **varYZRatio** | 30 | 최근 5스텝 시작점 분산 / baseline | < 0.85 (궤적 뭉침) |
| **dirR1** | 20 | 최근 5스텝 방향 일관성 [0,1] | > 0.85 (직진) |
| **stepVarH** | 10 | 수평 가속도 분산 | < 1.5 (수평 에너지 감소) |

| 페널티 | 점수 | 조건 |
|--------|------|------|
| 거친 보행 | -30 | stepVarH > 2.5 |
| 잰걸음 | -20 | rmsZdivFreq < 0.8 |

**커트라인**: 60점 이상 → `"stair"`, 미만 → `"ground"`

### FSM Hysteresis

2-step FSM: 연속 2걸음 같은 판정이 나와야 상태 변경. baseline 워밍업 2스텝.

### 피처별 Cohen's d (3개 보행 패턴 검증)

| 피처 | 일반보행 | 살살걸음 | 거칠게걸음 |
|------|----------|----------|------------|
| stairScore | 0.96 | 0.36 | 2.13 |
| varYZRatio | 0.84 | 0.73 | 1.03 |
| dirR1 | 0.74 | 0.64 | 0.78 |
| stepVarH | 0.78 | 1.04 | 1.28 |

단일 피처로는 모든 보행을 커버하지 못하지만, 점수 합산으로 상호 보완한다.

---

## 시각화 윈도우

### Window 1: Floor Detection (메인)
```
 Row 0a: pureZ 파형 + GT 배경 밴드
 Row 0b: localForwardY 파형 + GT 배경 밴드
 Row 1 : envZ(파랑) / envH(주황) / stairScore(초록)
 Row 2 : GT 배경 + pureZ + 스텝 마커
 Row 3 : GT vs Prediction 컬러 스트립
 Row 3b: 전체 데이터 GT 미니맵
 Row 4 : Info bar (파일명, 시간, 피처값)
 Row 5 : 컨트롤 힌트
```

### Window 2: Analysis
- 상단: 2D Scatter (localFwdY vs pureZ, 스텝별 궤적)
- 중단: Step Start Points 산점도 (GT 색상)
- 하단: 피처 상태바

### Window 3: Step Features (Weighted Scoring)
```
 Row 0: stairScore per step      (threshold 0.15, 40pt)
 Row 1: varYZRatio per step      (threshold 0.85, 30pt)
 Row 2: dirR1 per step           (threshold 0.85, 20pt)
 Row 3: Total Score per step     (cutline 60)
 Status: 점수 분해 + FSM 상태
```

### 색상 규칙
- Ground: 초록
- Up/Stair: 빨강
- Down: 파랑

---

## 키보드 조작

| 키 | 동작 |
|----|------|
| `Space` | 실시간 재생 / 일시정지 |
| `A` / `D` | 이전 / 다음 프레임 (스크럽) |
| `W` / `S` | 스크럽 보폭 증가 / 감소 |
| `1` `2` `5` `0` | 재생 속도 1x / 2x / 5x / 0.5x |
| `R` | 녹화 시작/중지 (TSV 출력) |
| `F` | 새 데이터 파일 열기 |
| `Q` / `Esc` | 종료 |

---

## 출력 파일

`data/output/` 디렉토리에 자동 저장:

| 파일 | 생성 시점 | 내용 |
|------|-----------|------|
| `batch_*.tsv` | `--batch` 모드 | 매 샘플 전체 피처 (localFwdY, localLatX 포함) |
| `step_analysis_*.tsv` | 재생 완료 / 배치 | 스텝별 분석 결과 |
| `processed_*.txt` | `R` 키 녹화 | 실시간 녹화 데이터 |

---

## 프로젝트 구조

```
IMU_Floor_deteciton_simulation/
├── CMakeLists.txt
├── README.md
├── imu_floor_detection.h    # 감지 알고리즘 헤더 (Android 이식용)
├── imu_floor_detection.cpp  # 감지 알고리즘 구현
├── main.cpp                 # 시뮬레이션 메인 루프
├── viz.h                    # 시각화 구조체 & 함수 선언
├── viz.cpp                  # 3개 윈도우 시각화 구현
├── include/ -> ../PDR_simulation/include
├── src/     -> ../PDR_simulation/src
├── mock/    -> ../PDR_simulation/mock
├── build/                   # 빌드 출력
├── data/
│   ├── Floor_sim_data/      # 입력 데이터
│   │   ├── Floor_sim_data_sample.txt
│   │   ├── Floor_sim_data_일반보행.txt
│   │   ├── Floor_sim_data_살살걸음.txt
│   │   └── Floor_sim_data_거칠게 걸음.txt
│   └── output/              # 분석 결과 TSV
└── analysis/                # Python 분석 스크립트
```
