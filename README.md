# Indoor-Navigation-Sim

실내 네비게이션을 위한 PDR(보행자 관성 항법) 및 IMU 기반 시뮬레이션 프로젝트 모음

## 프로젝트 구조

```
Indoor-Navigation-Sim/
├── PDR_simulation/                    # PDR 알고리즘 비교 시뮬레이터
├── IMU_Floor_deteciton_simulation/    # IMU 기반 층 감지 시뮬레이터
└── zeromap-simulation/                # 나침반 기반 PDR 시뮬레이션
```

## 프로젝트 설명

### 1. PDR_simulation (PDR 알고리즘 비교 시뮬레이터)

보행자 관성 항법 알고리즘의 **신규 버전**과 **기존 버전**을 동일한 센서 데이터에 대해 동시에 실행하고, 결과를 시각적으로 비교 분석하는 데스크톱 시뮬레이터입니다.

- **듀얼 파이프라인:** 개선된 알고리즘 vs 기존 Android 원본 알고리즘 동시 실행
- **시각화:** OpenCV 기반 3개 윈도우 (PDR Dashboard, World Map, Analysis)
- **기술:** C++17, CMake, OpenCV 4.x

### 2. IMU_Floor_deteciton_simulation (IMU 층 감지 시뮬레이터)

IMU 센서 데이터를 활용한 실시간 계단/평지 감지 알고리즘 시뮬레이터입니다.

- **Weighted Scoring FSM** 기반 계단/평지 판정
- **보행 패턴 분류:** 일반/살살/거친 걸음
- **Android 이식 가능:** 감지 로직이 OpenCV 무의존으로 설계
- **참고:** `include/`, `src/`, `mock/` 디렉토리는 `PDR_simulation/`으로의 심링크 (공유 코드)

### 3. zeromap-simulation (나침반 기반 PDR 시뮬레이션)

나침반 방위각 기반 PDR 경로 추정 시뮬레이터입니다.

- **지원 건물:** 고려대 공학관, 아남병원
- **JSON 맵 데이터** 기반 실내 지도 렌더링
- **기술:** C++17, CMake, OpenCV 4.x, OpenMP, nlohmann/json

## 빌드 방법

### 요구사항
- C++17, CMake 3.14+
- OpenCV 4.x

### PDR_simulation
```bash
cd PDR_simulation
mkdir -p build && cd build
cmake .. && make -j$(nproc)
./pdr_sim
```

### IMU_Floor_deteciton_simulation
```bash
cd IMU_Floor_deteciton_simulation
mkdir -p build && cd build
cmake .. && make -j$(nproc)
./floor_sim ../data/Floor_sim_data/Floor_sim_data_sample.txt
```

### zeromap-simulation
```bash
cd zeromap-simulation
mkdir -p build && cd build
cmake .. && make -j$(nproc)
./MainApp
```

## 관련 프로젝트

- [KU-LAB](https://github.com/luke0911/KU-LAB) - 다중 건물 실내 네비게이션 시스템
- [PDR](https://github.com/luke0911/PDR) - PDR 알고리즘 개발
