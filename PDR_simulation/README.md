# PDR Simulation

보행자 관성 항법(Pedestrian Dead Reckoning) 알고리즘의 **신규 버전**과 **기존 버전**을 동일한 센서 데이터에 대해 동시에 실행하고, 결과를 시각적으로 비교 분석할 수 있는 데스크톱 시뮬레이터입니다.

Android 앱([zeromap-korea_university_campus-android](https://github.com/anthropics/zeromap-korea_university_campus-android))에서 사용하는 PDR 알고리즘을 PC 환경에서 검증하고 개선하기 위한 목적으로 제작되었습니다.

## 핵심 기능

### 듀얼 파이프라인 실행

하나의 센서 데이터 파일을 입력으로 두 알고리즘을 동시에 구동합니다:

- **New Pipeline** — `src/` 디렉토리의 개선된 알고리즘 (Step Detection, Step Length 추정)
- **Old Pipeline** — `src_old/` 디렉토리의 기존 Android 원본 알고리즘 (Unity-build 래퍼를 통해 `Old::` 네임스페이스로 격리)

### 입력 데이터

텍스트 파일 형식으로, 각 줄에 3개의 값을 포함합니다:

```
시간(초)   글로벌Z축_가속도   자이로_방향각(도)
```

### 시각화 (OpenCV)

| 윈도우 | 설명 |
|--------|------|
| **PDR Dashboard** | 좌측: Local Path (New=빨간색, Old=파란색 경로) / 우측: 가속도 파형 + Step 정보 패널 + Step Length 히스토리 차트 |
| **World Map View** | 지도 이미지 위에 PDR 경로를 오버레이. 시작점 클릭, 경로 회전, 지도 회전/반전 조작 가능 |
| **Controls** | 키보드 단축키 및 현재 상태 요약 |
| **Analysis** (T키 토글) | 5개 분석 차트 — Peak/Valley 검출, Amplitude vs StepLength, Frequency vs StepLength, Raw vs Smoothed SL, Frequency & Amplitude 시계열 |

### 키보드 조작

| 키 | 기능 |
|----|------|
| `Space` | 자동 재생 / 정지 |
| `A` / `D` | 이전 / 다음 프레임 |
| `W` / `S` | 재생 속도(stride) 증가 / 감소 |
| `I` / `J` / `K` / `L` | 지도 위 시작점 이동 (상/좌/하/우) |
| `U` / `O` | PDR 경로 회전 (-1도 / +1도) |
| `R` | 지도 90도 회전 |
| `H` / `V` | 지도 좌우 / 상하 반전 |
| `1` / `2` | New / Old 경로 토글 |
| `F` | 새 PDR 데이터 파일 열기 |
| `M` | 새 맵 파일 열기 |
| `T` | Analysis 창 토글 |
| `Q` / `Esc` | 종료 |

## 프로젝트 구조

```
PDR_simulation/
├── main.cpp                     # 시뮬레이션 엔진 + 메인 루프
├── viz.h / viz.cpp              # 시각화 (Dashboard, WorldMap, Analysis, Controls)
├── include/                     # New 알고리즘 헤더 (Android 원본)
├── include_old/                 # Old 알고리즘 헤더 (include/에서 복사)
├── src/                         # New 알고리즘 소스 (Android 원본, onHand.cpp만 개선)
├── src_old/                     # Old 알고리즘 소스 (Android 원본 그대로)
├── src_wrapper/                 # Old 알고리즘 Unity-build 래퍼
│   ├── OldAlgorithmWrapper.h    #   Opaque pointer API
│   └── OldAlgorithmWrapper.cpp  #   Unity-build (src_old/*.cpp를 Old:: 네임스페이스로 포함)
├── mock/                        # Android 의존성 모의 구현
│   ├── jni.h                    #   JNIEnv_ 스텁 (컴파일용)
│   └── android/log.h            #   LOGD → fprintf(stderr)
├── data/                        # 센서 데이터 + 지도 파일
└── CMakeLists.txt               # 빌드 설정
```

## 빌드 및 실행

### 의존성

- CMake 3.14+
- C++17
- OpenCV

### 빌드

```bash
mkdir -p build && cd build
cmake ..
make
```

### 실행

```bash
# macOS 파일 선택 다이얼로그로 데이터/맵 선택
./pdr_sim

# 커맨드라인 인자로 직접 지정
./pdr_sim <data_file> [map_file]
```

## 설계 원칙

- **Android 양방향 호환** — `src/`와 `include/`는 Android 프로젝트와 동일한 파일을 유지하여, 시뮬레이터에서 검증한 알고리즘을 Android에 바로 적용 가능
- **네임스페이스 격리** — Old 알고리즘은 Unity-build + `namespace Old{}`로 감싸서 New와의 심볼 충돌 방지
- **Opaque Pointer 패턴** — `OldPDRContext*`를 통해 Old 내부 구현이 외부로 노출되지 않음
- **Mock 시스템** — JNI/Android 로그 등 Android 전용 의존성을 모의 구현하여 데스크톱에서 컴파일 가능
