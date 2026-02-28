cmake_minimum_required(VERSION 3.28)
project(ZeroMapSimulation)

# 시뮬레이션 모드를 기본값으로 설정
if(NOT DEFINED SIMULATION_MODE)
    set(SIMULATION_MODE ON)
endif()


set(CMAKE_CXX_STANDARD 17)

# OpenCV 설정 (먼저 찾기)
find_package(OpenCV REQUIRED)

# OpenCV 정보 출력 (디버깅용)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# OpenMP 설정
if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    endif()
endif()

add_definitions(-DSIMULATION_MODE)
add_compile_definitions(SIMULATION_MODE)

find_package(OpenMP)

# Include 디렉토리 설정
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# 각각의 실행 파일 추가
add_executable(MainApp main.cpp
        "main.cpp"
        main.cpp)

# 모든 타겟에 OpenCV 라이브러리 링크 (OpenMP 있든 없든)
target_link_libraries(MainApp PRIVATE ${OpenCV_LIBS})

# OpenMP가 있으면 추가로 링크
if(OpenMP_CXX_FOUND)
    target_link_libraries(MainApp PRIVATE OpenMP::OpenMP_CXX)
endif()