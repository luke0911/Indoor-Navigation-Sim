cmake_minimum_required(VERSION 3.14)
project(pdr_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# Include paths
include_directories(
        ${CMAKE_SOURCE_DIR}               # [추가됨] 프로젝트 최상위 루트 (src_old 접근용)
        ${CMAKE_SOURCE_DIR}/mock
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src           # New 헤더 경로
        ${CMAKE_SOURCE_DIR}/src_wrapper   # Wrapper 헤더 경로
)

# Source files
set(SOURCES
        main.cpp
        viz.cpp

        # [New Version Sources]
        src/Filter/MovingAverage.cpp
        src/Filter/AdaptiveFilter.cpp
        src/Sensor/SensorManager.cpp
        src/Sensor/CoordinateTransform.cpp
        src/StepDetection/onHand.cpp
        src/StepDetection/handSwing.cpp
        src/StepDetection/inPocket.cpp
        src/StepLength/StepLength.cpp
        src/StepLength/SLRequire.cpp
        src/SimDataLogger.cpp

        # [Old Version Wrapper]
        src_wrapper/OldAlgorithmWrapper.cpp
)

add_executable(pdr_sim ${SOURCES})
target_compile_definitions(pdr_sim PRIVATE PDR_SIMULATOR)
target_link_libraries(pdr_sim ${OpenCV_LIBS})

# Suppress macro redefinition warnings in the unity-build wrapper
set_source_files_properties(src_wrapper/OldAlgorithmWrapper.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-macro-redefined")

# Fix ODR violation: StepLength.cpp defines a local class MovingAverage
# that collides with Filter/MovingAverage.h's class at link time.
# Rename it to avoid the linker picking the wrong constructor.
set_source_files_properties(src/StepLength/StepLength.cpp
        PROPERTIES COMPILE_FLAGS "-DMovingAverage=SL_MovingAverage")

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)