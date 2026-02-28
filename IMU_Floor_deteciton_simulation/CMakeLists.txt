cmake_minimum_required(VERSION 3.14)
project(floor_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# Include paths
include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/mock
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Source files
set(SOURCES
        main.cpp
        viz.cpp
        imu_floor_detection.cpp

        # [PDR Step Detection Sources - shared via symlink]
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
)

add_executable(floor_sim ${SOURCES})
target_compile_definitions(floor_sim PRIVATE PDR_SIMULATOR)
target_link_libraries(floor_sim ${OpenCV_LIBS})

# Fix ODR violation: StepLength.cpp defines a local class MovingAverage
set_source_files_properties(src/StepLength/StepLength.cpp
        PROPERTIES COMPILE_FLAGS "-DMovingAverage=SL_MovingAverage")

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)
