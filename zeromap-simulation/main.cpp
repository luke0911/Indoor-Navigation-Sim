// 시뮬레이션 모드 설정
#ifdef SIMULATION_MODE
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <omp.h>         // OpenMP 헤더
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <regex>
#include <filesystem>
#include <print.h>
#include <chrono>  // 추가

struct AAssetManager {};
AAssetManager* g_assetManager = nullptr;
#define LOGI(...) std::cout

using json = nlohmann::json;
using namespace std;
namespace fs = std::filesystem;
#else
// 기존 안드로이드 헤더들
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <omp.h>         // OpenMP 헤더
#include <nlohmann/json.hpp>
#include <jni.h>
#include <regex>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <queue>
#include <chrono>  // 이미 있으면 OK
#include <opencv2/opencv.hpp>



#define LOG_TAG "NativeLib"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// 전역 변수로 AAssetManager 포인터를 저장합니다.
std::string nativeMessage = "";
AAssetManager* g_assetManager = nullptr;
#endif

// 파일 상단에 헬퍼 함수 추가
inline double normalizeAngle(double angle) {
    angle = fmod(angle, 360.0);
    if (angle < 0) angle += 360.0;
    return angle;
}

// ========== 새로운 방향 추정 시스템 ==========

// 나침반 분석기
// 나침반 분석기
struct CompassAnalyzer {
    std::vector<std::pair<int, float>> compass_history;
    std::vector<float> gyro_history;
    std::vector<float> direction_offsets;
    std::vector<float> step_lengths;

    void addMeasurement(int step, float compass, float gyro, float stepLen) {
        compass_history.push_back({step, compass});
        gyro_history.push_back(gyro);
        step_lengths.push_back(stepLen);

        float offset = compass - gyro;
        while (offset > 180) offset -= 360;
        while (offset < -180) offset += 360;

        direction_offsets.push_back(offset);
    }

    double getRobustTrend() const {
        if (direction_offsets.size() < 20) return -1;

        std::vector<int> histogram(360, 0);

        for (float offset : direction_offsets) {
            double angle = offset + 180;
            int bin = (int)angle % 360;

            if (bin >= 0 && bin < 360) {
                histogram[bin]++;
            }
        }

        int max_density = 0;
        int best_center = 0;
        for (int i = 0; i < 360; i++) {
            int density = 0;
            for (int j = -5; j <= 5; j++) {
                int idx = (i + j + 360) % 360;
                density += histogram[idx];
            }
            if (density > max_density) {
                max_density = density;
                best_center = i;
            }
        }

        double dense_region_center = best_center - 180;
        double inlier_ratio = (double)max_density / direction_offsets.size();

        if (inlier_ratio > 0.3) {
            return dense_region_center;
        }

        std::vector<float> sorted = direction_offsets;
        std::sort(sorted.begin(), sorted.end());
        double median = sorted[sorted.size() / 2];

        return median;
    }

    double evaluateGyroCompassConsistency() const {
        if (direction_offsets.size() < 10) return 0.0;

        int consistent_count = 0;
        int total_count = 0;

        size_t max_i = std::min(direction_offsets.size(), (size_t)50);

        for (size_t i = 1; i < max_i; i++) {
            float diff = std::abs(direction_offsets[i] - direction_offsets[i-1]);

            if (diff < 15) {
                consistent_count++;
            }
            total_count++;
        }

        return total_count > 0 ? (double)consistent_count / total_count : 0.0;
    }

    double getTemporalStability() const {
        if (direction_offsets.size() < 50) return 0.5;

        int recent_size = std::min(20, (int)direction_offsets.size());

        auto calc_variance = [](const std::vector<float>& data, int start, int count) {
            double mean = 0;
            for (int i = start; i < start + count && i < (int)data.size(); i++) {
                mean += data[i];
            }
            mean /= count;

            double variance = 0;
            for (int i = start; i < start + count && i < (int)data.size(); i++) {
                double diff = data[i] - mean;
                variance += diff * diff;
            }
            return variance / count;
        };

        double recent_var = calc_variance(direction_offsets,
                                          direction_offsets.size() - recent_size,
                                          recent_size);

        double result = 1.0 / (1.0 + recent_var);

        return result;
    }
};

// 자이로 드리프트 추적기
struct GyroDriftTracker {
    double baseline_offset;
    double accumulated_drift;
    double drift_rate;
    std::vector<double> drift_history;
    int last_correction_step;

    GyroDriftTracker() : baseline_offset(0), accumulated_drift(0),
                         drift_rate(0), last_correction_step(0) {}

    void setBaseline(double offset) {
        baseline_offset = offset;
    }

    void updateDriftEstimate(int current_step, float measured_gyro,
                             double compass_trend, bool loc_good, double match_quality) {
        if (compass_trend > 0) {
            double expected = measured_gyro + baseline_offset + accumulated_drift;
            double compass_drift = compass_trend - expected;

            while (compass_drift > 180) compass_drift -= 360;
            while (compass_drift < -180) compass_drift += 360;

            if (std::abs(compass_drift) < 10) {
                drift_history.push_back(compass_drift);
                if (drift_history.size() > 50) {
                    drift_history.erase(drift_history.begin());
                }
            }
        }

        if (current_step - last_correction_step >= 50 && drift_history.size() >= 20) {
            std::vector<double> sorted = drift_history;
            std::sort(sorted.begin(), sorted.end());
            double median_drift = sorted[sorted.size() / 2];

            if (std::abs(median_drift) > 2.0) {
                accumulated_drift += median_drift;
                drift_history.clear();
            }
            last_correction_step = current_step;
        }
    }

    double getCurrentOffset() const {
        return baseline_offset + accumulated_drift;
    }
};

// 방향 가설
struct OrientationHypothesis {
    double angle;
    double confidence;
    int success_count;
    int failure_count;
    double compass_alignment;
    double last_match_quality;

    OrientationHypothesis(double a = 0) : angle(a), confidence(0.1),
                                          success_count(0), failure_count(0),
                                          compass_alignment(0), last_match_quality(0) {}
};

struct HealthMetrics {
    double compass_reliability;
    double localization_success_rate;
    double drift_magnitude;
    double overall_health;

    HealthMetrics() : compass_reliability(0), localization_success_rate(1.0),
                      drift_magnitude(0), overall_health(0.5) {}
};

#ifndef SIMULATION_MODE
extern "C"
JNIEXPORT jstring JNICALL
Java_com_fifth_maplocationlib_NativeLib_getStringFromNative(JNIEnv* env, jobject /* this */) {
    return env->NewStringUTF(nativeMessage.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_fifth_maplocationlib_NativeLib_setAssetManager(JNIEnv *env, jobject /* this */, jobject assetManager) {
    g_assetManager = AAssetManager_fromJava(env, assetManager);
}
#endif

using json = nlohmann::json;
using namespace std;

std::map<int, std::pair<int, int>> originMap;

json loadJsonData(const char* filename) {

#ifdef SIMULATION_MODE
    // 시뮬레이션: 일반 파일 시스템에서 읽기
    std::ifstream file(filename);
    if (!file.is_open()) {
        cerr << "File open failed: " << filename << endl;
        return json();
    }
    try {
        json j;
        file >> j;
        return j;
    } catch (json::parse_error& e) {
        cerr << "JSON parse error: " << e.what() << endl;
        return json();
    }
#else
    AAsset* asset = AAssetManager_open(g_assetManager, filename, AASSET_MODE_BUFFER);
    if (!asset) {
        cerr << "Asset open failed: " << filename << endl;
        return json(); // 빈 json 객체 반환
    }
    off_t length = AAsset_getLength(asset);
    string content;
    content.resize(length);
    AAsset_read(asset, &content[0], length);
    AAsset_close(asset);

    try {
        return json::parse(content);
    } catch (json::parse_error& e) {
        cerr << "JSON parse error in file " << filename << ": " << e.what() << endl;
        return json(); // 빈 json 객체 반환
    }
#endif
}


// Loaded at initializeEngine() from assets (testbed/enb/...)
json stairsInfo;      // enb/floorchange_stairs_areas.json
json elevatorInfo;    // enb/floorchange_elevator_areas.json

using Point = pair<int, int>;
int gStepCount = 0;
string global_testbed_name = "neb";


// 전역 변수: 수렴 여부 및 마지막 매칭 결과 (PDR 업데이트에 사용)
bool didConverge = false;
pair<double, double> cur_pos_global; // (x, y)
double cur_distance; // (x, y)

// =============================================================
/// 지도 데이터 관련 구조체 및 함수
// =============================================================

struct MapPoint {
    double x, y;
    MapPoint(double x = 0, double y = 0) : x(x), y(y) {}
};

struct Node {
    string id;
    MapPoint coords;
};

struct Edge {
    string start;
    string end;
    double distance;
};

struct MapData {
    vector<Node> nodes;
    vector<Edge> edges;
};

// MapMatchingConfig 구조체
struct MapMatchingConfig {
    vector<pair<MapPoint, MapPoint>> matchLines;  // 맵매칭할 선분들
    vector<vector<MapPoint>> noMatchZones;        // 맵매칭 금지 영역들
};

MapPoint operator-(const MapPoint& a, const MapPoint& b) {
    return MapPoint(a.x - b.x, a.y - b.y);
}

double dot(const MapPoint& a, const MapPoint& b) {
    return a.x * b.x + a.y * b.y;
}

double magnitude(const MapPoint& p) {
    return sqrt(p.x * p.x + p.y * p.y);
}

// JSON에서 맵매칭 설정 로드
MapMatchingConfig loadMapMatchingConfig(const char* filename) {
    MapMatchingConfig config;

#ifdef SIMULATION_MODE
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Map matching config file not found");
    }
    json j;
    file >> j;
#else
    AAsset* asset = AAssetManager_open(g_assetManager, filename, AASSET_MODE_BUFFER);
    if (!asset) {
        // throw 대신 빈 config 반환하고 호출부에서 처리
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
                            "Map matching config not found: %s", filename);
        throw std::runtime_error("Map matching config file not found");
    }

    off_t length = AAsset_getLength(asset);
    string content;
    content.resize(length);
    AAsset_read(asset, &content[0], length);
    AAsset_close(asset);

    json j;
    try {
        j = json::parse(content);
    } catch (json::parse_error& e) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "JSON parse error in %s: %s", filename, e.what());
        throw std::runtime_error("Map matching config parse error");
    }
#endif

    // 공통 파싱 로직
    if (j.contains("match_lines")) {
        for (const auto& line : j["match_lines"]) {
            MapPoint start(line[0][0], line[0][1]);
            MapPoint end(line[1][0], line[1][1]);
            config.matchLines.push_back({start, end});
        }
    }

    if (j.contains("no_match_zones")) {
        for (const auto& zone : j["no_match_zones"]) {
            vector<MapPoint> polygon;
            for (const auto& point : zone) {
                polygon.push_back(MapPoint(point[0], point[1]));
            }
            config.noMatchZones.push_back(polygon);
        }
    }

    return config;
}

// 점이 폴리곤 내부에 있는지 확인 (Ray Casting Algorithm)
bool isPointInPolygon(const MapPoint& point, const vector<MapPoint>& polygon) {
    int n = polygon.size();
    bool inside = false;

    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = polygon[i].x, yi = polygon[i].y;
        double xj = polygon[j].x, yj = polygon[j].y;

        bool intersect = ((yi > point.y) != (yj > point.y)) &&
                         (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }

    return inside;
}

// 점이 맵매칭 금지 영역에 있는지 확인
bool isInNoMatchZone(const MapPoint& point, const MapMatchingConfig& config) {
    for (const auto& zone : config.noMatchZones) {
        if (isPointInPolygon(point, zone)) {
            return true;
        }
    }
    return false;
}


MapData parseJsonToMapData(const string& jsonString) {
    MapData mapData;
    try {
        json j = json::parse(jsonString);
        for (const auto& node : j["nodes"]) {
            Node n;
            n.id = node["id"];
            n.coords.x = node["coords"][0];
            n.coords.y = node["coords"][1];
            mapData.nodes.push_back(n);
        }
        for (const auto& edge : j["edges"]) {
            Edge e;
            e.start = edge["start"];
            e.end = edge["end"];
            e.distance = edge["distance"];
            mapData.edges.push_back(e);
        }
    } catch (json::parse_error& e) {
        cerr << "JSON parse error: " << e.what() << endl;
    } catch (exception& e) {
        cerr << "Error parsing JSON: " << e.what() << endl;
    }
    return mapData;
}

MapPoint find_closest_point_on_edge(const MapPoint& p, const MapPoint& a, const MapPoint& b) {
    MapPoint ap = p - a;
    MapPoint ab = b - a;
    double ab2 = dot(ab, ab);
    double ap_ab = dot(ap, ab);
    double t = ap_ab / ab2;
    if(t < 0.0) return a;
    else if(t > 1.0) return b;
    else return MapPoint(a.x + ab.x * t, a.y + ab.y * t);
}


MapPoint getMapMatchingResult(const MapPoint& current_position, const MapData& data) {
    double min_distance = numeric_limits<double>::infinity();
    MapPoint closest_point;
    for (const auto& edge : data.edges) {
        auto start_node = find_if(data.nodes.begin(), data.nodes.end(),
                                  [&](const Node& node) { return node.id == edge.start; });
        auto end_node = find_if(data.nodes.begin(), data.nodes.end(),
                                [&](const Node& node) { return node.id == edge.end; });
        if (start_node != data.nodes.end() && end_node != data.nodes.end()) {
            MapPoint point = find_closest_point_on_edge(current_position, start_node->coords, end_node->coords);
            double distance = magnitude(point - current_position);
            if (distance < min_distance) {
                min_distance = distance;
                closest_point = point;
            }
        }
    }
    return closest_point;
}

// 업그레이드된 맵매칭 함수
MapPoint getMapMatchingResultV2(const MapPoint& current_position,
                                const MapMatchingConfig& config) {
    // 맵매칭 금지 영역에 있으면 원래 위치 그대로 반환
    if (isInNoMatchZone(current_position, config)) {
        return current_position;
    }

    // 가장 가까운 맵매칭 라인 찾기
    double min_distance = numeric_limits<double>::infinity();
    MapPoint closest_point = current_position;

    for (const auto& line : config.matchLines) {
        MapPoint point = find_closest_point_on_edge(current_position, line.first, line.second);
        double distance = magnitude(point - current_position);

        if (distance < min_distance) {
            min_distance = distance;
            closest_point = point;
        }
    }

    return closest_point;
}

// 기존 Edge 기반 맵매칭과 혼합 사용
MapPoint getMapMatchingResultHybrid(const MapPoint& current_position,
                                    const MapData& mapData,
                                    const MapMatchingConfig& config) {
    // 1. 맵매칭 금지 영역 체크
    if (isInNoMatchZone(current_position, config)) {
        return current_position;
    }

    // 2. 커스텀 맵매칭 라인이 있으면 우선 사용
    if (!config.matchLines.empty()) {
        double min_distance = numeric_limits<double>::infinity();
        MapPoint closest_point = current_position;

        for (const auto& line : config.matchLines) {
            MapPoint point = find_closest_point_on_edge(current_position, line.first, line.second);
            double distance = magnitude(point - current_position);

            if (distance < min_distance) {
                min_distance = distance;
                closest_point = point;
            }
        }

        // 커스텀 라인과의 거리가 충분히 가까우면 사용
        if (min_distance < 50.0) {  // 임계값 (5미터)
            return closest_point;
        }
    }

    // 3. 기존 Edge 기반 맵매칭 사용 (fallback)
    return getMapMatchingResult(current_position, mapData);
}

// =============================================================
/// BitMatrix 클래스: 0/1 데이터를 64비트 단위로 저장
// =============================================================

class BitMatrix {
public:
    int height, width, blocksPerRow;
    vector<Point> one_pixels;  // 추가
    vector<vector<uint64_t>> rows;

    // 기본 생성자 추가
    BitMatrix() : height(0), width(0), blocksPerRow(0) {}

    BitMatrix(int h, int w) : height(h), width(w) {
        blocksPerRow = (w + 63) / 64;
        rows.assign(h, vector<uint64_t>(blocksPerRow, 0ULL));
    }
    void setBit(int r, int c) {
        if (r < 0 || r >= height || c < 0 || c >= width) return;
        int blockIdx = c / 64;
        int bitIdx = c % 64;
        rows[r][blockIdx] |= (1ULL << bitIdx);
        one_pixels.push_back({r, c});  // 추가
    }

    bool getBit(int r, int c) const {
        if (r < 0 || r >= height || c < 0 || c >= width) return false;
        int blockIdx = c / 64;
        int bitIdx = c % 64;
        return (rows[r][blockIdx] >> bitIdx) & 1ULL;
    }
};

// =============================================================
/// 비트 연산 함수들
// =============================================================

// 지정된 행(row)의 colStart부터 bits 길이 만큼의 비트를 추출
uint64_t extract_bits_from_row(const vector<uint64_t>& row, int colStart, int bits) {
    int blockIndex = colStart / 64;
    int offset = colStart % 64;

    if (blockIndex >= row.size()) {
        return 0;
    }

    if (offset + bits <= 64) {
        return (row[blockIndex] >> offset) & ((1ULL << bits) - 1);
    } else {
        int bits_in_first = 64 - offset;
        uint64_t part1 = row[blockIndex] >> offset;
        int remaining = bits - bits_in_first;

        if (blockIndex + 1 >= row.size()) {
            return part1;
        }

        uint64_t part2 = row[blockIndex + 1] & ((1ULL << remaining) - 1);
        return part1 | (part2 << bits_in_first);
    }
}

int calc_similarity_fast_parallel(const BitMatrix &B, int offset_row, int offset_col, const BitMatrix &T) {
    int similarity = 0;

#pragma omp parallel for reduction(+:similarity)
    for (size_t idx = 0; idx < T.one_pixels.size(); idx++) {
        const auto& p = T.one_pixels[idx];
        int b_row = offset_row + p.first;
        int b_col = offset_col + p.second;

        if (b_row >= 0 && b_row < B.height && b_col >= 0 && b_col < B.width) {
            if (B.getBit(b_row, b_col)) {
                similarity++;
            }
        }
    }

    return similarity;
}

// 두 BitMatrix(B: 큰 지도, T: 테스트 패턴)의 겹치는 영역에 대해
// bitwise AND 후 popcount를 이용하여 유사도를 계산
int calc_similarity_at_position(const BitMatrix &B, int i, int j, const BitMatrix &T) {
    // 범위 체크 추가
    if (i < 0 || i + T.height > B.height) {
        return 0;
    }

    if (j < 0 || j + T.width > B.width) {
        return 0;
    }

    int similarity = 0;
#pragma omp parallel for reduction(+: similarity) schedule(guided)
    for (int r = 0; r < T.height; r++) {
        for (int b = 0; b < T.blocksPerRow; b++) {
            int colStart = j + b * 64;
            int bits_in_block = 64;
            if (b == T.blocksPerRow - 1) {
                int rem = T.width % 64;
                if (rem != 0) bits_in_block = rem;
            }

            // 범위 체크
            if (i + r >= B.height || colStart >= B.width) {
                continue;
            }

            uint64_t B_block = extract_bits_from_row(B.rows[i + r], colStart, bits_in_block);
            uint64_t T_block = T.rows[r][b];
            similarity += __builtin_popcountll(B_block & T_block);
        }
    }
    return similarity;
}

// =============================================================
/// ROI와 OpenMP를 활용한 병렬 슬라이딩 윈도우 매칭 함수
// =============================================================

std::pair<int, std::vector<Point>> sliding_window_similarity_optimized(
        const BitMatrix &B, const BitMatrix &T,
        int roi_center_row = -1, int roi_center_col = -1, int roi_radius = -1,
        const Point &T_anchor = {0, 0}
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // PDR이 맵보다 커도 매칭 허용 (음수 오프셋 허용)
    int start_row = std::min(0, B.height - T.height);
    int end_row = std::max(0, B.height - T.height);
    int start_col = std::min(0, B.width - T.width);
    int end_col = std::max(0, B.width - T.width);

    if (roi_center_row >= 0 && roi_center_col >= 0 && roi_radius >= 0) {
        start_row = std::max(0, roi_center_row - roi_radius - T_anchor.first);
        end_row = std::min(B.height - T.height, roi_center_row + roi_radius - T_anchor.first);
        start_col = std::max(0, roi_center_col - roi_radius - T_anchor.second);
        end_col = std::min(B.width - T.width, roi_center_col + roi_radius - T_anchor.second);
    }

    if (start_row > end_row || start_col > end_col) {
        return {-1, {}};
    }

    int crop_start_row = start_row;
    int crop_start_col = start_col;
    int crop_height = std::min(end_row - start_row + T.height, B.height - start_row);
    int crop_width = std::min(end_col - start_col + T.width, B.width - start_col);

    cv::Mat B_crop(crop_height, crop_width, CV_32F, cv::Scalar(0));
    for (int r = 0; r < crop_height; r++) {
        for (int c = 0; c < crop_width; c++) {
            if (B.getBit(crop_start_row + r, crop_start_col + c)) {
                B_crop.at<float>(r, c) = 1.0f;
            }
        }
    }

    cv::Mat T_mat(T.height, T.width, CV_32F, cv::Scalar(0));
    for (int r = 0; r < T.height; r++) {
        for (int c = 0; c < T.width; c++) {
            if (T.getBit(r, c)) T_mat.at<float>(r, c) = 1.0f;
        }
    }

    cv::Mat result;
    cv::matchTemplate(B_crop, T_mat, result, cv::TM_CCORR);

    float global_max = -1;
    std::vector<Point> global_positions;

    for (int r = 0; r < result.rows; r++) {
        for (int c = 0; c < result.cols; c++) {
            int orig_row = crop_start_row + r;
            int orig_col = crop_start_col + c;

            int check_row = orig_row + T_anchor.first;
            int check_col = orig_col + T_anchor.second;

            if (check_row < 0 || check_row >= B.height ||
                check_col < 0 || check_col >= B.width) {
                continue;
            }

            if (!B.getBit(check_row, check_col)) continue;

            float val = result.at<float>(r, c);

            if (val > global_max) {
                global_max = val;
                global_positions.clear();
                global_positions.push_back({orig_row, orig_col});
            } else if (std::abs(val - global_max) < 0.5f) {
                global_positions.push_back({orig_row, orig_col});
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

#ifdef SIMULATION_MODE
//    print(color::cyan, "  [Sliding Window Match - FFT]", color::reset);
//    print("    - B crop size:", crop_height, "x", crop_width);
//    print("    - T size:", T.height, "x", T.width);
//    print("    - ROI:", roi_radius >= 0 ? "Enabled" : "Disabled");
//    print("    - Max similarity:", static_cast<int>(global_max));
//    print("    - Candidates:", global_positions.size());
    print("    - Time:", color::yellow, duration.count(), "ms", color::reset);
#endif

    return {static_cast<int>(global_max), global_positions};
}


// =============================================================
/// BitMatrix 생성: 좌표 목록으로부터 BitMatrix 생성
// =============================================================

BitMatrix create_bitmatrix_from_coordinates(const vector<Point>& coordinates) {
    if (coordinates.empty()) {
        return BitMatrix(0, 0);
    }

    int max_x = 0, max_y = 0;
    for (const auto& coord : coordinates) {
        max_x = max(max_x, coord.first);
        max_y = max(max_y, coord.second);
    }

    // 너무 큰 매트릭스 방지
    if (max_x > 100000 || max_y > 100000) {
        cerr << "Warning: BitMatrix too large (" << max_x << "x" << max_y << "), limiting size" << endl;
        return BitMatrix(0, 0);
    }

    BitMatrix bm(max_x + 1, max_y + 1);
    for (const auto& coord : coordinates) {
        if (coord.first >= 0 && coord.first <= max_x &&
            coord.second >= 0 && coord.second <= max_y) {
            bm.setBit(coord.first, coord.second);
        }
    }

    return bm;
}

pair<double, Point> analyzeCoordinates(const vector<Point>& coordinates) {

    if (coordinates.empty()) {
        return {0.0, {0, 0}};
    }

    long long sum_x = 0, sum_y = 0;
    for (const auto& coord : coordinates) {
        sum_x += coord.first;
        sum_y += coord.second;
    }
    double mean_x = static_cast<double>(sum_x) / coordinates.size();
    double mean_y = static_cast<double>(sum_y) / coordinates.size();


    double total_distance = 0.0;
    for (const auto& coord : coordinates) {
        double dx = coord.first - mean_x;
        double dy = coord.second - mean_y;
        total_distance += sqrt(dx * dx + dy * dy);
    }
    double mean_distance = (total_distance / coordinates.size()) * 0.1;


    return {mean_distance, {static_cast<int>(mean_x), static_cast<int>(mean_y)}};
}

// =============================================================
/// GyroBuffer 클래스
// =============================================================

class GyroBuffer {
private:
    deque<double> buffer;
    const size_t maxSize = 6;
public:
    void add(double gyroValue) {
        if (buffer.size() == maxSize) {
            buffer.pop_front();
        }
        buffer.push_back(gyroValue);
    }
    bool didBigRotation() {
        if (buffer.size() < 6) return false;
        double diff = fabs(buffer.front() - buffer.back());
        diff = fmod(diff, 360.0);
        if (diff > 180) diff = 360 - diff;
        return diff > 70;
    }
};

// =============================================================
/// PositionTracker 클래스 (BitMatrix, ROI, OpenMP 최적화 적용)
// =============================================================
struct ProcessResult {
    double x;
    double y;
    double distance;
    bool valid;
    int gyroCaliValue;

    // 캘리브레이션 디버깅 추가
    bool calibrationDone;
    vector<double> candidateDiffs;
    int bestCandidateIdx;
    vector<int> calibHistory;

    // 맵매칭 관련 추가
    double original_x;           // 맵매칭 전 원본 좌표
    double original_y;
    bool has_mapmatching;        // 맵매칭 적용 여부

    vector<Point> candidates;  // 추가

    vector<Point> best_history;  // 추가
    double base_orientation;      // 추가

};

// 텍스트 파일로부터 2D int 벡터 파싱
vector<vector<int>> parseMatrixFromText(const char* data, size_t length) {
    vector<vector<int>> matrix;
    string text(data, length);
    istringstream stream(text);
    string line;
    while (getline(stream, line)) {
        vector<int> row;
        istringstream lineStream(line);
        int value;
        while (lineStream >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    return matrix;
}

// 2D int 벡터를 BitMatrix로 변환
BitMatrix convertToBitMatrix(const vector<vector<int>>& mat2D) {
    if (mat2D.empty()) return BitMatrix(0,0);
    int rows = mat2D.size();
    int cols = 0;
    for (const auto& row : mat2D) {
        cols = max(cols, (int)row.size());
    }
    BitMatrix bm(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < mat2D[i].size(); j++) {
            if (mat2D[i][j] == 1)
                bm.setBit(i, j);
        }
    }
    return bm;
}

BitMatrix readBitMatrixFromTextFile(const char* filename) {
#ifdef SIMULATION_MODE
    ifstream file(filename);
    vector<vector<int>> tempMatrix;
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<int> row;
        int value;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty())
            tempMatrix.push_back(row);
    }
    int height = tempMatrix.size();
    int width = 0;
    for (const auto& row : tempMatrix) {
        width = max(width, (int)row.size());
    }
    BitMatrix bm(height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < tempMatrix[i].size(); j++) {
            if (tempMatrix[i][j] == 1)
                bm.setBit(i, j);
        }
    }
    return bm;
#else
    // assets에서 파일 열기
    AAsset* asset = AAssetManager_open(g_assetManager, filename, AASSET_MODE_UNKNOWN);
    if (!asset) {
        cerr << "Asset open failed: " << filename << endl;
        return BitMatrix(0, 0);
    }
    off_t length = AAsset_getLength(asset);
    string content;
    content.resize(length);
    AAsset_read(asset, &content[0], length);
    AAsset_close(asset);

    // 기존 코드와 동일하게 텍스트 내용을 파싱합니다.
    vector<vector<int>> tempMatrix;
    istringstream fileStream(content);
    string line;
    while (getline(fileStream, line)) {
        istringstream iss(line);
        vector<int> row;
        int value;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty())
            tempMatrix.push_back(row);
    }
    int height = tempMatrix.size();
    int width = 0;
    for (const auto& row : tempMatrix) {
        width = max(width, (int)row.size());
    }
    BitMatrix bm(height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < tempMatrix[i].size(); j++) {
            if (tempMatrix[i][j] == 1)
                bm.setBit(i, j);
        }
    }
    return bm;
#endif
}

// assets에서 JSON 파일을 읽어 MapData 생성 (기존 ifstream 코드와 유사)
MapData loadMapData(const char* filename) {

#ifdef SIMULATION_MODE
    ifstream file(filename);
    json j;
    file >> j;
    MapData data;
    for (const auto& node : j["nodes"]) {
        data.nodes.push_back({
                                     node["id"],
                                     MapPoint(node["coords"][0], node["coords"][1])
                             });
    }
    for (const auto& edge : j["edges"]) {
        data.edges.push_back({
                                     edge["start"],
                                     edge["end"],
                                     edge["distance"]
                             });
    }
    return data;
#else
    AAsset* asset = AAssetManager_open(g_assetManager, filename, AASSET_MODE_UNKNOWN);
    if (!asset) {
        cerr << "Asset open failed: " << filename << endl;
        return MapData();
    }
    off_t length = AAsset_getLength(asset);
    string content;
    content.resize(length);
    AAsset_read(asset, &content[0], length);
    AAsset_close(asset);

    json j;
    try {
        j = json::parse(content);
    } catch (json::parse_error& e) {
        cerr << "JSON parse error: " << e.what() << endl;
        return MapData();
    }
    MapData data;
    for (const auto& node : j["nodes"]) {
        data.nodes.push_back({
                                     node["id"],
                                     MapPoint(node["coords"][0], node["coords"][1])
                             });
    }
    for (const auto& edge : j["edges"]) {
        data.edges.push_back({
                                     edge["start"],
                                     edge["end"],
                                     edge["distance"]
                             });
    }
    return data;
#endif
}

// =============================================================
/// PositionTracker 클래스 (BitMatrix, ROI, OpenMP 최적화 적용)
// =============================================================

class PositionTracker {
private:

    // 기존 멤버 변수들 근처에 추가
    double sticky_best_orientation = 0.0;      // 현재 고정된 best 각도
    int sticky_best_similarity = 0;            // 해당 각도의 similarity
    bool has_sticky_orientation = false;       // sticky 각도가 설정되었는지


    // 가장 좋은 각도 추적용 변수
    int consecutive_best_count = 0;  // 연속으로 best가 된 횟수
    int last_best_idx = -1;         // 이전 best 인덱스
    bool fine_tuning_mode = false;  // 미세 조정 모드 활성화 여부
    bool fine_tuning_complete_mode = false;  // 미세 조정 모드 활성화 여부
    double fine_tuning_base_angle = 0.0; // 미세 조정 기준 각도

    vector<vector<Point>> multiple_history_paths;
    vector<Point> multiple_cur_pos;
    vector<bool> active_paths; // 활성화된 경로 추적

    // PositionTracker 클래스 내부 (private 멤버 변수 부분)
    double prevDiffCompassGyro = -1;     // 이전 step의 나침반-자이로 차이 (없으면 -1)
    int consecutiveStableCount = 0;      // diffCompassGyroStepByStep 값이 20 미만인 연속 횟수

    int cur_max_similarity = 0;
    int very_max_similarity = 0;
    bool reSearchFlag = false;
    int reSearchRoi = 100;
    BitMatrix binary_map;
    Point cur_pos;
    vector<Point> history_of_pos;
    GyroBuffer gyroBuffer;
    MapData map_data;
    MapMatchingConfig map_matching_config;  // 추가
    bool has_map_matching_config;           // 추가: 설정 파일 존재 여부

    vector<OrientationHypothesis> orientation_hypotheses;
    HealthMetrics health_metrics;
    deque<bool> recent_loc_results;

    bool orientation_stable;
    int orientation_stable_count;


    bool has_last_matched;
    int roi_radius = 90000; // ROI 반경 (셀 단위)
    // 현재 층 관리 변수 (생성자에서 start_floor로 초기화)
    int currentFloor;

    // 계단 관련 상태
    bool onStairs = false;
    float lastGyro = 0.0f;
    Point roi_center; // 새 층 도착 시 설정된 ROI center

    vector<Point> storedStairCoords; // 계단 좌표들을 저장
    vector<Point> storedElevatorCoords; // 계단 좌표들을 저장

    bool newFloorMode = false;          // 새 층 모드 활성화 여부
    Point newFloorArrivalCoord = {-1, -1}; // 도착한 층의 계단 도착 좌표
    string prevFloorElevation = "-";

    int previousFloor = -9;

    bool gyroCalibrationDone = false;
    int finalGyroCaliValue = 0;  // 최종 보정값 (0, 90, 180, 270 중 하나)
    vector<int> gyroCaliRankHistory;  // 최근 10회의 1등 후보 기록
    vector<double> prevDiffs;         // 이전 스텝의 각 후보별 차이 값 (크기 4)

    // 후보 각도 배열 (상수로 두어도 됨)
//    const vector<int> gyroCaliCandidate = {0, 45, 115, 123, 135, 225, 270, 315};
    const vector<int> gyroCaliCandidate = {0, 45, 90, 135, 180, 225, 270, 315};
//    const vector<int> gyroCaliCandidate = {0, 0, 90, 90, 180, 180, 270, 270};

    // 캘리브레이션 진행 동안의 센서값 누적 (나중에 보정 적용)
    vector<float> pendingGyros;
    vector<float> pendingStepLengths;

    // 추가: testbed 경로 저장
    std::string testbedPath;

    size_t new_floor_start_index = 0;

    string getFloorText(int floor) {
        return (floor < 0 ? "B" + to_string(abs(floor)) : to_string(floor)) + "F";
    }
public:
    PositionTracker(const BitMatrix& binMap, const MapData& mapData,
                    int start_floor, const std::string& testbedDir)
            : binary_map(binMap), map_data(mapData), currentFloor(start_floor),
              testbedPath(testbedDir), has_map_matching_config(false),
              orientation_stable(false), orientation_stable_count(0) {

        // 초기 가설 생성 (전방위)
        for (int angle = 0; angle < 360; angle += 45) {
            orientation_hypotheses.push_back(OrientationHypothesis(angle));
        }

        // 맵매칭 설정 로드
        string mmConfigFile = testbedDir + "/mapmatching_config_" +
                              getFloorText(start_floor) + ".json";
        try {
            map_matching_config = loadMapMatchingConfig(mmConfigFile.c_str());
            has_map_matching_config = true;
        } catch (...) {
            has_map_matching_config = false;
        }

        global_testbed_name = testbedPath;
    }

    CompassAnalyzer compass_analyzer;
    GyroDriftTracker drift_tracker;

    void updateMaps(const BitMatrix &newBinaryMap, const MapData &newMapData) {
        binary_map = newBinaryMap;
        map_data = newMapData;

        string mmConfigFile = testbedPath + "/mapmatching_config_" +
                              getFloorText(currentFloor) + ".json";

        try {
            map_matching_config = loadMapMatchingConfig(mmConfigFile.c_str());
            has_map_matching_config = true;
#ifdef SIMULATION_MODE
            cout << "Loaded map matching config for floor " << currentFloor << endl;
#else
            __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                                "Loaded map matching config for floor %d", currentFloor);
#endif
        } catch (...) {
            has_map_matching_config = false;
#ifdef SIMULATION_MODE
            cout << "No map matching config for floor " << currentFloor << endl;
#else
            __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                                "No map matching config for floor %d", currentFloor);
#endif
        }
    }

    void resetHistory() {
        // 기본 경로 초기화
        history_of_pos.clear();
        cur_pos = {0, 0};


        multiple_history_paths.clear();
        multiple_cur_pos.clear();
        pendingGyros.clear();
        pendingStepLengths.clear();

        // 미세 조정 모드 초기화
//        fine_tuning_mode = false;
        consecutive_best_count = 0;
        last_best_idx = -1;

        // 모든 경로 활성화
        fill(active_paths.begin(), active_paths.end(), true);

        // Sticky angle 초기화 추가
        has_sticky_orientation = false;
        sticky_best_orientation = 0.0;
        sticky_best_similarity = 0;
    }

    // 현재 층 반환
    int getCurrentFloor() const {
        return currentFloor;
    }

    // 재탐색 시작
    void reSearchStart(std::pair<double, double> center, int range) {
        reSearchFlag = true;
        reSearchRoi = range * 10;  // 미터 → 픽셀 변환
        roi_center = {static_cast<int>(center.first), static_cast<int>(center.second)};

        // 히스토리 초기화
        resetHistory();

        // 새 위치 기준으로 설정
        cur_pos_global = center;
    }

    // 자이로 캘리브레이션 리셋
    void resetGyroCalibration() {
        // 나침반 분석기 초기화
        compass_analyzer.compass_history.clear();
        compass_analyzer.gyro_history.clear();
        compass_analyzer.direction_offsets.clear();
        compass_analyzer.step_lengths.clear();

        // 드리프트 트래커 초기화
        drift_tracker.baseline_offset = 0;
        drift_tracker.accumulated_drift = 0;
        drift_tracker.drift_history.clear();
        drift_tracker.last_correction_step = 0;

        // 가설 초기화
        orientation_hypotheses.clear();
        for (int angle = 0; angle < 360; angle += 45) {
            orientation_hypotheses.push_back(OrientationHypothesis(angle));
        }

        // 히스토리 초기화
        resetHistory();

        // 전역 변수 초기화
        didConverge = false;

        // Sticky angle 초기화 추가
        has_sticky_orientation = false;
        sticky_best_orientation = 0.0;
        sticky_best_similarity = 0;
    }


    void setArrivedInfo(float gyro, int elevationMode) {

        std::string stairElevation = "";


        if (currentFloor != previousFloor) {
#ifdef SIMULATION_MODE
            print(color::yellow, "  === FLOOR CHANGED! ===", color::reset);
#endif
            storedStairCoords.clear();
            storedElevatorCoords.clear();

            new_floor_start_index = compass_analyzer.gyro_history.size();

            if (elevationMode == 2) {  // 엘리베이터
#ifdef SIMULATION_MODE
                print("  Elevator mode, gyro=", gyro);
#endif
                int roundedGyro = static_cast<int>(round(gyro));
                json coords = ::elevatorInfo[std::to_string(roundedGyro)];
#ifdef SIMULATION_MODE
                print("  Elevator coords found: ", coords.empty() ? "NO" : "YES");
#endif
                if (!coords.empty()) {
                    for (auto& pt : coords) {
                        storedElevatorCoords.push_back({ pt[0], pt[1] });
                    }
                }
            }
            else {  // 계단
                stairElevation = (currentFloor > previousFloor) ? "상승" : "하강";
#ifdef SIMULATION_MODE
                print("  Stairs mode: ", stairElevation);
                print("  Looking for stairsInfo[", previousFloor, "][", stairElevation, "]");
#endif

                if (::stairsInfo.contains(std::to_string(previousFloor)) &&
                    ::stairsInfo[std::to_string(previousFloor)].contains(stairElevation)) {

                    json modeInfo = ::stairsInfo[std::to_string(previousFloor)][stairElevation];
#ifdef SIMULATION_MODE
                    print(color::green, "  stairsInfo found!", color::reset);
#endif

                    std::vector<int> directions;
                    for (auto& el : modeInfo.items()) {
                        int d = std::stoi(el.key());
                        directions.push_back(d);
                    }

#ifdef SIMULATION_MODE
                    print("  Available directions: ", directions.size());
                    for (int d : directions) {
                        print("    - ", d);
                    }
#endif

                    int bestDir = directions[0];
                    double minDiff = fabs(gyro - bestDir);
                    for (int d : directions) {
                        double diff = fabs(gyro - d);
                        if (diff > 180) diff = 360 - diff;
                        if (diff < minDiff) {
                            minDiff = diff;
                            bestDir = d;
                        }
                    }

#ifdef SIMULATION_MODE
                    print("  Best direction: ", bestDir, " (gyro was ", gyro, ")");
#endif

                    json coords = modeInfo[std::to_string(bestDir)];
                    if (!coords.empty()) {
                        for (auto& pt : coords) {
                            storedStairCoords.push_back({ pt[0], pt[1] });
#ifdef SIMULATION_MODE
                            print("    Stair coord: (", pt[0], ", ", pt[1], ")");
#endif
                        }
                    }

                    // 가장 가까운 계단 좌표 선택
                    if (storedStairCoords.size() >= 2) {
                        double minDist = std::numeric_limits<double>::max();
                        Point closestStairPt;
                        bool foundOne = false;

                        for (const auto &stairPt : storedStairCoords) {
                            int dx = stairPt.first - cur_pos_global.first;
                            int dy = stairPt.second - cur_pos_global.second;
                            double dist = sqrt(dx * dx + dy * dy) * 0.1;
#ifdef SIMULATION_MODE
                            print("    Distance to (", stairPt.first, ", ", stairPt.second, "): ", dist, "m");
#endif
                            if (dist < minDist) {
                                minDist = dist;
                                closestStairPt = {stairPt.first, stairPt.second};
                                foundOne = true;
                            }
                        }
                        if (foundOne) {
                            std::vector<Point> updatedStairCoords;
                            updatedStairCoords.push_back(closestStairPt);
                            storedStairCoords = updatedStairCoords;
#ifdef SIMULATION_MODE
                            print(color::green, "  Closest stair selected: (", closestStairPt.first, ", ",
                                  closestStairPt.second, ")", color::reset);
#endif
                        }
                    }
                } else {
#ifdef SIMULATION_MODE
                    print(color::red, "  stairsInfo NOT found for floor ", previousFloor, color::reset);
#endif
                }
            }

            // 새 층 맵 로드
            std::string floorText = (currentFloor < 0 ? "B" + std::to_string(abs(currentFloor)) : std::to_string(currentFloor)) + "F";
            std::string binaryMapFile = testbedPath + "/zeromap_" + floorText + " - origin(" +
                                        std::to_string(originMap[currentFloor].first) + "," +
                                        std::to_string(originMap[currentFloor].second) + ").txt";
            std::string indoorMapFile = testbedPath + "/indoor_map_" + floorText + ".json";

#ifdef SIMULATION_MODE
            print("  Loading new map: ", floorText);
            print("    Binary map: ", binaryMapFile);
            print("    Indoor map: ", indoorMapFile);
#endif

            BitMatrix newBinaryMap = readBitMatrixFromTextFile(binaryMapFile.c_str());
            MapData newMapData = loadMapData(indoorMapFile.c_str());

#ifdef SIMULATION_MODE
            print("    Map loaded: ", newBinaryMap.height, "x", newBinaryMap.width);
#endif

            updateMaps(newBinaryMap, newMapData);
            resetHistory();

            std::vector<std::pair<int, int>>& coords = !storedElevatorCoords.empty() ? storedElevatorCoords : storedStairCoords;

#ifdef SIMULATION_MODE
            print("  Arrival coords count: ", coords.size());
#endif

            Point arrival = {
                    coords.empty() ? 0 : coords[0].first - originMap[currentFloor].first,
                    coords.empty() ? 0 : coords[0].second - originMap[currentFloor].second
            };

#ifdef SIMULATION_MODE
            print("  Origin for floor ", currentFloor, ": (", originMap[currentFloor].first, ", ",
                  originMap[currentFloor].second, ")");
            print("  Arrival point (local): (", arrival.first, ", ", arrival.second, ")");
#endif

            newFloorMode = true;
            roi_center = arrival;
            cur_pos = arrival;
            reSearchFlag = false;

#ifdef SIMULATION_MODE
            print(color::green, "  newFloorMode=true, roi_center=(", roi_center.first, ", ",
                  roi_center.second, ")", color::reset);
#endif
        }
    }

    void updatePosition(double gyro, double stepLength, vector<Point>& target_history, Point& target_pos) {
        double rad = gyro * M_PI / 180.0;
        double new_x = target_pos.first - sin(rad) * stepLength * 10;
        double new_y = target_pos.second + cos(rad) * stepLength * 10;

        // stepLength를 기반으로 분할 수 결정
        int divisions = round(stepLength * 10);
        divisions = max(divisions, 1);
        divisions = min(divisions, 100);  // 최대값 제한 추가

        // 보간: 중간 경로 기록
        for (int i = 1; i <= divisions; i++) {
            int interp_x = round(target_pos.first + (new_x - target_pos.first) / divisions * i);
            int interp_y = round(target_pos.second + (new_y - target_pos.second) / divisions * i);

            // 범위 체크 추가
            if (abs(interp_x) < 100000 && abs(interp_y) < 100000) {
                target_history.push_back({interp_x, interp_y});
            }
        }

        target_pos = {static_cast<int>(round(new_x)), static_cast<int>(round(new_y))};
    }

    double getCurrentOrientation() const {
        return drift_tracker.getCurrentOffset();
    }

    const HealthMetrics& getHealthMetrics() const {
        return health_metrics;
    }

    const vector<OrientationHypothesis>& getHypotheses() const {
        return orientation_hypotheses;
    }

    double getCompassTrend() const {
        return compass_analyzer.getRobustTrend();
    }

    pair<double, Point> calculatePosition(const vector<Point>& path_to_use, int& out_max_similarity,
                                          int& out_total_similarity, vector<Point>& out_candidate_positions) {
#ifdef SIMULATION_MODE
        auto start_time = std::chrono::high_resolution_clock::now();
//        print(color::green, "  [Calculate Position]", color::reset);
//        print("    - PDR path size:", path_to_use.size(), "points");
//        print("    - newFloorMode:", newFloorMode ? "YES" : "NO");
//        print("    - roi_center: (", roi_center.first, ", ", roi_center.second, ")");
#endif


        if (path_to_use.empty()) {
            out_max_similarity = -1;
            out_total_similarity = 0;
            out_candidate_positions = {{-1, -1}};
            return {99999, {0, 0}};
        }

        int min_x = numeric_limits<int>::max(), min_y = numeric_limits<int>::max();
        for (const auto& pos : path_to_use) {
            min_x = min(min_x, pos.first);
            min_y = min(min_y, pos.second);
        }

        vector<Point> normalized_pos;
        normalized_pos.reserve(path_to_use.size());

        for (const auto& pos : path_to_use) {
            normalized_pos.push_back({pos.first - min_x, pos.second - min_y});
        }

        int total_similarity = normalized_pos.size();

#ifdef SIMULATION_MODE
        auto bitmap_start = std::chrono::high_resolution_clock::now();
#endif

        BitMatrix test_matrix = create_bitmatrix_from_coordinates(normalized_pos);

        if (test_matrix.height == 0 || test_matrix.width == 0) {
            out_max_similarity = -1;
            out_total_similarity = 0;
            out_candidate_positions = {{-1, -1}};
            return {99999, {0, 0}};
        }

#ifdef SIMULATION_MODE
        auto bitmap_end = std::chrono::high_resolution_clock::now();
        auto bitmap_duration = std::chrono::duration_cast<std::chrono::microseconds>(bitmap_end - bitmap_start);
//        print("    - Test matrix:", test_matrix.height, "x", test_matrix.width,
//              "(" + color::gray + std::to_string(bitmap_duration.count()) + "μs" + color::reset + ")");
#endif

        int roi_center_row = -1, roi_center_col = -1;
        int effective_roi = newFloorMode ? 50 : roi_radius;

        std::string mode_str = "Normal";
        if (reSearchFlag) {
            roi_center_row = roi_center.first;
            roi_center_col = roi_center.second;
            effective_roi = reSearchRoi;
            mode_str = "ReSearch";
        }
        else if (newFloorMode) {
            roi_center_row = roi_center.first;
            roi_center_col = roi_center.second;
            mode_str = "NewFloor";
        } else if (has_last_matched) {
            roi_center_row = 85;
            roi_center_col = 600;
        }

        Point T_anchor = normalized_pos.front();

        auto [max_similarity, max_positions] = sliding_window_similarity_optimized(
                binary_map, test_matrix,
                roi_center_row, roi_center_col, effective_roi,
                T_anchor
        );


        out_max_similarity = max_similarity;
        out_total_similarity = total_similarity;

#ifdef SIMULATION_MODE
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        print(color::green, "  [Position Calculated]", color::yellow,
              "Total: " + std::to_string(duration.count()) + "ms", color::reset);
#endif

        if (max_similarity > 0 && !max_positions.empty()) {

            vector<Point> result_positions;
            result_positions.reserve(max_positions.size());


            for (size_t idx = 0; idx < max_positions.size(); idx++) {
                const auto& pos = max_positions[idx];

                int result_x = pos.first + normalized_pos.back().first;
                int result_y = pos.second + normalized_pos.back().second;


                if (result_x >= 0 && result_x < binary_map.height &&
                    result_y >= 0 && result_y < binary_map.width) {
                    result_positions.push_back({result_x, result_y});
                } else {
                }
            }


            if (result_positions.empty()) {
                out_candidate_positions = {{-1, -1}};
                return {-1, {-1, -1}};
            }

            out_candidate_positions = result_positions;

            auto [mean_distance, mean_coord] = analyzeCoordinates(result_positions);
            has_last_matched = true;

            Point final_result = {
                    static_cast<int>(mean_coord.first + originMap[currentFloor].first),
                    static_cast<int>(mean_coord.second + originMap[currentFloor].second)
            };


            return {mean_distance, final_result};
        } else {
            out_candidate_positions = {{-1, -1}};
            return {-1, {-1, -1}};
        }
    }

    // 원래 calculatePosition 함수를 래핑
    pair<double, Point> calculatePosition() {
        int max_sim, total_sim;
        vector<Point> candidates;
        auto result = calculatePosition(history_of_pos, max_sim, total_sim, candidates);
        cur_max_similarity = max_sim;
        very_max_similarity = total_sim;
        return result;
    }

    // 다중 가설 테스트
    // PositionTracker 클래스 내부
    // PositionTracker 클래스 내부
    OrientationHypothesis testMultipleOrientations(const vector<double>& candidates) {
        OrientationHypothesis best;
        best.confidence = -1;
        best.angle = 0;
        best.last_match_quality = 0;

        // 현재 history가 비어있으면 테스트 불가
        if (history_of_pos.empty()) {
            cout << "Warning: Cannot test orientations, history is empty" << endl;
            return best;
        }

        // 현재 history를 기준으로 테스트 (pendingGyros 사용하지 않음)
        int best_max_sim = 0;
        int best_total_sim = 0;

        for (double test_angle : candidates) {
            try {
                // 회전 오프셋 계산
                double rotation_offset = test_angle - drift_tracker.getCurrentOffset();

                // 현재 경로를 회전
                vector<Point> rotated_history;
                rotated_history.reserve(history_of_pos.size());

                // 중심점 계산
                double center_x = 0, center_y = 0;
                for (const auto& p : history_of_pos) {
                    center_x += p.first;
                    center_y += p.second;
                }
                center_x /= history_of_pos.size();
                center_y /= history_of_pos.size();

                // 회전 적용
                double rad = rotation_offset * M_PI / 180.0;
                double cos_theta = cos(rad);
                double sin_theta = sin(rad);

                for (const auto& p : history_of_pos) {
                    double dx = p.first - center_x;
                    double dy = p.second - center_y;

                    int new_x = round(center_x + dx * cos_theta - dy * sin_theta);
                    int new_y = round(center_y + dx * sin_theta + dy * cos_theta);

                    rotated_history.push_back({new_x, new_y});
                }

                // 매칭 시도
                int max_sim = 0, total_sim = 0;
                vector<Point> candidates_pos;

                auto [dist, coord] = calculatePosition(rotated_history, max_sim, total_sim, candidates_pos);

                double quality = (total_sim > 0) ? (double)max_sim / total_sim : 0;

                if (quality > best.last_match_quality) {
                    best.angle = test_angle;
                    best.last_match_quality = quality;
                    best.confidence = quality;
                    best_max_sim = max_sim;
                    best_total_sim = total_sim;
                }

            } catch (const std::exception& e) {
                cout << "Exception testing angle " << test_angle << ": " << e.what() << endl;
                continue;
            } catch (...) {
                cout << "Unknown exception testing angle " << test_angle << endl;
                continue;
            }
        }

        cout << "Best orientation: " << best.angle << " (quality: " << best.last_match_quality << ")" << endl;

        return best;
    }

    void updateHealthMetrics(double compass_trend, double compass_stability,
                             bool loc_success, double match_quality) {

        health_metrics.compass_reliability = (compass_trend > 0) ? compass_stability : 0.0;

        recent_loc_results.push_back(loc_success);
        if (recent_loc_results.size() > 20) {
            recent_loc_results.pop_front();
        }

        int success_count = count(recent_loc_results.begin(), recent_loc_results.end(), true);

        health_metrics.localization_success_rate = (double)success_count / recent_loc_results.size();

        health_metrics.drift_magnitude = abs(drift_tracker.accumulated_drift);
        double drift_health = 1.0 / (1.0 + health_metrics.drift_magnitude / 10.0);

        health_metrics.overall_health =
                0.3 * health_metrics.compass_reliability +
                0.5 * health_metrics.localization_success_rate +
                0.2 * drift_health;

    }

    ProcessResult processStep(float gyro, float compass, float stepLength,
                              int stepCount, int floor, float arrivedGyroValue, int elevationMode) {
#ifdef SIMULATION_MODE
        auto step_start = std::chrono::high_resolution_clock::now();
        print("");
        print(color::magenta, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", color::reset);
        print(color::magenta, "  Step", stepCount, color::reset);
        print("  Raw Input - gyro:", gyro, "compass:", compass, "stepLen:", stepLength);
        print(color::magenta, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", color::reset);
#endif

#ifndef SIMULATION_MODE
        __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "=== Step %d ===", stepCount);
        __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "Raw compass=%.2f, gyro=%.2f", compass, gyro);
        __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "testbedPath=%s", testbedPath.c_str());
#endif

        currentFloor = floor;
        if (previousFloor == -9) previousFloor = currentFloor;

        double adjustedCompass = fmod((compass - 339.38 + 360), 360);
        if (this->testbedPath.find("enb") != std::string::npos) {
            adjustedCompass = fmod((compass - 339.38 - 90 + 360), 360);
        } else if (this->testbedPath.find("anamhosp") != std::string::npos) {
            adjustedCompass = fmod((compass - 339.38 - 70 + 360), 360);
        }

        adjustedCompass = normalizeAngle(adjustedCompass);  // 안전하게 정규화
#ifndef SIMULATION_MODE
        __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "Adjusted compass=%.2f", adjustedCompass);
#endif

        // 나침반 데이터 수집
        compass_analyzer.addMeasurement(stepCount, adjustedCompass, gyro, stepLength);  // stepLength 추가

#ifndef SIMULATION_MODE
        __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "direction_offsets size=%zu",
                            compass_analyzer.direction_offsets.size());
        if (!compass_analyzer.direction_offsets.empty()) {
            __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "Latest offset=%.2f",
                                compass_analyzer.direction_offsets.back());
        }
#endif
        // 초기 캘리브레이션 단계 (30걸음까지)
        if (stepCount <= 30) {
            pendingGyros.push_back(gyro);
            pendingStepLengths.push_back(stepLength);

            // processStep 함수 내부, 30스텝 지점
            if (stepCount == 30) {
                // 오프셋 경향성 분석 (compass - gyro)
                double offset_trend = compass_analyzer.getRobustTrend();
#ifdef SIMULATION_MODE
                print(color::cyan, "Initial orientation estimation...", color::reset);
                print("  Offset trend (compass - gyro):", offset_trend);
#endif

#ifndef SIMULATION_MODE
                __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "==============================");
                __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "Step 30 - Calibration!");
                __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "offset_trend=%.2f", offset_trend);
                __android_log_print(ANDROID_LOG_INFO, "ZeroMap", "==============================");

                // 히스토그램 분석 결과도 출력
                for (size_t i = 0; i < compass_analyzer.direction_offsets.size(); i++) {
                    if (i % 5 == 0) {  // 5개마다 출력
                        __android_log_print(ANDROID_LOG_DEBUG, "ZeroMap", "offset[%zu]=%.2f",
                                            i, compass_analyzer.direction_offsets[i]);
                    }
                }
#endif
                // baseline = offset_trend (이게 보정값)
                // 절대 방향 = gyro + offset_trend
                drift_tracker.setBaseline(offset_trend);

                orientation_hypotheses.clear();

                for (int offset = -20; offset <= 20; offset += 5) {
                    double angle = fmod(offset_trend + offset + 360, 360);
                    OrientationHypothesis hypo(angle);
                    hypo.confidence = (offset == 0) ? 0.8 : 0.3;
                    orientation_hypotheses.push_back(hypo);
                }
#ifdef SIMULATION_MODE
                print(color::green, "  Offset (compass - gyro):", offset_trend, "degrees", color::reset);
#endif
                // PDR 경로 재생성
                for (size_t i = 0; i < pendingGyros.size(); i++) {
                    double adjusted = pendingGyros[i] + offset_trend;  // gyro + offset
                    this->updatePosition(adjusted, pendingStepLengths[i], history_of_pos, cur_pos);
                }

                pendingGyros.clear();
                pendingStepLengths.clear();

                return {0, 0, 99999, false, 0, false, {}, -1, {}, 0, 0, false, {}, {}, 0.0};
            }


            return {0, 0, 99999, false, 0, false, {}, -1, {}, 0, 0, false, {}, {}, 0.0};
        }

        // 정상 작동 단계 (31 스텝~)

// 20스텝마다 경향성 업데이트
        if (stepCount > 30 && stepCount % 20 == 0) {
            double new_trend = compass_analyzer.getRobustTrend();
            if (new_trend > -180) {
                double current_baseline = drift_tracker.baseline_offset;
                double diff = new_trend - current_baseline;

                while (diff > 180) diff -= 360;
                while (diff < -180) diff += 360;

                if (abs(diff) > 2.0) {
                    drift_tracker.baseline_offset = new_trend;
#ifdef SIMULATION_MODE
                    print(color::yellow, "  Baseline updated:", current_baseline, "->", new_trend, "deg", color::reset);
#endif
                }
            }
        }

// 현재 경향성 기준값
        double base_orientation = drift_tracker.baseline_offset;

// 경향성 기준으로 PDR 경로 생성 (기본)
        double adjustedGyro = gyro + base_orientation;
        this->setArrivedInfo(adjustedGyro, elevationMode);
        this->updatePosition(adjustedGyro, stepLength, history_of_pos, cur_pos);

        previousFloor = floor;

        if (!didConverge) {

            // 사용할 데이터 범위 결정
            size_t start_idx = newFloorMode ? new_floor_start_index : 0;
            size_t end_idx = compass_analyzer.gyro_history.size();

            // ========== 1단계: 10도 간격 대략 탐색 (5개) ==========
            vector<int> coarse_deltas = {-20, -10, 0, 10, 20};

            int coarse_best_delta = 0;
            int coarse_best_sim = 0;
            // 회전 테스트 시작 전

#pragma omp parallel for
            for (size_t idx = 0; idx < coarse_deltas.size(); idx++) {
                int delta = coarse_deltas[idx];
                double test_orientation = base_orientation + delta;

                // PDR 경로 생성
                vector<Point> test_history;
                Point test_pos = {0, 0};

                // start_idx부터 end_idx까지만 사용
                for (size_t i = start_idx; i < end_idx; i++) {
                    double test_gyro = compass_analyzer.gyro_history[i] + test_orientation;
                    double step_len = compass_analyzer.step_lengths[i];

                    double rad = test_gyro * M_PI / 180.0;
                    double new_x = test_pos.first - sin(rad) * step_len * 10;
                    double new_y = test_pos.second + cos(rad) * step_len * 10;

                    int divisions = round(step_len * 10);
                    divisions = std::max(divisions, 1);
                    divisions = std::min(divisions, 100);

                    for (int d = 1; d <= divisions; d++) {
                        int interp_x = round(test_pos.first + (new_x - test_pos.first) / divisions * d);
                        int interp_y = round(test_pos.second + (new_y - test_pos.second) / divisions * d);
                        test_history.push_back({interp_x, interp_y});
                    }

                    test_pos = {static_cast<int>(round(new_x)), static_cast<int>(round(new_y))};
                }

                // 매칭
                if (test_history.size() > 10) {
                    int max_sim = 0, total_sim = 0;
                    vector<Point> candidates;
                    auto [dist, coord] = calculatePosition(test_history, max_sim, total_sim, candidates);

#pragma omp critical
                    {
                        if (max_sim > coarse_best_sim) {
                            coarse_best_sim = max_sim;
                            coarse_best_delta = delta;
                        }
                    }
                }
            }


            // ========== 2단계: 최적 구간에서 2도 간격 세밀 탐색 (5개) ==========
            vector<int> fine_deltas;
            for (int d = coarse_best_delta - 4; d <= coarse_best_delta + 4; d += 2) {
                fine_deltas.push_back(d);
            }

            // Sticky angle이 있으면 해당 delta도 포함
            if (has_sticky_orientation) {
                int sticky_delta = static_cast<int>(round(sticky_best_orientation - base_orientation));
                // 이미 포함되어 있지 않으면 추가
                if (std::find(fine_deltas.begin(), fine_deltas.end(), sticky_delta) == fine_deltas.end()) {
                    fine_deltas.push_back(sticky_delta);
#ifdef SIMULATION_MODE
                    print(color::gray, "  [Sticky] Added delta ", sticky_delta, " to fine_deltas", color::reset);
#endif
                }
            }

            double best_orientation = base_orientation;
            int best_max_sim = 0;
            int best_total_sim = 0;
            Point best_coord = {0, 0};
            vector<Point> best_candidates;
            vector<Point> best_history;

            struct RotationResult {
                double orientation;
                int max_sim;
                int total_sim;
                Point coord;
                vector<Point> candidates;
                vector<Point> history;
            };

            vector<RotationResult> results(fine_deltas.size());

#pragma omp parallel for
            for (size_t idx = 0; idx < fine_deltas.size(); idx++) {
                int delta = fine_deltas[idx];
                double test_orientation = base_orientation + delta;

                // PDR 경로 생성
                vector<Point> test_history;
                Point test_pos = {0, 0};

                for (size_t i = start_idx; i < end_idx; i++) {
                    double test_gyro = compass_analyzer.gyro_history[i] + test_orientation;
                    double step_len = compass_analyzer.step_lengths[i];

                    double rad = test_gyro * M_PI / 180.0;
                    double new_x = test_pos.first - sin(rad) * step_len * 10;
                    double new_y = test_pos.second + cos(rad) * step_len * 10;

                    int divisions = round(step_len * 10);
                    divisions = max(divisions, 1);
                    divisions = min(divisions, 100);

                    for (int d = 1; d <= divisions; d++) {
                        int interp_x = round(test_pos.first + (new_x - test_pos.first) / divisions * d);
                        int interp_y = round(test_pos.second + (new_y - test_pos.second) / divisions * d);
                        test_history.push_back({interp_x, interp_y});
                    }

                    test_pos = {static_cast<int>(round(new_x)), static_cast<int>(round(new_y))};
                }

                results[idx].orientation = test_orientation;
                results[idx].history = test_history;
                results[idx].max_sim = 0;

                if (test_history.size() > 10) {
                    int max_sim = 0, total_sim = 0;
                    vector<Point> candidates;
                    auto [dist, coord] = calculatePosition(test_history, max_sim, total_sim, candidates);

                    results[idx].max_sim = max_sim;
                    results[idx].total_sim = total_sim;
                    results[idx].coord = coord;
                    results[idx].candidates = candidates;
                }
            }

            // 최적 결과 선택 (Sticky Angle 전략 적용)
            int current_orientation_sim = 0;  // 현재 sticky 각도의 similarity

            for (const auto& res : results) {
                // 현재 sticky 각도의 similarity 기록
                if (has_sticky_orientation &&
                    std::abs(res.orientation - sticky_best_orientation) < 1.0) {
                    current_orientation_sim = res.max_sim;
                }

                if (res.max_sim > best_max_sim) {
                    best_max_sim = res.max_sim;
                    best_total_sim = res.total_sim;
                    best_orientation = res.orientation;
                    best_coord = res.coord;
                    best_candidates = res.candidates;
                    best_history = res.history;
                } else if (res.max_sim == best_max_sim && res.max_sim > 0) {
                    double current_diff = abs(best_orientation - base_orientation);
                    double new_diff = abs(res.orientation - base_orientation);

                    if (new_diff < current_diff) {
                        best_orientation = res.orientation;
                        best_total_sim = res.total_sim;
                        best_coord = res.coord;
                        best_candidates = res.candidates;
                        best_history = res.history;
                    }
                }
            }

// Sticky Angle 전략: 15% 이상 개선되지 않으면 기존 각도 유지
            if (has_sticky_orientation && current_orientation_sim > 0) {
                double improvement_ratio = (double)(best_max_sim - current_orientation_sim) / current_orientation_sim;

                if (improvement_ratio < 0.0) {
                    // 개선이 충분하지 않으면 기존 각도 유지
                    for (const auto& res : results) {
                        if (std::abs(res.orientation - sticky_best_orientation) < 1.0) {
                            best_orientation = res.orientation;
                            best_max_sim = res.max_sim;
                            best_total_sim = res.total_sim;
                            best_coord = res.coord;
                            best_candidates = res.candidates;
                            best_history = res.history;
                            break;
                        }
                    }
#ifdef SIMULATION_MODE
                    print(color::yellow, "  [Sticky] Keeping orientation ", sticky_best_orientation,
                          " (improvement only ", fixed(improvement_ratio * 100, 1), "%)", color::reset);
#endif
                } else {
                    // 충분히 개선되면 새 각도로 업데이트
                    sticky_best_orientation = best_orientation;
                    sticky_best_similarity = best_max_sim;
#ifdef SIMULATION_MODE
                    print(color::green, "  [Sticky] Switching to ", best_orientation,
                          " (improvement ", fixed(improvement_ratio * 100, 1), "%)", color::reset);
#endif
                }
            } else if (best_max_sim > 0) {
                // 첫 유효한 결과로 sticky 초기화
                sticky_best_orientation = best_orientation;
                sticky_best_similarity = best_max_sim;
                has_sticky_orientation = true;
#ifdef SIMULATION_MODE
                print(color::cyan, "  [Sticky] Initialized to ", best_orientation, color::reset);
#endif
            }

// processStep 함수 내, 회전 테스트 후
#ifdef SIMULATION_MODE
            print(color::cyan, "  Rotation test: base=", base_orientation,
                  " coarse_best=", coarse_best_delta,
                  " final=", best_orientation, " (delta=", best_orientation - base_orientation, ")",
                  color::reset);
            print("  best_max_sim=", best_max_sim, " best_total_sim=", best_total_sim);
            print("  best_coord: (", best_coord.first, ", ", best_coord.second, ")");
            print("  candidates count:", best_candidates.size());
#endif

            // ========== 나머지 기존 코드 ==========
            cur_max_similarity = best_max_sim;
            very_max_similarity = best_total_sim;

            double mean_distance = 0;
            Point final_coord = best_coord;
            vector<Point> candidates_positions = best_candidates;

            if (!candidates_positions.empty() && candidates_positions[0].first != -1) {
                auto [dist, mean_coord] = analyzeCoordinates(candidates_positions);
                mean_distance = dist;
            } else {
                mean_distance = 99999;
            }

            bool loc_success = (mean_distance < 5.0);
            double match_quality = (very_max_similarity > 0) ?
                                   (double)cur_max_similarity / very_max_similarity : 0;

            double compass_trend = compass_analyzer.getRobustTrend();
            double compass_stability = compass_analyzer.getTemporalStability();
            updateHealthMetrics(compass_trend, compass_stability, loc_success, match_quality);

            drift_tracker.updateDriftEstimate(stepCount, gyro, compass_trend,
                                              loc_success, match_quality);

            Point mapmatched_coord = final_coord;
            bool mapmatching_applied = false;

            if (has_map_matching_config && final_coord.first > 0 && final_coord.second > 0) {
                MapPoint current_pos(final_coord.first, final_coord.second);
                MapPoint matched_pos = getMapMatchingResultHybrid(current_pos, map_data, map_matching_config);
                mapmatched_coord = {static_cast<int>(matched_pos.x), static_cast<int>(matched_pos.y)};

                if (mapmatched_coord.first != final_coord.first ||
                    mapmatched_coord.second != final_coord.second) {
                    mapmatching_applied = true;
                }
            }

            ProcessResult result {
                    static_cast<double>(mapmatched_coord.first),
                    static_cast<double>(mapmatched_coord.second),
                    mean_distance, true,
                    static_cast<int>(best_orientation),
                    true, {}, -1, {},
                    static_cast<double>(final_coord.first),
                    static_cast<double>(final_coord.second),
                    mapmatching_applied,
                    candidates_positions,
                    best_history,
                    base_orientation
            };

            cur_pos_global = {mapmatched_coord.first, mapmatched_coord.second};
            cur_distance = mean_distance;

#ifdef SIMULATION_MODE
            auto step_end = std::chrono::high_resolution_clock::now();
            auto step_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);
            print("");
            print(color::magenta, "  [Step Complete]", color::yellow,
                  "Time: " + std::to_string(step_duration.count()) + "ms", color::reset);
            print(color::magenta, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", color::reset);
#endif

            return result;

        } else {
            double adjustedGyro = gyro + drift_tracker.getCurrentOffset();
            double new_x = cur_pos_global.first + sin(adjustedGyro * M_PI / 180.0) * stepLength * 10;
            double new_y = cur_pos_global.second + cos(adjustedGyro * M_PI / 180.0) * stepLength * 10;
            cur_pos_global = {round(new_x), round(new_y)};

            return {cur_pos_global.first, cur_pos_global.second, 0.1, true,
                    static_cast<int>(drift_tracker.getCurrentOffset()), true, {}, -1, {},
                    cur_pos_global.first, cur_pos_global.second, false, {}, {}, base_orientation};
        }
    }

    // 시뮬레이션용 getter 함수들 추가
    const vector<Point>& getHistoryOfPos() const {
        return history_of_pos;
    }

    Point getCurPos() const {
        return cur_pos;
    }

// 이 함수 추가
    pair<double, Point> calculatePositionWithCandidates(vector<Point>& out_candidates) {
        if (history_of_pos.size() < 1) {
            out_candidates.clear();
            return {99999, {-1, -1}};
        }

        int max_sim, total_sim;
        auto result = calculatePosition(history_of_pos, max_sim, total_sim, out_candidates);
        cur_max_similarity = max_sim;
        very_max_similarity = total_sim;
        return result;
    }
};

#ifndef SIMULATION_MODE
std::vector<std::string> listAssetFiles(const std::string& directory) {
    std::vector<std::string> result;

    AAssetDir* assetDir = AAssetManager_openDir(g_assetManager, directory.c_str());
    if (assetDir == nullptr) {
        return result;
    }

    const char* filename;
    while ((filename = AAssetDir_getNextFileName(assetDir)) != nullptr) {
        result.push_back(std::string(filename));
    }

    AAssetDir_close(assetDir);
    return result;
}

// assets에서 파일 내용을 읽는 함수
std::string readAssetFile(const std::string& filename) {
    AAsset* asset = AAssetManager_open(g_assetManager, filename.c_str(), AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        return "";
    }

    off_t length = AAsset_getLength(asset);
    std::string content;
    content.resize(length);

    AAsset_read(asset, &content[0], length);
    AAsset_close(asset);

    return content;
}

// 원점 정보를 로드하는 함수 (디렉토리 지정)
void loadOriginOfMap(std::map<int, std::pair<int, int>>& originMap, const std::string& directory) {
    std::regex pattern(R"(zeromap_((B)?(\d+))F - origin\((\d+),(\d+)\)\.txt)");
    // assets/{directory} 폴더에서 파일 목록 가져오기
    std::vector<std::string> fileList = listAssetFiles(directory);
    for (const auto& filename : fileList) {
        std::smatch match;
        if (std::regex_match(filename, match, pattern)) {
            int floorNumber = 0;
            if (match[2].matched) {
                floorNumber = -std::stoi(match[3].str());
            } else {
                floorNumber = std::stoi(match[3].str());
            }
            int x = std::stoi(match[4].str());
            int y = std::stoi(match[5].str());
            originMap[floorNumber] = {x, y};
        }
    }
}
#endif


// ====================================================================
// JNI 인터페이스 구현: Java_com_example_fifth_maplocationlib_NativeLib_processStep
// ====================================================================
static PositionTracker* gTracker = nullptr;

#ifndef SIMULATION_MODE
extern "C"
JNIEXPORT void JNICALL
Java_com_fifth_maplocationlib_NativeLib_initializeEngine(JNIEnv *env, jobject, jint floor, jstring testbed) {
// jstring -> std::string 변환
    const char* tb = env->GetStringUTFChars(testbed, nullptr);
    std::string testbedStr = tb ? tb : "";
    env->ReleaseStringUTFChars(testbed, tb);

// 원점 정보 로드 (testbed 디렉토리 내에서 검색)
    loadOriginOfMap(originMap, testbedStr);

    __android_log_print(ANDROID_LOG_INFO, "LOG_CHECK", "===== Origin Map Contents =====");
    __android_log_print(ANDROID_LOG_INFO, "LOG_CHECK", "Total entries: %d, %s", originMap.size(), testbedStr.c_str());
    for (const auto& entry : originMap) {
        __android_log_print(ANDROID_LOG_INFO, "LOG_CHECK", "Floor %d: Origin at (%d, %d)",
                            entry.first, entry.second.first, entry.second.second);
    }
    __android_log_print(ANDROID_LOG_INFO, "LOG_CHECK", "==============================");

// Load floor-change areas (stairs & elevator) from assets: <testbed>/enb/... json files
    {
        std::string stairsPath = testbedStr + "/floorchange_stairs_areas.json";
        std::string elevatorPath = testbedStr + "/floorchange_elevator_areas.json";

        json s = loadJsonData(stairsPath.c_str());
        if (!s.is_discarded() && !s.is_null()) {
            stairsInfo = s;
            __android_log_print(ANDROID_LOG_INFO, "LOG_CHECK", "Loaded stairs areas: %s", stairsPath.c_str());
        } else {
            __android_log_print(ANDROID_LOG_INFO, "LOG_CHECK", "Failed to load stairs areas: %s", stairsPath.c_str());
        }

        json e = loadJsonData(elevatorPath.c_str());
        if (!e.is_discarded() && !e.is_null()) {
            elevatorInfo = e;
            __android_log_print(ANDROID_LOG_INFO, "LOG_CHECK", "Loaded elevator areas: %s", elevatorPath.c_str());
        } else {
            __android_log_print(ANDROID_LOG_INFO, "LOG_CHECK", "Failed to load elevator areas: %s", elevatorPath.c_str());
        }
    }

    int start_floor = floor;
    string floorText = (start_floor < 0 ? "B" + to_string(abs(start_floor)) : to_string(start_floor)) + "F";
    string binaryMapFile = testbedStr + "/zeromap_" + floorText + " - origin(" + to_string(originMap[start_floor].first) + "," + to_string(originMap[start_floor].second) + ").txt";
    string indoorMapFile = testbedStr + "/indoor_map_" + floorText + ".json";

    BitMatrix binaryMap = readBitMatrixFromTextFile(binaryMapFile.c_str());
    MapData mapData = loadMapData(indoorMapFile.c_str());
// 기존과 같이 PositionTracker 객체 생성 후 전역 변수에 저장
    if (gTracker) {
        delete gTracker;
    }
    gTracker = new PositionTracker(binaryMap, mapData, start_floor, testbedStr);
}


// 2. processStep: 센서 데이터(자이로, 보폭, 걸음 수 등)를 받아 위치 계산 결과를 float 배열로 반환
extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_fifth_maplocationlib_NativeLib_processStep(JNIEnv *env, jobject /* this */,
                                                    jfloat gyro,
                                                    jfloat compass,
                                                    jfloat stepLength,
                                                    jint stepCount,
                                                    jint floor,
                                                    jfloat arrivedGyroValue,
                                                    jint elevationMode) {

    if (!gTracker) {
// 엔진이 초기화되지 않은 경우 null 반환 또는 에러 처리
        return nullptr;
    }
    __android_log_print(ANDROID_LOG_INFO, "codetime", "OK2");

// enginePtr를 PositionTracker 포인터로 변환
    PositionTracker* tracker = gTracker;

// 센서 입력을 토대로 processStep() 호출
    ProcessResult res = gTracker->processStep(gyro, compass, stepLength, stepCount, floor, arrivedGyroValue, elevationMode);

    __android_log_print(ANDROID_LOG_INFO, "codetime", "OK3");
// 결과를 jfloatArray로 준비합니다.
// 반환 배열은 [x, y, currentFloor, orientation]로 구성합니다.
// 여기서는 orientation 값으로 gyro를 그대로 사용합니다.


    jfloatArray resultArray = env->NewFloatArray(4);
    if (resultArray == nullptr) return nullptr;
    jfloat out[4];
    out[0] = static_cast<jfloat>(res.x);
    out[1] = static_cast<jfloat>(res.y);
    out[2] = static_cast<jfloat>(res.distance);
    out[3] = static_cast<jfloat>(res.gyroCaliValue);;

    if (res.distance > 10000.0f) {
        out[0] = -1.0f;
        out[1] = -1.0f;
    }
    __android_log_print(ANDROID_LOG_INFO, "result2", "(%f, %f) / %f", out[0], out[1], res.distance);

    env->SetFloatArrayRegion(resultArray, 0, 4, out);
    return resultArray;
}

// 3. 엔진 해제 함수
extern "C"
JNIEXPORT void JNICALL
Java_com_fifth_maplocationlib_NativeLib_destroyEngine(JNIEnv *env, jobject) {
    if (gTracker) {
        delete gTracker;
        gTracker = nullptr;
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_fifth_maplocationlib_NativeLib_reSearchStartInStairs(JNIEnv *env, jobject, jint stairsCoords_x, jint stairsCoords_y) {
    jint new_stairsCoords_x = stairsCoords_x - originMap[gTracker->getCurrentFloor()].first;
    jint new_stairsCoords_y = stairsCoords_y - originMap[gTracker->getCurrentFloor()].second;
    gTracker->reSearchStart({(double)new_stairsCoords_x, (double)new_stairsCoords_y}, 2);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_fifth_maplocationlib_NativeLib_resetGyroCalibration(JNIEnv *env, jobject /* this */) {
    if (gTracker) {
        gTracker->resetGyroCalibration();
    }
}


extern "C"
JNIEXPORT void JNICALL
Java_com_fifth_maplocationlib_NativeLib_reSearchStart(JNIEnv *env, jobject, jint search_range) {
    gTracker->reSearchStart(cur_pos_global, search_range);
}
#endif

#ifdef SIMULATION_MODE

// 시뮬레이션 전용 함수들

struct SimulationData {
    float gyro, compass, stepLen;
    int floor;
};

std::vector<SimulationData> loadSimulationData(const std::string& filepath) {
    std::vector<SimulationData> data;
    std::ifstream file(filepath);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        float x, y, gyro, compass, stepLen;
        int floor;
        std::string dummy;

        // $x\t$y\t$gyro\t$compass\t$stepLen\t$floor\t-\t-
        ss >> x >> y >> gyro >> compass >> stepLen >> floor >> dummy >> dummy;

        data.push_back({gyro, compass, stepLen, floor});
    }
    return data;
}

void loadOriginOfMap(const std::string& directoryPath) {
    std::map<int, std::pair<int, int>> mapFileName;
    // 정규표현식 패턴:
    // zeromap_((B)?(\d+))F - origin\((\d+),(\d+)\).txt
    // 그룹 설명:
    // 1. 전체 층 정보 ("B1" 또는 "1")
    // 2. "B" 접두사 (지하층이면 매칭)
    // 3. 층 숫자
    // 4,5. origin 좌표 (x, y)
    std::regex pattern(R"(zeromap_((B)?(\d+))F - origin\((\d+),(\d+)\)\.txt)");

    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(filename, match, pattern)) {
            int floorNumber = 0;
            // 그룹 2에 "B"가 매칭되면 지하층으로 판단해 음수값으로 변환 (예: B1 -> -1)
            if (match[2].matched) {
                floorNumber = -std::stoi(match[3].str());
            } else {
                floorNumber = std::stoi(match[3].str());
            }
            int x = std::stoi(match[4].str());
            int y = std::stoi(match[5].str());
            originMap[floorNumber] = {x, y};
        }
    }
}

// 시뮬레이션 결과 저장 구조체
struct StepResult {
    int stepNumber;
    float gyro, compass, stepLen;
    int floor;
    ProcessResult result;
    vector<Point> pdr_history;  // 해당 스텝까지의 PDR 경로
    vector<Point> candidates;   // 후보 위치들
    Point cur_pos;              // 현재 PDR 위치

    // 캘리브레이션 디버깅 정보 추가
    bool calibrationDone;
    vector<double> candidateDiffs;  // 각 후보별 차이값
    int bestCandidateIdx;           // 1등 후보 인덱스
    vector<int> calibHistory;       // 현재까지의 히스토리
    int finalCalibValue;            // 최종 캘리브레이션 값

    // 맵매칭 결과 추가
    Point mapmatched_pos;     // 맵매칭 적용된 위치
    bool has_mapmatching;     // 맵매칭이 적용되었는지 여부

    // 방향 추정 정보 추가
    double current_orientation;
    HealthMetrics health;
    vector<OrientationHypothesis> hypotheses;
    double compass_trend;

    // 이 스텝까지의 나침반/자이로 히스토리 추가
    vector<pair<int, float>> compass_history_snapshot;  // 추가
    vector<float> gyro_history_snapshot;                // 추가
    vector<float> direction_offsets_snapshot;  // 이름 변경
    double accumulated_drift;                           // 추가
    double baseline_offset;

    // 회전 테스트 결과 추가
    vector<Point> best_pdr_history;  // 추가: best 각도로 생성된 PDR 경로
    double best_orientation;          // 추가: best 각도
    double base_orientation;          // 추가: 경향성 기준 각도
};

// 방향 추정 시각화 함수
void visualizeOrientationEstimation(const StepResult& step,
                                    const CompassAnalyzer& compass_analyzer,
                                    const GyroDriftTracker& drift_tracker,
                                    const vector<OrientationHypothesis>& hypotheses,
                                    const HealthMetrics& health) {
    int width = 1200;
    int height = 800;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 제목
    std::string title = "Orientation Estimation - Step " + std::to_string(step.stepNumber);
    cv::putText(canvas, title, cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

    // ========== 좌상단: 절대 방향 추정 히스토그램 ==========
    int hist_x = 50, hist_y = 60, hist_w = 400, hist_h = 200;  // 높이 줄임: 250 → 200

    cv::rectangle(canvas, cv::Point(hist_x, hist_y),
                  cv::Point(hist_x + hist_w, hist_y + hist_h),
                  cv::Scalar(200, 200, 200), 1);

    // ========== 좌상단: 오프셋 히스토그램 ==========
    cv::putText(canvas, "Compass - Gyro Offset Distribution", cv::Point(hist_x + 10, hist_y + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

// 히스토그램 데이터 생성 (1도 단위)
    vector<int> histogram(360, 0);

    for (float offset : compass_analyzer.direction_offsets) {
        double angle = offset + 180;
        int bin = (int)angle % 360;
        if (bin >= 0 && bin < 360) {
            histogram[bin]++;
        }
    }

    int max_val = *max_element(histogram.begin(), histogram.end());
    if (max_val == 0) max_val = 1;

// 경향성 계산 (한 번만)
    double offset_trend = -999;
    int trend_bin = -1;

    if (!compass_analyzer.direction_offsets.empty() &&
        compass_analyzer.direction_offsets.size() >= 20) {
        offset_trend = compass_analyzer.getRobustTrend();
        if (offset_trend > -180) {
            double display_angle = offset_trend + 180;
            trend_bin = (int)display_angle % 360;
        }
    }

// 히스토그램 그리기 (1도 단위)
    for (int i = 0; i < 360; i++) {
        if (histogram[i] == 0) continue;

        int bar_h = (histogram[i] * (hist_h - 40)) / max_val;
        int bar_x = hist_x + 10 + i * (hist_w - 20) / 360;
        int bar_y = hist_y + hist_h - 10 - bar_h;

        // 경향성 빈 주변(±2도)이면 빨간색, 아니면 회색
// 경향성 빈만 빨간색
        bool is_trend = (i == trend_bin);
        cv::Scalar color = is_trend ? cv::Scalar(0, 0, 255) : cv::Scalar(200, 200, 200);

        cv::rectangle(canvas,
                      cv::Point(bar_x, bar_y),
                      cv::Point(bar_x + max((hist_w - 20) / 360, 1), hist_y + hist_h - 10),
                      color, -1);
    }

// 텍스트 출력 (겹침 방지)
    if (offset_trend > -180) {
        std::string trend_text = "Trend: " + fixed(offset_trend, 1) + " deg";
        cv::putText(canvas, trend_text,
                    cv::Point(hist_x + 10, hist_y + hist_h + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);

        double current_gyro = compass_analyzer.gyro_history.back();
        double absolute_direction = normalizeAngle(current_gyro + offset_trend);
        std::string abs_text = "Absolute Dir: " + fixed(absolute_direction, 1) + " deg";
        cv::putText(canvas, abs_text,
                    cv::Point(hist_x + 200, hist_y + hist_h + 20),  // 오른쪽으로 이동
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 100, 0), 1);
    }

    // ========== 좌측 중단: 원본 나침반 vs 자이로 비교 그래프 ==========
    int graph_x = 50, graph_y = 580, graph_w = 400, graph_h = 120;  // y: 540 → 580, h: 150 → 120

    cv::rectangle(canvas, cv::Point(graph_x, graph_y),
                  cv::Point(graph_x + graph_w, graph_y + graph_h),
                  cv::Scalar(200, 200, 200), 1);

    cv::putText(canvas, "Compass vs Gyro (last 50 steps)", cv::Point(graph_x + 10, graph_y + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    // 중간선
    cv::line(canvas, cv::Point(graph_x, graph_y + graph_h / 2),
             cv::Point(graph_x + graph_w, graph_y + graph_h / 2),
             cv::Scalar(150, 150, 150), 1);

    // 최근 50개 데이터 플롯
    int start_idx = max(0, (int)compass_analyzer.compass_history.size() - 50);
    int data_count = compass_analyzer.compass_history.size() - start_idx;

    if (data_count > 1) {
        double x_step = (double)graph_w / max(data_count - 1, 1);

        // 나침반 값 (파란색)
        for (int i = 1; i < data_count; i++) {
            float angle1 = compass_analyzer.compass_history[start_idx + i - 1].second;
            float angle2 = compass_analyzer.compass_history[start_idx + i].second;

            int x1 = graph_x + (i - 1) * x_step;
            int x2 = graph_x + i * x_step;
            int y1 = graph_y + graph_h / 2 - (angle1 - 180) * graph_h / 360;
            int y2 = graph_y + graph_h / 2 - (angle2 - 180) * graph_h / 360;

            cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2),
                     cv::Scalar(255, 100, 0), 2, cv::LINE_AA);
        }

        // 자이로 값 (빨간색)
        for (int i = 1; i < data_count; i++) {
            float angle1 = compass_analyzer.gyro_history[start_idx + i - 1];
            float angle2 = compass_analyzer.gyro_history[start_idx + i];

            int x1 = graph_x + (i - 1) * x_step;
            int x2 = graph_x + i * x_step;
            int y1 = graph_y + graph_h / 2 - (angle1 - 180) * graph_h / 360;
            int y2 = graph_y + graph_h / 2 - (angle2 - 180) * graph_h / 360;

            cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2),
                     cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }

    // 범례
    cv::line(canvas, cv::Point(graph_x + 10, graph_y + graph_h + 10),
             cv::Point(graph_x + 30, graph_y + graph_h + 10),
             cv::Scalar(255, 100, 0), 2);
    cv::putText(canvas, "Compass", cv::Point(graph_x + 35, graph_y + graph_h + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);

    cv::line(canvas, cv::Point(graph_x + 100, graph_y + graph_h + 10),
             cv::Point(graph_x + 120, graph_y + graph_h + 10),
             cv::Scalar(0, 0, 255), 2);
    cv::putText(canvas, "Gyro", cv::Point(graph_x + 125, graph_y + graph_h + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);

    // ========== 우상단: 방향 가설 원형 시각화 (기존 유지) ==========
    int circle_center_x = 700, circle_center_y = 200;
    int circle_radius = 150;

    cv::circle(canvas, cv::Point(circle_center_x, circle_center_y),
               circle_radius, cv::Scalar(200, 200, 200), 2);

    cv::putText(canvas, "Current Orientation", cv::Point(550, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    // 방위각 표시
    cv::putText(canvas, "N", cv::Point(circle_center_x - 5, circle_center_y - circle_radius - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    cv::putText(canvas, "E", cv::Point(circle_center_x + circle_radius + 10, circle_center_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    cv::putText(canvas, "S", cv::Point(circle_center_x - 5, circle_center_y + circle_radius + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    cv::putText(canvas, "W", cv::Point(circle_center_x - circle_radius - 20, circle_center_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    // 가설들 표시
    for (const auto& hypo : hypotheses) {
        double rad = (hypo.angle - 90) * M_PI / 180.0;
        int line_length = circle_radius * hypo.confidence;
        int end_x = circle_center_x + cos(rad) * line_length;
        int end_y = circle_center_y + sin(rad) * line_length;

        cv::Scalar color = hypo.confidence > 0.5 ?
                           cv::Scalar(0, 200, 0) : cv::Scalar(150, 150, 150);
        int thickness = hypo.confidence > 0.5 ? 3 : 1;

        cv::line(canvas, cv::Point(circle_center_x, circle_center_y),
                 cv::Point(end_x, end_y), color, thickness, cv::LINE_AA);
    }

    // 현재 방향 (빨간색)
    double current_orientation = drift_tracker.getCurrentOffset();
    double current_rad = (current_orientation - 90) * M_PI / 180.0;
    int arrow_x = circle_center_x + cos(current_rad) * circle_radius;
    int arrow_y = circle_center_y + sin(current_rad) * circle_radius;

    cv::arrowedLine(canvas, cv::Point(circle_center_x, circle_center_y),
                    cv::Point(arrow_x, arrow_y),
                    cv::Scalar(0, 0, 255), 3, cv::LINE_AA, 0, 0.3);

    cv::putText(canvas, fixed(current_orientation, 1) + "deg",
                cv::Point(circle_center_x - 30, circle_center_y + circle_radius + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

    // 절대 방향 추정값도 표시 (녹색)
    if (offset_trend > -180) {
        double trend_rad = (offset_trend - 90) * M_PI / 180.0;
        int trend_x = circle_center_x + cos(trend_rad) * (circle_radius - 20);
        int trend_y = circle_center_y + sin(trend_rad) * (circle_radius - 20);

        cv::circle(canvas, cv::Point(trend_x, trend_y), 8,
                   cv::Scalar(0, 255, 0), -1);
    }

    // ========== 중단: 건강 지표 (기존 유지) ==========
    int health_y = 380;

    cv::putText(canvas, "System Health", cv::Point(50, health_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

    auto drawHealthBar = [&](const string& label, double value, int y_pos, cv::Scalar color) {
        cv::putText(canvas, label, cv::Point(50, y_pos),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

        int bar_w = 300;
        int bar_h = 20;
        int bar_x = 250;

        cv::rectangle(canvas, cv::Point(bar_x, y_pos - 15),
                      cv::Point(bar_x + bar_w, y_pos - 15 + bar_h),
                      cv::Scalar(200, 200, 200), -1);

        int fill_w = bar_w * value;
        cv::rectangle(canvas, cv::Point(bar_x, y_pos - 15),
                      cv::Point(bar_x + fill_w, y_pos - 15 + bar_h),
                      color, -1);

        cv::putText(canvas, fixed(value, 2), cv::Point(bar_x + bar_w + 10, y_pos),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    };

    drawHealthBar("Compass Reliability:", health.compass_reliability, health_y + 40,
                  cv::Scalar(100, 150, 255));
    drawHealthBar("Loc Success Rate:", health.localization_success_rate, health_y + 70,
                  cv::Scalar(0, 200, 0));
    drawHealthBar("Drift Health:", 1.0 / (1.0 + health.drift_magnitude / 10.0), health_y + 100,
                  cv::Scalar(255, 150, 0));
    drawHealthBar("Overall Health:", health.overall_health, health_y + 130,
                  health.overall_health > 0.7 ? cv::Scalar(0, 200, 0) :
                  health.overall_health > 0.5 ? cv::Scalar(255, 200, 0) : cv::Scalar(0, 0, 255));

    // ========== 하단: 통계 정보 업데이트 ==========
    int stats_x = 650, stats_y = 580;

    cv::putText(canvas, "Statistics", cv::Point(stats_x, stats_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

    double compass_stability = compass_analyzer.getTemporalStability();
    double consistency = compass_analyzer.evaluateGyroCompassConsistency();

    cv::putText(canvas, "Total Samples: " + std::to_string(compass_analyzer.compass_history.size()),
                cv::Point(stats_x, stats_y + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

// 오프셋 표준편차 계산
    double mean_offset = 0;
    for (float offset : compass_analyzer.direction_offsets) {
        mean_offset += offset;
    }
    mean_offset /= compass_analyzer.direction_offsets.size();

    double std_dev = 0;
    for (float offset : compass_analyzer.direction_offsets) {
        double diff = offset - mean_offset;
        std_dev += diff * diff;
    }
    std_dev = sqrt(std_dev / compass_analyzer.direction_offsets.size());

    cv::putText(canvas, "Offset Std Dev: " + fixed(std_dev, 2) + " deg",
                cv::Point(stats_x, stats_y + 55),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    cv::putText(canvas, "Temporal Stability: " + fixed(compass_stability, 2),
                cv::Point(stats_x, stats_y + 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    cv::putText(canvas, "Consistency Score: " + fixed(consistency, 2),
                cv::Point(stats_x, stats_y + 105),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    cv::putText(canvas, "Drift: " + fixed(drift_tracker.accumulated_drift, 2) + " deg",
                cv::Point(stats_x, stats_y + 130),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    cv::imshow("Orientation Estimation (A/D: navigate, Q: quit)", canvas);
}


void visualizePDRPath(const vector<Point>& pdr_history,
                      const vector<Point>& best_pdr_history,
                      int stepNumber,
                      double base_orientation,
                      double best_orientation) {
    if (pdr_history.empty() && best_pdr_history.empty()) return;

    // 90도 반시계방향 회전 변환: (x, y) -> (-y, x)
    auto rotate90 = [](const Point& p) -> Point {
        return {p.second, -p.first};
    };

    // 회전된 좌표로 경계 계산
    int min_x = INT_MAX, max_x = INT_MIN;
    int min_y = INT_MAX, max_y = INT_MIN;

    for (const auto& p : pdr_history) {
        Point rotated = rotate90(p);
        min_x = min(min_x, rotated.first);
        max_x = max(max_x, rotated.first);
        min_y = min(min_y, rotated.second);
        max_y = max(max_y, rotated.second);
    }

    for (const auto& p : best_pdr_history) {
        Point rotated = rotate90(p);
        min_x = min(min_x, rotated.first);
        max_x = max(max_x, rotated.first);
        min_y = min(min_y, rotated.second);
        max_y = max(max_y, rotated.second);
    }

    int padding = 50;
    int width = max(max_x - min_x + 2 * padding, 400);
    int height = max(max_y - min_y + 2 * padding, 400);

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 제목
    std::string title = "PDR Path - Step " + std::to_string(stepNumber);
    cv::putText(canvas, title, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

    // 기존 PDR 경로 (회색, 얇게)
    for (size_t i = 1; i < pdr_history.size(); i++) {
        Point p1 = rotate90(pdr_history[i-1]);
        Point p2 = rotate90(pdr_history[i]);

        int x1 = p1.first - min_x + padding;
        int y1 = p1.second - min_y + padding;
        int x2 = p2.first - min_x + padding;
        int y2 = p2.second - min_y + padding;

        cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2),
                 cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }

    // Best PDR 경로 (파란색, 굵게)
    for (size_t i = 1; i < best_pdr_history.size(); i++) {
        Point p1 = rotate90(best_pdr_history[i-1]);
        Point p2 = rotate90(best_pdr_history[i]);

        int x1 = p1.first - min_x + padding;
        int y1 = p1.second - min_y + padding;
        int x2 = p2.first - min_x + padding;
        int y2 = p2.second - min_y + padding;

        cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2),
                 cv::Scalar(255, 100, 0), 2, cv::LINE_AA);
    }

    // 시작점 (녹색)
    if (!best_pdr_history.empty()) {
        Point start = rotate90(best_pdr_history.front());
        int start_x = start.first - min_x + padding;
        int start_y = start.second - min_y + padding;
        cv::circle(canvas, cv::Point(start_x, start_y), 8,
                   cv::Scalar(0, 255, 0), -1);
    }

    // 끝점 (빨간색)
    if (!best_pdr_history.empty()) {
        Point end = rotate90(best_pdr_history.back());
        int end_x = end.first - min_x + padding;
        int end_y = end.second - min_y + padding;
        cv::circle(canvas, cv::Point(end_x, end_y), 8,
                   cv::Scalar(0, 0, 255), -1);
    }

    // 범례
    cv::line(canvas, cv::Point(10, height - 50), cv::Point(30, height - 50),
             cv::Scalar(200, 200, 200), 2);
    cv::putText(canvas, "Base (" + fixed(base_orientation, 1) + " deg)",
                cv::Point(35, height - 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

    cv::line(canvas, cv::Point(10, height - 30), cv::Point(30, height - 30),
             cv::Scalar(255, 100, 0), 2);
    cv::putText(canvas, "Best (" + fixed(best_orientation, 1) + " deg, delta=" +
                        fixed(best_orientation - base_orientation, 1) + ")",
                cv::Point(35, height - 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 100, 0), 1);

    cv::imshow("PDR Path", canvas);
}


void visualizeResults(const BitMatrix& map, const std::vector<Point>& candidates,
                      const Point& final_pos, const Point& origin) {
    // 지도 시각화
    cv::Mat mapImg(map.height, map.width, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < map.height; i++) {
        for (int j = 0; j < map.width; j++) {
            if (map.getBit(i, j)) {
                mapImg.at<cv::Vec3b>(i, j) = cv::Vec3b(200, 200, 200);
            }
        }
    }

    // 후보군 위치 표시 (파란색)
    for (const auto& c : candidates) {
        cv::circle(mapImg, cv::Point(c.second, c.first), 3, cv::Scalar(255, 0, 0), -1);
    }

    // 최종 위치 표시 (빨간색)
    if (final_pos.first >= 0 && final_pos.second >= 0) {
        int adj_x = final_pos.first - origin.first;
        int adj_y = final_pos.second - origin.second;
        cv::circle(mapImg, cv::Point(adj_y, adj_x), 5, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("Map Matching Result", mapImg);
}

std::vector<StepResult> g_allResults;  // 전역 결과 저장
int g_currentViewStep = 0;             // 현재 보고 있는 스텝

void visualizeMapMatching(const BitMatrix& map, const vector<Point>& candidates,
                          const Point& final_pos, const Point& mapmatched_pos,
                          bool has_mapmatching, const Point& origin,
                          int stepNumber, double distance) {
    // 지도 크기에 맞춰 이미지 생성
    int display_width = std::min(map.width, 1200);
    int display_height = std::min(map.height, 1200);

    cv::Mat mapImg(map.height, map.width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 지도 그리기 (회색) - Y 좌표 반전
    for (int i = 0; i < map.height; i++) {
        for (int j = 0; j < map.width; j++) {
            if (map.getBit(i, j)) {
                int flipped_i = map.height - 1 - i;
                mapImg.at<cv::Vec3b>(flipped_i, j) = cv::Vec3b(200, 200, 200);
            }
        }
    }

    // 후보군 위치 표시 (파란색 점들) - Y 좌표 반전
    for (const auto& c : candidates) {
        if (c.first >= 0 && c.first < map.height &&
            c.second >= 0 && c.second < map.width) {
            int flipped_row = map.height - 1 - c.first;
            cv::circle(mapImg, cv::Point(c.second, flipped_row), 3,
                       cv::Scalar(255, 0, 0), -1);
        }
    }

    // 최종 위치 표시 (빨간색 원) - Y 좌표 반전
    if (final_pos.first >= 0 && final_pos.second >= 0) {
        int adj_x = final_pos.first - origin.first;
        int adj_y = final_pos.second - origin.second;

        if (adj_x >= 0 && adj_x < map.height &&
            adj_y >= 0 && adj_y < map.width) {
            int flipped_row = map.height - 1 - adj_x;
            cv::circle(mapImg, cv::Point(adj_y, flipped_row), 8,
                       cv::Scalar(0, 0, 255), -1);
            cv::circle(mapImg, cv::Point(adj_y, flipped_row), 12,
                       cv::Scalar(0, 0, 255), 2);
        }
    }

    // 맵매칭 결과 표시 (녹색 마름모) - Y 좌표 반전 (추가)
    if (has_mapmatching && mapmatched_pos.first >= 0 && mapmatched_pos.second >= 0) {
        int mm_x = mapmatched_pos.first - origin.first;
        int mm_y = mapmatched_pos.second - origin.second;

        if (mm_x >= 0 && mm_x < map.height &&
            mm_y >= 0 && mm_y < map.width) {
            int flipped_row = map.height - 1 - mm_x;
            cv::Point center(mm_y, flipped_row);

            // 마름모 그리기
            int size = 10;
            std::vector<cv::Point> diamond = {
                    cv::Point(center.x, center.y - size),      // 위
                    cv::Point(center.x + size, center.y),      // 오른쪽
                    cv::Point(center.x, center.y + size),      // 아래
                    cv::Point(center.x - size, center.y)       // 왼쪽
            };

            // 채워진 마름모
            cv::fillConvexPoly(mapImg, diamond.data(), 4, cv::Scalar(0, 200, 0));
            // 테두리
            cv::polylines(mapImg, diamond, true, cv::Scalar(0, 150, 0), 2);

            // 연결선 그리기 (원래 위치 -> 맵매칭 위치)
            if (final_pos.first >= 0) {
                int orig_flipped_row = map.height - 1 - (final_pos.first - origin.first);
                cv::line(mapImg,
                         cv::Point(final_pos.second - origin.second, orig_flipped_row),
                         center,
                         cv::Scalar(0, 150, 0), 1, cv::LINE_AA);
            }
        }
    }

    // 화면 크기에 맞춰 리사이즈
    cv::Mat resized;
    if (map.width > display_width || map.height > display_height) {
        double scale = std::min((double)display_width / map.width,
                                (double)display_height / map.height);
        cv::resize(mapImg, resized, cv::Size(), scale, scale);
    } else {
        resized = mapImg;
    }

    // 정보 표시
    std::string info1 = "Step: " + std::to_string(stepNumber);
    std::string info2 = "Distance: " + std::to_string(distance) + "m";
    std::string info3 = "Candidates: " + std::to_string(candidates.size());
    std::string info4 = "Position: (" + std::to_string(final_pos.first) +
                        ", " + std::to_string(final_pos.second) + ")";

    cv::putText(resized, info1, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(resized, info2, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(resized, info3, cv::Point(10, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(resized, info4, cv::Point(10, 120),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // 맵매칭 정보 추가
    if (has_mapmatching) {
        std::string info5 = "Map-matched: (" + std::to_string(mapmatched_pos.first) +
                            ", " + std::to_string(mapmatched_pos.second) + ")";
        cv::putText(resized, info5, cv::Point(10, 150),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 150, 0), 2);
    }

    // 범례 추가
    int legend_y = resized.rows - 80;
    cv::circle(resized, cv::Point(20, legend_y), 8, cv::Scalar(0, 0, 255), -1);
    cv::putText(resized, "Final Position", cv::Point(35, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    if (has_mapmatching) {
        legend_y += 25;
        std::vector<cv::Point> legend_diamond = {
                cv::Point(20, legend_y - 8),
                cv::Point(28, legend_y),
                cv::Point(20, legend_y + 8),
                cv::Point(12, legend_y)
        };
        cv::fillConvexPoly(resized, legend_diamond.data(), 4, cv::Scalar(0, 200, 0));
        cv::putText(resized, "Map-matched", cv::Point(35, legend_y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::imshow("Map Matching Result (A/D: navigate, Q: quit)", resized);
}

void visualizeCalibration(const StepResult& step, const vector<int>& gyroCaliCandidate,
                          const std::vector<StepResult>& historyUpToNow) {
    if (step.calibrationDone && step.candidateDiffs.empty()) {
        // 캘리브레이션 완료 후에는 간단한 메시지만
        cv::Mat canvas(300, 600, CV_8UC3, cv::Scalar(255, 255, 255));
        std::string msg = "Calibration Complete!";
        std::string val = "Final Value: " + std::to_string(step.finalCalibValue);
        cv::putText(canvas, msg, cv::Point(150, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 200, 0), 2);
        cv::putText(canvas, val, cv::Point(150, 180),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 200), 2);
        cv::imshow("Gyro Calibration (A/D: navigate, Q: quit)", canvas);
        return;
    }

    if (step.candidateDiffs.empty()) return;

    int width = 900;
    int height = 750;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 제목
    std::string title = "Gyro Calibration - Step " + std::to_string(step.stepNumber);
    cv::putText(canvas, title, cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

    // ========== 상단: 막대 그래프 영역 ==========
    int graph1_x = 50;
    int graph1_y = 60;
    int graph1_w = 400;
    int graph1_h = 200;

    // 축 그리기
    cv::line(canvas, cv::Point(graph1_x, graph1_y + graph1_h),
             cv::Point(graph1_x + graph1_w, graph1_y + graph1_h),
             cv::Scalar(0, 0, 0), 2);
    cv::line(canvas, cv::Point(graph1_x, graph1_y),
             cv::Point(graph1_x, graph1_y + graph1_h),
             cv::Scalar(0, 0, 0), 2);

    // Y축 라벨
    cv::putText(canvas, "Diff", cv::Point(10, graph1_y + graph1_h / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    // 최대 차이값 찾기
    double max_diff = *std::max_element(step.candidateDiffs.begin(),
                                        step.candidateDiffs.end());
    max_diff = std::max(max_diff, 50.0);

    // 막대 그래프 그리기
    int num_candidates = step.candidateDiffs.size();
    int bar_width = graph1_w / (num_candidates * 2);

    for (int i = 0; i < num_candidates; i++) {
        int bar_h = (step.candidateDiffs[i] / max_diff) * graph1_h;
        int bar_x = graph1_x + (graph1_w / num_candidates) * i + bar_width / 2;
        int bar_y = graph1_y + graph1_h - bar_h;

        cv::Scalar color = (i == step.bestCandidateIdx) ?
                           cv::Scalar(0, 200, 0) : cv::Scalar(200, 100, 0);

        cv::rectangle(canvas,
                      cv::Point(bar_x, bar_y),
                      cv::Point(bar_x + bar_width, graph1_y + graph1_h),
                      color, -1);

        std::string label = std::to_string(gyroCaliCandidate[i]) + " deg";
        cv::putText(canvas, label, cv::Point(bar_x - 5, graph1_y + graph1_h + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

        std::string diff_text = fixed(step.candidateDiffs[i], 1);
        cv::putText(canvas, diff_text, cv::Point(bar_x - 5, bar_y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }

// ========== 하단: 시계열 선 그래프 영역 ==========
    int graph2_x = 50;
    int graph2_y = 320;
    int graph2_w = 800;
    int graph2_h = 250;

    // 축 그리기
    cv::line(canvas, cv::Point(graph2_x, graph2_y + graph2_h),
             cv::Point(graph2_x + graph2_w, graph2_y + graph2_h),
             cv::Scalar(0, 0, 0), 2);
    cv::line(canvas, cv::Point(graph2_x, graph2_y),
             cv::Point(graph2_x, graph2_y + graph2_h),
             cv::Scalar(0, 0, 0), 2);

    // Y축 라벨
    cv::putText(canvas, "Angle", cv::Point(5, graph2_y + graph2_h / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(canvas, "(deg)", cv::Point(5, graph2_y + graph2_h / 2 + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    // X축 라벨
    cv::putText(canvas, "Step", cv::Point(graph2_x + graph2_w / 2, graph2_y + graph2_h + 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    // 각도 정규화 함수
    auto normalizeAngle = [](double angle) -> double {
        angle = fmod(angle, 360.0);
        if (angle < 0) angle += 360.0;
        return angle;
    };

    // 각도를 Y 좌표로 변환
    auto angleToY = [&](double angle) -> int {
        return graph2_y + graph2_h - (angle / 360.0) * graph2_h;
    };

    // 각도 언래핑 함수 (0-360 경계 점프 방지)
    auto unwrapAngle = [](double current, double previous) -> double {
        double diff = current - previous;
        if (diff > 180) {
            return current - 360;
        } else if (diff < -180) {
            return current + 360;
        }
        return current;
    };

    // X 위치 계산 (히스토리 데이터 개수에 따라)
    int num_points = historyUpToNow.size();
    if (num_points < 2) return; // 최소 2개 필요

    double x_step = (double)graph2_w / (num_points - 1);

    // 언래핑된 각도 저장
    std::vector<double> unwrapped_compass;
    std::vector<double> unwrapped_gyro;
    std::vector<double> unwrapped_gyro_90;
    std::vector<double> unwrapped_gyro_180;
    std::vector<double> unwrapped_gyro_270;

    // 첫 번째 값은 정규화만
    unwrapped_compass.push_back(normalizeAngle(historyUpToNow[0].compass));
    unwrapped_gyro.push_back(normalizeAngle(historyUpToNow[0].gyro));
    unwrapped_gyro_90.push_back(normalizeAngle(historyUpToNow[0].gyro + 90));
    unwrapped_gyro_180.push_back(normalizeAngle(historyUpToNow[0].gyro + 180));
    unwrapped_gyro_270.push_back(normalizeAngle(historyUpToNow[0].gyro + 270));

    // 나머지는 언래핑
    for (int i = 1; i < num_points; i++) {
        const auto& hist_step = historyUpToNow[i];

        double compass_angle = normalizeAngle(hist_step.compass);
        double gyro_angle = normalizeAngle(hist_step.gyro);
        double gyro_90 = normalizeAngle(hist_step.gyro + 90);
        double gyro_180 = normalizeAngle(hist_step.gyro + 180);
        double gyro_270 = normalizeAngle(hist_step.gyro + 270);

        unwrapped_compass.push_back(unwrapAngle(compass_angle, unwrapped_compass.back()));
        unwrapped_gyro.push_back(unwrapAngle(gyro_angle, unwrapped_gyro.back()));
        unwrapped_gyro_90.push_back(unwrapAngle(gyro_90, unwrapped_gyro_90.back()));
        unwrapped_gyro_180.push_back(unwrapAngle(gyro_180, unwrapped_gyro_180.back()));
        unwrapped_gyro_270.push_back(unwrapAngle(gyro_270, unwrapped_gyro_270.back()));
    }

    // 전체 범위 찾기
    double min_angle = 0, max_angle = 360;
    std::vector<double> all_angles;
    all_angles.insert(all_angles.end(), unwrapped_compass.begin(), unwrapped_compass.end());
    all_angles.insert(all_angles.end(), unwrapped_gyro.begin(), unwrapped_gyro.end());
    all_angles.insert(all_angles.end(), unwrapped_gyro_90.begin(), unwrapped_gyro_90.end());
    all_angles.insert(all_angles.end(), unwrapped_gyro_180.begin(), unwrapped_gyro_180.end());
    all_angles.insert(all_angles.end(), unwrapped_gyro_270.begin(), unwrapped_gyro_270.end());

    if (!all_angles.empty()) {
        min_angle = *std::min_element(all_angles.begin(), all_angles.end());
        max_angle = *std::max_element(all_angles.begin(), all_angles.end());

        // 여유 공간 추가
        double range = max_angle - min_angle;
        min_angle -= range * 0.1;
        max_angle += range * 0.1;

        // 최소 범위 보장
        if (max_angle - min_angle < 90) {
            double center = (min_angle + max_angle) / 2;
            min_angle = center - 45;
            max_angle = center + 45;
        }
    }

    // 언래핑된 각도를 Y 좌표로 변환
    auto unwrappedAngleToY = [&](double angle) -> int {
        double normalized = (angle - min_angle) / (max_angle - min_angle);
        return graph2_y + graph2_h - (normalized * graph2_h);
    };

    // 각 라인을 그리기 위한 포인트 벡터
    std::vector<cv::Point> compass_points;
    std::vector<cv::Point> gyro_points;
    std::vector<cv::Point> gyro_90_points;
    std::vector<cv::Point> gyro_180_points;
    std::vector<cv::Point> gyro_270_points;

    for (int i = 0; i < num_points; i++) {
        int x = graph2_x + (int)(i * x_step);

        compass_points.push_back(cv::Point(x, unwrappedAngleToY(unwrapped_compass[i])));
        gyro_points.push_back(cv::Point(x, unwrappedAngleToY(unwrapped_gyro[i])));
        gyro_90_points.push_back(cv::Point(x, unwrappedAngleToY(unwrapped_gyro_90[i])));
        gyro_180_points.push_back(cv::Point(x, unwrappedAngleToY(unwrapped_gyro_180[i])));
        gyro_270_points.push_back(cv::Point(x, unwrappedAngleToY(unwrapped_gyro_270[i])));
    }

    // Y축 눈금 (적응형)
    int num_ticks = 5;
    for (int i = 0; i <= num_ticks; i++) {
        double angle = min_angle + (max_angle - min_angle) * i / num_ticks;
        int y_line = unwrappedAngleToY(angle);

        cv::line(canvas, cv::Point(graph2_x, y_line),
                 cv::Point(graph2_x + graph2_w, y_line),
                 cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

        // 360으로 나눈 나머지 표시
        double display_angle = fmod(angle, 360.0);
        if (display_angle < 0) display_angle += 360.0;

        cv::putText(canvas, fixed(display_angle, 0),
                    cv::Point(graph2_x - 35, y_line + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(150, 150, 150), 1);
    }

    // 선 그래프 그리기
    // Compass (검은색, 굵게)
    for (size_t i = 1; i < compass_points.size(); i++) {
        cv::line(canvas, compass_points[i-1], compass_points[i],
                 cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    }

    // Gyro (빨간색)
    for (size_t i = 1; i < gyro_points.size(); i++) {
        cv::line(canvas, gyro_points[i-1], gyro_points[i],
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // Gyro + 90 (주황색)
    for (size_t i = 1; i < gyro_90_points.size(); i++) {
        cv::line(canvas, gyro_90_points[i-1], gyro_90_points[i],
                 cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
    }

    // Gyro + 180 (파란색)
    for (size_t i = 1; i < gyro_180_points.size(); i++) {
        cv::line(canvas, gyro_180_points[i-1], gyro_180_points[i],
                 cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }

    // Gyro + 270 (보라색)
    for (size_t i = 1; i < gyro_270_points.size(); i++) {
        cv::line(canvas, gyro_270_points[i-1], gyro_270_points[i],
                 cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
    }

    // 현재 스텝 마커 (마지막 포인트)
    cv::circle(canvas, compass_points.back(), 5, cv::Scalar(0, 0, 0), -1);
    cv::circle(canvas, gyro_points.back(), 4, cv::Scalar(0, 0, 255), -1);
    cv::circle(canvas, gyro_90_points.back(), 4, cv::Scalar(0, 165, 255), -1);
    cv::circle(canvas, gyro_180_points.back(), 4, cv::Scalar(255, 0, 0), -1);
    cv::circle(canvas, gyro_270_points.back(), 4, cv::Scalar(255, 0, 255), -1);

    // 범례
    int legend_x = graph2_x + graph2_w - 180;
    int legend_y = graph2_y + 20;
    int legend_spacing = 25;

    cv::line(canvas, cv::Point(legend_x, legend_y),
             cv::Point(legend_x + 30, legend_y), cv::Scalar(0, 0, 0), 3);
    cv::putText(canvas, "Compass", cv::Point(legend_x + 35, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    legend_y += legend_spacing;
    cv::line(canvas, cv::Point(legend_x, legend_y),
             cv::Point(legend_x + 30, legend_y), cv::Scalar(0, 0, 255), 2);
    cv::putText(canvas, "Gyro", cv::Point(legend_x + 35, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

    legend_y += legend_spacing;
    cv::line(canvas, cv::Point(legend_x, legend_y),
             cv::Point(legend_x + 30, legend_y), cv::Scalar(0, 165, 255), 2);
    cv::putText(canvas, "Gyro+90", cv::Point(legend_x + 35, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 165, 255), 1);

    legend_y += legend_spacing;
    cv::line(canvas, cv::Point(legend_x, legend_y),
             cv::Point(legend_x + 30, legend_y), cv::Scalar(255, 0, 0), 2);
    cv::putText(canvas, "Gyro+180", cv::Point(legend_x + 35, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);

    legend_y += legend_spacing;
    cv::line(canvas, cv::Point(legend_x, legend_y),
             cv::Point(legend_x + 30, legend_y), cv::Scalar(255, 0, 255), 2);
    cv::putText(canvas, "Gyro+270", cv::Point(legend_x + 35, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 255), 1);

    // ========== 정보 표시 ==========
    int info_y = 600;
    std::string best_text = "Best Candidate: " +
                            std::to_string(gyroCaliCandidate[step.bestCandidateIdx]) +
                            " (diff: " + fixed(step.candidateDiffs[step.bestCandidateIdx], 2) + ")";
    cv::putText(canvas, best_text, cv::Point(20, info_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 100, 0), 2);

    // 히스토리 카운트
    std::map<int, int> hist_count;
    for (int val : step.calibHistory) {
        hist_count[val]++;
    }

    info_y += 30;
    std::string hist_text = "History (" + std::to_string(step.calibHistory.size()) + "/10): ";
    for (const auto& entry : hist_count) {
        hist_text += std::to_string(entry.first) + " deg(" + std::to_string(entry.second) + ") ";
    }
    cv::putText(canvas, hist_text, cv::Point(20, info_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 200), 1);

    // 진행 상황
    info_y += 30;
    int max_count = 0;
    for (const auto& entry : hist_count) {
        max_count = std::max(max_count, entry.second);
    }
    std::string progress = "Progress: " + std::to_string(max_count) + "/9 needed";
    cv::Scalar prog_color = (max_count >= 9) ? cv::Scalar(0, 200, 0) : cv::Scalar(200, 0, 0);
    cv::putText(canvas, progress, cv::Point(20, info_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, prog_color, 2);

    cv::imshow("Gyro Calibration (A/D: navigate, Q: quit)", canvas);
}

// showStep 함수 업데이트
// 전역 변수 추가
BitMatrix g_currentBinaryMap;
int g_currentDisplayFloor = -999;

void showStep(int stepIndex, const std::string& testbedPath) {
    if (stepIndex < 0 || stepIndex >= g_allResults.size()) return;

    const auto& step = g_allResults[stepIndex];

    // 층이 바뀌었으면 맵 다시 로드
    if (step.floor != g_currentDisplayFloor) {
        g_currentDisplayFloor = step.floor;

        std::string floorText = (g_currentDisplayFloor < 0 ?
                                 "B" + std::to_string(abs(g_currentDisplayFloor)) :
                                 std::to_string(g_currentDisplayFloor)) + "F";
        std::string binaryMapFile = testbedPath + "/zeromap_" + floorText + " - origin(" +
                                    std::to_string(originMap[g_currentDisplayFloor].first) + "," +
                                    std::to_string(originMap[g_currentDisplayFloor].second) + ").txt";

        g_currentBinaryMap = readBitMatrixFromTextFile(binaryMapFile.c_str());

        print(color::yellow, "Visualization: Loading map for floor ", floorText, color::reset);
    }

    Point currentOrigin = {originMap[g_currentDisplayFloor].first,
                           originMap[g_currentDisplayFloor].second};

    // 임시 CompassAnalyzer와 DriftTracker 생성
    CompassAnalyzer temp_compass;
    temp_compass.compass_history = step.compass_history_snapshot;
    temp_compass.gyro_history = step.gyro_history_snapshot;
    temp_compass.direction_offsets = step.direction_offsets_snapshot;

    GyroDriftTracker temp_drift;
    temp_drift.baseline_offset = step.baseline_offset;
    temp_drift.accumulated_drift = step.accumulated_drift;
    temp_drift.drift_rate = 0;
    temp_drift.last_correction_step = 0;

    // 방향 추정 시각화
    visualizeOrientationEstimation(step, temp_compass, temp_drift,
                                   step.hypotheses, step.health);

    // PDR 경로 시각화
    visualizePDRPath(step.pdr_history,
                     step.best_pdr_history,
                     step.stepNumber,
                     step.base_orientation,
                     step.best_orientation);

    // 지도 매칭 결과 시각화 (현재 층의 맵과 origin 사용)
    Point finalPos(step.result.original_x, step.result.original_y);
    visualizeMapMatching(g_currentBinaryMap, step.candidates, finalPos,
                         step.mapmatched_pos, step.has_mapmatching,
                         currentOrigin, step.stepNumber, step.result.distance);

    // 콘솔 출력
    print("");
    print(color::cyan, "Step ", step.stepNumber, " / ", g_allResults.size(),
          " (Floor: ", step.floor, "F)", color::reset);
    print("  Position: (", step.result.x, ", ", step.result.y, ")");
    print("  Distance: ", step.result.distance);
    print("  Orientation: ", step.best_orientation, " (base: ", step.base_orientation, ")");
}



int main(int argc, char** argv) {
    // 설정
    std::string testbedPath = "/Users/idohun/WorkSpace/PDR/zeromap-simulation/map_data/enb";  // 기본값 설정
    std::string dataPath = "/Users/idohun/WorkSpace/PDR/zeromap-simulation/test_dataset/pdr_log_20260116_094818.txt";

    // 커맨드라인 인자가 있으면 우선 사용
    if (argc >= 2) {
        testbedPath = argv[1];
    }
    if (argc >= 3) {
        dataPath = argv[2];
    }

    std::cout << "=== Simulation Configuration ===" << std::endl;
    std::cout << "Testbed Path: " << testbedPath << std::endl;
    std::cout << "Data Path: " << dataPath << std::endl;
    std::cout << "===============================" << std::endl;

    // Origin map 로드
    loadOriginOfMap(testbedPath);

    std::cout << "Loaded " << originMap.size() << " floor origins" << std::endl;

    // 층 정보 로드
    std::string stairsPath = testbedPath + "/floorchange_stairs_areas.json";
    std::string elevatorPath = testbedPath + "/floorchange_elevator_areas.json";
    stairsInfo = loadJsonData(stairsPath.c_str());
    elevatorInfo = loadJsonData(elevatorPath.c_str());

    // 시뮬레이션 데이터 로드
    auto simData = loadSimulationData(dataPath);

    if (simData.empty()) {
        std::cerr << "No simulation data loaded from: " << dataPath << std::endl;
        return -1;
    }

    std::cout << "Loaded " << simData.size() << " simulation steps" << std::endl;

    int start_floor = simData[0].floor;
    g_currentDisplayFloor = start_floor;  // 전역 변수 사용

    std::cout << "Starting floor: " << start_floor << std::endl;

    // 엔진 초기화
    string floorText = (start_floor < 0 ? "B" + to_string(abs(start_floor)) : to_string(start_floor)) + "F";
    string binaryMapFile = testbedPath + "/zeromap_" + floorText + " - origin(" +
                           to_string(originMap[start_floor].first) + "," +
                           to_string(originMap[start_floor].second) + ").txt";
    string indoorMapFile = testbedPath + "/indoor_map_" + floorText + ".json";

    std::cout << "Loading binary map: " << binaryMapFile << std::endl;
    std::cout << "Loading indoor map: " << indoorMapFile << std::endl;

    BitMatrix binaryMap = readBitMatrixFromTextFile(binaryMapFile.c_str());
    MapData mapData = loadMapData(indoorMapFile.c_str());

    g_currentBinaryMap = binaryMap;  // 전역 변수에 저장

    std::cout << "Map size: " << binaryMap.height << " x " << binaryMap.width << std::endl;

    PositionTracker tracker(binaryMap, mapData, start_floor, testbedPath);

    std::cout << "\n=== Starting Simulation ===" << std::endl;

    // 시뮬레이션 실행 - 모든 결과 저장
    // main 함수의 시뮬레이션 루프 부분
    // main 함수의 시뮬레이션 루프
    for (size_t i = 0; i < simData.size(); i++) {
        auto& step = simData[i];

        // 층이 바뀌었으면 맵 다시 로드
        if (step.floor != g_currentDisplayFloor) {
            g_currentDisplayFloor = step.floor;

            std::string floorText = (g_currentDisplayFloor < 0 ?
                                     "B" + std::to_string(abs(g_currentDisplayFloor)) :
                                     std::to_string(g_currentDisplayFloor)) + "F";
            std::string newBinaryMapFile = testbedPath + "/zeromap_" + floorText + " - origin(" +
                                           std::to_string(originMap[g_currentDisplayFloor].first) + "," +
                                           std::to_string(originMap[g_currentDisplayFloor].second) + ").txt";

            g_currentBinaryMap = readBitMatrixFromTextFile(newBinaryMapFile.c_str());

            print(color::yellow, "=== Floor changed to ", floorText, " ===", color::reset);
            print("  New map size: ", g_currentBinaryMap.height, "x", g_currentBinaryMap.width);
            print("  New origin: (", originMap[g_currentDisplayFloor].first, ", ",
                  originMap[g_currentDisplayFloor].second, ")");
        }

        try {
            auto result = tracker.processStep(step.gyro, step.compass, step.stepLen,
                                              i + 1, step.floor, 0.0f, 0);

            StepResult stepResult;
            stepResult.stepNumber = i + 1;
            stepResult.gyro = step.gyro;
            stepResult.compass = step.compass - 339.38 - 90 + 360;
            stepResult.stepLen = step.stepLen;
            stepResult.floor = step.floor;
            stepResult.result = result;

            stepResult.pdr_history = tracker.getHistoryOfPos();
            stepResult.cur_pos = tracker.getCurPos();
            stepResult.mapmatched_pos = {static_cast<int>(result.x),
                                         static_cast<int>(result.y)};
            stepResult.has_mapmatching = result.has_mapmatching;

            // 방향 추정 정보 저장
            stepResult.current_orientation = tracker.getCurrentOrientation();
            stepResult.health = tracker.getHealthMetrics();
            stepResult.hypotheses = tracker.getHypotheses();
            stepResult.compass_trend = tracker.getCompassTrend();

            // 히스토리 스냅샷 저장 (추가)
            stepResult.compass_history_snapshot = tracker.compass_analyzer.compass_history;
            stepResult.gyro_history_snapshot = tracker.compass_analyzer.gyro_history;
            stepResult.direction_offsets_snapshot = tracker.compass_analyzer.direction_offsets;  // 변경
            stepResult.accumulated_drift = tracker.drift_tracker.accumulated_drift;
            stepResult.baseline_offset = tracker.drift_tracker.baseline_offset;

            stepResult.candidates = result.candidates;

            // 회전 테스트 결과 저장
            stepResult.best_pdr_history = result.best_history;  // ProcessResult에도 추가 필요
            stepResult.best_orientation = result.gyroCaliValue;  // 이미 best_orientation 저장됨
            stepResult.base_orientation = tracker.drift_tracker.baseline_offset;

            g_allResults.push_back(stepResult);

        } catch (const std::exception& e) {
            std::cerr << "\nException at step " << (i+1) << ": " << e.what() << std::endl;
            break;
        }
    }

    std::cout << "\n=== Simulation Complete ===" << std::endl;
    std::cout << "Total steps: " << g_allResults.size() << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  A: Previous step" << std::endl;
    std::cout << "  D: Next step" << std::endl;
    std::cout << "  Q: Quit" << std::endl;

    // 시각화 시작
    g_currentDisplayFloor = start_floor;
    g_currentBinaryMap = binaryMap;  // 초기 맵 설정
    g_currentViewStep = 0;
    showStep(g_currentViewStep, testbedPath);

// 키보드 입력 처리
    while (true) {
        int key = cv::waitKey(0);

        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        } else if (key == 'a' || key == 'A') {
            if (g_currentViewStep > 0) {
                g_currentViewStep--;
                showStep(g_currentViewStep, testbedPath);
            }
        } else if (key == 'd' || key == 'D') {
            if (g_currentViewStep < g_allResults.size() - 1) {
                g_currentViewStep++;
                showStep(g_currentViewStep, testbedPath);
            }
        }
    }

    cv::destroyAllWindows();
    return 0;
}


#endif // SIMULATION_MODE