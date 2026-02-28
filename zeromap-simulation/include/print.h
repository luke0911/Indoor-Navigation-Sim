// include/print.h
#ifndef PRINT_H
#define PRINT_H

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

// 색상 코드
namespace color {
    const std::string red = "\033[31m";
    const std::string green = "\033[32m";
    const std::string yellow = "\033[33m";
    const std::string blue = "\033[34m";
    const std::string magenta = "\033[35m";
    const std::string cyan = "\033[36m";
    const std::string white = "\033[37m";
    const std::string gray = "\033[90m";
    const std::string reset = "\033[0m";
}

// 기본 print 함수 - inline 추가
template<typename... Args>
inline void print(Args... args) {
    ((std::cout << args << " "), ...);
    std::cout << '\n';
}

// 구분자 없이 출력 - inline 추가
template<typename... Args>
inline void printRaw(Args... args) {
    ((std::cout << args), ...);
    std::cout << '\n';
}

// 줄바꿈 없이 출력 - inline 추가
template<typename... Args>
inline void printNoLine(Args... args) {
    ((std::cout << args << " "), ...);
}

// 화면 클리어 - inline 추가
inline void clear() {
    std::cout << "\033[2J\033[1;1H";
}

// Python의 f-string처럼 사용할 수 있는 format 함수 - inline 추가
template<typename T>
inline std::string str(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

// 소수점 자리수 지정 - inline 추가
inline std::string fixed(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

#endif