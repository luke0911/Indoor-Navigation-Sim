#include "Filter/MovingAverage.h"

MovingAverage::MovingAverage(int period)
        : period(period > 0 ? period : 0), sum(0.0f) {}

void MovingAverage::newData(float data) {
    sum += data;
    window.push(data);

    if (window.size() > period) {
        sum -= window.front();
        window.pop();
    }
}

float MovingAverage::getAvg() const {
    if (window.empty())
        return 0.0f;
    return sum / window.size();
}

void MovingAverage::reset() {
    while (!window.empty()) {
        window.pop();
    }
    sum = 0.0f;
}