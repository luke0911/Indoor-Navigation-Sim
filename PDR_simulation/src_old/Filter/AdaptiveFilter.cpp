#include "Filter/AdaptiveFilter.h"
#include <cmath>

AdaptiveFilter::AdaptiveFilter(float alphaSlow, float alphaFast, float threshold)
        : alphaSlow(alphaSlow),
          alphaFast(alphaFast),
          threshold(threshold),
          filteredValue(0.0f),
          initialized(false) {}

float AdaptiveFilter::update(float newValue) {
    if (!initialized) {
        filteredValue = newValue;
        initialized = true;
    } else {
        float diff = std::fabs(newValue - filteredValue);
        float alpha = (diff > threshold) ? alphaFast : alphaSlow;
        filteredValue = alpha * newValue + (1.0f - alpha) * filteredValue;
    }
    return filteredValue;
}

void AdaptiveFilter::reset() {
    filteredValue = 0.0f;
    initialized = false;
}