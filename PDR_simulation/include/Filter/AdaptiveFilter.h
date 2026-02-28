#ifndef ADAPTIVE_FILTER_H
#define ADAPTIVE_FILTER_H

class AdaptiveFilter {
private:
    float alphaSlow;
    float alphaFast;
    float threshold;
    float filteredValue;
    bool initialized;

public:
    AdaptiveFilter(float alphaSlow, float alphaFast, float threshold);
    float update(float newValue);
    void reset();
};

#endif // ADAPTIVE_FILTER_H
