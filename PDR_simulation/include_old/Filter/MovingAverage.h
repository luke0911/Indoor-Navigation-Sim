#ifndef MOVING_AVERAGE_H
#define MOVING_AVERAGE_H

#include <queue>

class MovingAverage {
private:
    std::queue<float> window;
    int period;
    float sum;

public:
    explicit MovingAverage(int period);
    void newData(float data);
    float getAvg() const;
    void reset();
};

#endif // MOVING_AVERAGE_H
