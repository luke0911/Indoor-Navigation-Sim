#ifndef PDRRESULT_H
#define PDRRESULT_H

struct PDR {
    double stepLength;
    double direction;           // degrees [0,360)
    int totalStepCount;
};

#endif // PDRRESULT_H
