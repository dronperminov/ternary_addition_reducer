#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>

#include "scheme.h"
#include "addition_reducer.h"

const int modes[] = {
    GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE, GREEDY_INTERSECTIONS_MODE,
    GREEDY_ALTERNATIVE_MODE, GREEDY_ALTERNATIVE_MODE, GREEDY_ALTERNATIVE_MODE, GREEDY_ALTERNATIVE_MODE,
    GREEDY_RANDOM_MODE, GREEDY_RANDOM_MODE,
    WEIGHTED_RANDOM_MODE,
    GREEDY_POTENTIAL_MODE,
    MIX_MODE
};

const int modesCount = sizeof(modes) / sizeof(modes[0]);

class SchemeReducer {
    int dimension[3];
    int rank;
    int count;

    std::string path;
    std::vector<AdditionReducer> uvw[3];
    AdditionReducer best[3];

    int bestAdditions[3];
    int bestFreshVars[3];
    std::vector<int> indices[3];
    std::vector<std::mt19937> generators;

    int reducedAdditions;
    int reducedFreshVars;

    std::uniform_real_distribution<double> uniformDistribution;
    std::uniform_int_distribution<int> modeDistribution;
public:
    SchemeReducer(int count, const std::string path, int seed);

    bool initialize(std::istream &is);
    void reduce(int maxNoImprovements, int startAdditions, double partialInitializationRate, int topCount = 10);
private:
    bool parseScheme(const Scheme &scheme);
    void reduceIteration(double partialInitializationRate);
    bool updateBest(int index, int topCount);
    bool update(int startAdditions, int topCount);
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int topCount);
    void save() const;

    std::string getSavePath() const;
    std::string getDimension() const;
    std::string prettyTime(double elapsed);
};
