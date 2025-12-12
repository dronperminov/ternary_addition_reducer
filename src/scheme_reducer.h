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

struct StrategyWeights {
    double greedyIntersections;
    double greedyAlternative;
    double greedyRandom;
    double weightedRandom;
    double greedyPotential;
    double mix;
};

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
public:
    SchemeReducer(int count, const std::string path, int seed);

    bool initialize(std::istream &is);
    void reduce(const StrategyWeights &weights, int maxNoImprovements, int startAdditions, double partialInitializationRate, int topCount = 10);
private:
    bool parseScheme(const Scheme &scheme);
    void reduceIteration(const StrategyWeights &weights, double partialInitializationRate);
    bool updateBest(int index, int topCount);
    bool update(int startAdditions, int topCount);
    void report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int topCount);
    void save() const;

    SelectSubexpressionMode selectStrategy(const StrategyWeights &weights, std::mt19937 &generator);
    std::string getSavePath() const;
    std::string getDimension() const;
    std::string prettyTime(double elapsed) const;
};
