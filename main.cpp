#include <iostream>
#include <fstream>
#include <ctime>

#include "src/arg_parser.h"
#include "src/scheme_reducer.h"

int main(int argc, char *argv[]) {
    ArgParser parser("ternary_addition_reducer", "Find best additions number of the fast matrix multiplication scheme");

    parser.add("-i", ArgType::String, "PATH", "path to init scheme", "");
    parser.add("-o", ArgType::String, "PATH", "path to save schemes", "schemes");
    parser.add("--count", ArgType::Natural, "INT", "number of reducers", "8");
    parser.add("--part-initialization-rate", ArgType::Real, "REAL", "probability of partial fresh variable initialization from best solution", "0.3");
    parser.add("--start-additions", ArgType::Natural, "INT", "upper bound of additions for check optimality", "0");
    parser.add("--max-no-improvements", ArgType::Natural, "INT", "max iterations without improvements", "3");
    parser.add("--top-count", ArgType::Natural, "INT", "number of reducers for reporting", "10");
    parser.add("--seed", ArgType::Natural, "INT", "random seed", "0");
    parser.add("--ga-weight", ArgType::Real, "REAL", "weight of greedy alternative strategy", "0.25");
    parser.add("--gr-weight", ArgType::Real, "REAL", "weight of greedy random strategy", "0.1");
    parser.add("--wr-weight", ArgType::Real, "REAL", "weight of weighted random strategy", "0.1");
    parser.add("--gi-weight", ArgType::Real, "REAL", "weight of greedy intersections strategy", "0.5");
    parser.add("--gp-weight", ArgType::Real, "REAL", "weight of greedy potential strategy", "0.0");
    parser.add("--mix-weight", ArgType::Real, "REAL", "weight of mixed strategy", "0.05");

    if (!parser.parse(argc, argv))
        return 0;

    std::string inputPath = parser.get("-i");
    std::string outputPath = parser.get("-o");

    int count = std::stoi(parser.get("--count"));
    double partialInitializationRate = std::stod(parser.get("--part-initialization-rate"));
    int startAdditions = std::stoi(parser.get("--start-additions"));
    int maxNoImprovements = std::stoi(parser.get("--max-no-improvements"));
    int topCount = std::stoi(parser.get("--top-count"));
    int seed = std::stoi(parser.get("--seed"));

    StrategyWeights strategyWeights;
    strategyWeights.greedyAlternative = std::stod(parser.get("--ga-weight"));
    strategyWeights.greedyRandom = std::stod(parser.get("--gr-weight"));
    strategyWeights.weightedRandom = std::stod(parser.get("--wr-weight"));
    strategyWeights.greedyIntersections = std::stod(parser.get("--gi-weight"));
    strategyWeights.greedyPotential = std::stod(parser.get("--gp-weight"));
    strategyWeights.mix = std::stod(parser.get("--mix-weight"));

    if (strategyWeights.getTotal() <= 0) {
        std::cout << "Strategy weights are invalid (sum <= 0)" << std::endl;
        return -1;
    }

    if (seed == 0)
        seed = time(0);

    std::cout << "Start additions reduction algorithm with parameters:" << std::endl;
    std::cout << "- count: " << count << std::endl;
    std::cout << "- input path: " << inputPath << std::endl;
    std::cout << "- output path: " << outputPath << std::endl;
    std::cout << "- partial initialization rate: " << partialInitializationRate << std::endl;

    if (startAdditions > 0)
        std::cout << "- start additions: " << startAdditions << std::endl;

    std::cout << "- max no improvements: " << maxNoImprovements << std::endl;
    std::cout << "- top count: " << topCount << std::endl;
    std::cout << "- seed: " << seed << std::endl;
    std::cout << std::endl;

    std::cout << "Strategy selection weights:" << std::endl;
    std::cout << "- greedy intersections: " << strategyWeights.greedyIntersections << std::endl;
    std::cout << "- greedy alternative: " << strategyWeights.greedyAlternative << std::endl;
    std::cout << "- greedy random: " << strategyWeights.greedyRandom << std::endl;
    std::cout << "- weighted random: " << strategyWeights.weightedRandom << std::endl;
    std::cout << "- greedy potential: " << strategyWeights.greedyPotential << std::endl;
    std::cout << "- mix: " << strategyWeights.mix << std::endl;
    std::cout << std::endl;

    std::ifstream f(inputPath);
    if (!f) {
        std::cout << "Unable to open file \"" << inputPath << "\"" << std::endl;
        return -1;
    }

    SchemeReducer reducer(count, outputPath, strategyWeights, seed);
    bool correct = reducer.initialize(f);
    f.close();

    if (!correct)
        return -1;

    reducer.reduce(maxNoImprovements, startAdditions, partialInitializationRate, topCount);
    return 0;
}
