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
    parser.add("--seed", ArgType::Natural, "INT", "random seed", "0");

    if (!parser.parse(argc, argv))
        return 0;

    std::string inputPath = parser.get("-i");
    std::string outputPath = parser.get("-o");

    int count = std::stoi(parser.get("--count"));
    double partialInitializationRate = std::stod(parser.get("--part-initialization-rate"));
    int maxNoImprovements = std::stoi(parser.get("--max-no-improvements"));
    int startAdditions = std::stoi(parser.get("--start-additions"));
    int seed = std::stoi(parser.get("--seed"));

    if (seed == 0)
        seed = time(0);

    std::cout << "Start additions reduction algorithm" << std::endl;
    std::cout << "- count: " << count << std::endl;
    std::cout << "- input path: " << inputPath << std::endl;
    std::cout << "- output path: " << outputPath << std::endl;
    std::cout << "- partial initialization rate: " << partialInitializationRate << std::endl;

    if (startAdditions > 0)
        std::cout << "- start additions: " << startAdditions << std::endl;

    std::cout << "- max no improvements: " << maxNoImprovements << std::endl;
    std::cout << "- seed: " << seed << std::endl;
    std::cout << std::endl;

    std::ifstream f(inputPath);
    if (!f) {
        std::cout << "Unable to open file \"" << inputPath << "\"" << std::endl;
        return -1;
    }

    SchemeReducer reducer(count, outputPath, seed);
    bool correct = reducer.initialize(f);
    f.close();

    if (!correct)
        return -1;

    reducer.reduce(maxNoImprovements, startAdditions, partialInitializationRate);
    return 0;
}
