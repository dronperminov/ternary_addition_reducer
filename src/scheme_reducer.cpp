#include "scheme_reducer.h"

SchemeReducer::SchemeReducer(int count, const std::string path, const StrategyWeights &strategyWeights, int seed) : uniformDistribution(0.0, 1.0) {
    this->count = count;
    this->path = path;
    this->strategyWeights = strategyWeights;

    for (int i = 0; i < 3; i++) {
        uvw[i] = std::vector<AdditionReducer>(count);
        indices[i].reserve(count);

        for (int j = 0; j < count; j++)
            indices[i].push_back(j);
    }

    int maxThreads = omp_get_max_threads();
    for (int i = 0; i < maxThreads; i++)
        generators.emplace_back(seed + i);
}

bool SchemeReducer::initialize(std::istream &is) {
    is >> dimension[0] >> dimension[1] >> dimension[2] >> rank;
    std::cout << "Reading scheme " << dimension[0] << "x" << dimension[1] << "x" << dimension[2] << " with " << rank << " multiplications: ";

    Scheme scheme(dimension[0], dimension[1], dimension[2], rank);
    if (!scheme.read(is)) {
        std::cout << "error, readed scheme is invalid" << std::endl;
        return false;
    }

    if (!parseScheme(scheme)) {
        std::cout << "error, readed scheme has non ternary coefficients" << std::endl;
        return false;
    }

    std::cout << "success" << std::endl << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < 3; i++) {
        best[i].copyFrom(init[i]);
        bestAdditions[i] = init[i].getNaiveAdditions();
        bestStrategies[i] = init[i].getStrategy();
        bestFreshVars[i] = 0;
    }

    naiveAdditions = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];
    reducedAdditions = naiveAdditions;
    reducedFreshVars = 0;

    std::cout << "Readed scheme params:" << std::endl;
    std::cout << "- dimensions: " << dimension[0] << "x" << dimension[1] << "x" << dimension[2] << std::endl;
    std::cout << "- multiplications (rank): " << rank << std::endl;
    std::cout << "- naive additions (U / V / W / total): " << bestAdditions[0] << " / " << bestAdditions[1] << " / " << bestAdditions[2] << " / " << reducedAdditions << std::endl;
    std::cout << std::endl;
    return true;
}

void SchemeReducer::reduce(int maxNoImprovements, int startAdditions, double partialInitializationRate, int topCount) {
    int noImprovements = 0;

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<double> elapsedTimes;
    topCount = std::min(topCount, count);

    for (int iteration = 1; noImprovements < maxNoImprovements; iteration++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        reduceIteration(iteration, partialInitializationRate);
        bool improved = update(startAdditions, topCount);
        auto t2 = std::chrono::high_resolution_clock::now();

        elapsedTimes.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0);
        report(startTime, iteration, elapsedTimes, topCount);

        if (improved) {
            noImprovements = 0;
        }
        else {
            noImprovements++;
            std::cout << "No improvements for " << noImprovements << " iterations" << std::endl;
        }
    }
}

bool SchemeReducer::parseScheme(const Scheme &scheme) {
    for (int i = 0; i < 3; i++)
        dimension[i] = scheme.dimension[i];

    rank = scheme.rank;

    for (int i = 0; i < scheme.rank; i++) {
        std::vector<int> expression(scheme.elements[0]);

        for (int j = 0; j < scheme.elements[0]; j++)
            expression[j] = scheme.uvw[0][i * scheme.elements[0] + j];

        if (!init[0].addExpression(expression))
            return false;
    }

    for (int i = 0; i < scheme.rank; i++) {
        std::vector<int> expression(scheme.elements[1]);

        for (int j = 0; j < scheme.elements[1]; j++)
            expression[j] = scheme.uvw[1][i * scheme.elements[1] + j];

        if (!init[1].addExpression(expression))
            return false;
    }

    for (int i = 0; i < scheme.elements[2]; i++) {
        std::vector<int> expression(scheme.rank);

        for (int j = 0; j < scheme.rank; j++)
            expression[j] = scheme.uvw[2][j * scheme.elements[2] + i];

        if (!init[2].addExpression(expression))
            return false;
    }

    return true;
}

void SchemeReducer::reduceIteration(int iteration, double partialInitializationRate) {
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        auto& generator = generators[omp_get_thread_num()];

        for (int j = 0; j < 3; j++) {
            uvw[j][i].copyFrom(init[j]);
            uvw[j][i].setStrategy(iteration == 1 && i == 0 ? Strategy::Greedy : strategyWeights.select(generator));

            if (uniformDistribution(generator) < partialInitializationRate && best[j].getFreshVars() > 0) {
                std::uniform_int_distribution<int> varsDistribution(1, best[j].getFreshVars() * 3 / 4);
                uvw[j][i].partialInitialize(best[j], varsDistribution(generator));
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < count; i++)
        for (int j = 0; j < 3; j++)
            uvw[j][i].reduce(generators[omp_get_thread_num()]);
}

bool SchemeReducer::updateBest(int index, int topCount) {
    std::partial_sort(indices[index].begin(), indices[index].begin() + topCount, indices[index].end(), [this, index](int index1, int index2) {
        int additions1 = uvw[index][index1].getAdditions();
        int additions2 = uvw[index][index2].getAdditions();

        if (additions1 != additions2)
            return additions1 < additions2;

        return uvw[index][index1].getFreshVars() < uvw[index][index2].getFreshVars();
    });

    int top = indices[index][0];
    int additions = uvw[index][top].getAdditions();
    int freshVars = uvw[index][top].getFreshVars();
    std::string strategy = uvw[index][top].getStrategy();

    if (additions < bestAdditions[index] || (additions == bestAdditions[index] && freshVars < bestFreshVars[index])) {
        bestAdditions[index] = additions;
        bestFreshVars[index] = freshVars;
        bestStrategies[index] = strategy;
        best[index].copyFrom(uvw[index][top]);
        return true;
    }

    return false;
}

bool SchemeReducer::update(int startAdditions, int topCount) {
    bool updated = false;

    for (int i = 0; i < 3; i++)
        if (updateBest(i, topCount))
            updated = true;

    if (!updated)
        return false;

    int additions = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];
    int freshVars = bestFreshVars[0] + bestFreshVars[1] + bestFreshVars[2];

    if (additions < reducedAdditions)
        std::cout << "Reduced scheme improved from " << reducedAdditions << " to " << additions << " additions (fresh vars: " << freshVars << ")" << std::endl;
    else
        std::cout << "Reduced scheme improved from " << reducedFreshVars << " fresh vars to " << freshVars << " fresh vars (additions: " << reducedAdditions << ")" << std::endl;

    reducedAdditions = additions;
    reducedFreshVars = freshVars;

    if (reducedAdditions < startAdditions || startAdditions == 0)
        save();

    return true;
}

void SchemeReducer::report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes, int topCount) {
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    double lastTime = elapsedTimes[elapsedTimes.size() - 1];
    double minTime = *std::min_element(elapsedTimes.begin(), elapsedTimes.end());
    double maxTime = *std::max_element(elapsedTimes.begin(), elapsedTimes.end());
    double meanTime = std::accumulate(elapsedTimes.begin(), elapsedTimes.end(), 0.0) / elapsedTimes.size();

    std::string dimension = getDimension();

    std::cout << std::endl;
    std::cout << "+--------------------------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "| " << std::left;
    std::cout << "Size: " << std::setw(20) << dimension << "   ";
    std::cout << "Reducers count: " << std::setw(10) << count << "   ";
    std::cout << std::setw(44) << std::right << ("Iteration: " + std::to_string(iteration));
    std::cout << " |" << std::endl;

    std::cout << "| " << std::left;
    std::cout << "Rank: " << std::setw(20) << rank << "   ";
    std::cout << "Naive additions: " << std::setw(9) << naiveAdditions << "   ";
    std::cout <<  std::setw(44) << std::right << ("Elapsed: " + prettyTime(elapsed));
    std::cout << " |" << std::endl;

    std::cout << std::right;
    std::cout << "+============================+============================+============================+=================+" << std::endl;
    std::cout << "|         Reducers U         |         Reducers V         |         Reducers W         |      Total      |" << std::endl;
    std::cout << "+----------+---------+-------+----------+---------+-------+----------+---------+-------+---------+-------+" << std::endl;
    std::cout << "| strategy | reduced | fresh | strategy | reduced | fresh | strategy | reduced | fresh | reduced | fresh |" << std::endl;
    std::cout << "+----------+---------+-------+----------+---------+-------+----------+---------+-------+---------+-------+" << std::endl;

    for (int i = 0; i < topCount && i < count; i++) {
        std::cout << "| ";

        int reduced = 0;
        int fresh = 0;

        for (int j = 0; j < 3; j++) {
            int index = indices[j][i];
            std::string strategy = uvw[j][index].getStrategy();
            int currReduced = uvw[j][index].getAdditions();
            int currFresh = uvw[j][index].getFreshVars();

            reduced += currReduced;
            fresh += currFresh;

            std::cout << std::left << std::setw(8) << strategy << "   " << std::right << std::setw(7) << currReduced << "   " << std::setw(5) << currFresh << " | ";
        }

        std::cout << std::setw(7) << reduced << "   " << std::setw(5) << fresh << " | ";
        std::cout << std::endl;
    }

    std::cout << "+----------------------------+----------------------------+----------------------------+-----------------+" << std::endl;
    std::cout << "- iteration time (last / min / max / mean): " << prettyTime(lastTime) << " / " << prettyTime(minTime) << " / " << prettyTime(maxTime) << " / " << prettyTime(meanTime) << std::endl;
    std::cout << "- best additions (U / V / W / total): " << bestAdditions[0] << " / " << bestAdditions[1] << " / " << bestAdditions[2] << " / " << reducedAdditions << std::endl;
    std::cout << "- best fresh vars (U / V / W / total): " << bestFreshVars[0] << " / " << bestFreshVars[1] << " / " << bestFreshVars[2] << " / " << reducedFreshVars << std::endl;
    std::cout << "- best strategies (U / V / W): " << bestStrategies[0] << " / " << bestStrategies[1] << " / " << bestStrategies[2] << std::endl;
    std::cout << std::endl;
}

void SchemeReducer::save() const {
    std::string path = getSavePath();

    std::ofstream f(path);

    f << "{" << std::endl;
    f << "    \"n\": [" << dimension[0] << ", " << dimension[1] << ", " << dimension[2] << "]," << std::endl;
    f << "    \"m\": " << rank << "," << std::endl;
    f << "    \"z2\": false," << std::endl;
    f << "    \"complexity\": {\"naive\": " << naiveAdditions << ", \"reduced\": " << reducedAdditions << "}," << std::endl;
    best[0].write(f, "u", "    ");
    f << "," << std::endl;
    best[1].write(f, "v", "    ");
    f << "," << std::endl;
    best[2].write(f, "w", "    ");
    f << std::endl;
    f << "}" << std::endl;
    f.close();

    std::cout << "Reduced scheme saved to \"" << path << "\"" << std::endl;
}

std::string SchemeReducer::getSavePath() const {
    std::stringstream ss;
    ss << path << "/";
    ss << dimension[0] << "x" << dimension[1] << "x" << dimension[2];
    ss << "_m" << rank;
    ss << "_cr" << reducedAdditions;
    ss << "_fv" << reducedFreshVars;
    ss << "_cn" << naiveAdditions;
    ss << "_ZT";
    ss << "_reduced.json";

    return ss.str();
}

std::string SchemeReducer::getDimension() const {
    std::stringstream ss;
    ss << dimension[0] << "x" << dimension[1] << "x" << dimension[2];
    return ss.str();
}

std::string SchemeReducer::prettyTime(double elapsed) const {
    std::stringstream ss;

    if (elapsed < 60) {
        ss << std::setprecision(3) << std::fixed << elapsed;
    }
    else {
        int seconds = int(elapsed + 0.5);
        int hours = seconds / 3600;
        int minutes = (seconds % 3600) / 60;

        ss << std::setw(2) << std::setfill('0') << hours << ":";
        ss << std::setw(2) << std::setfill('0') << minutes << ":";
        ss << std::setw(2) << std::setfill('0') << (seconds % 60);
    }

    return ss.str();
}
