#include "addition_reducer.h"

StrategyWeights::StrategyWeights() : uniformDistribution(0.0, 1.0) {
    greedyAlternative = 4;
    greedyRandom = 2;
    weightedRandom = 1;
    greedyIntersections = 8;
    greedyPotential = 0;
    mix = 0;
}

double StrategyWeights::getTotal() const {
    return greedyIntersections + greedyAlternative + greedyRandom + weightedRandom + greedyPotential + mix;
}

Strategy StrategyWeights::select(std::mt19937 &generator) {
    Strategy strategies[] = {
        Strategy::GreedyAlternative, Strategy::GreedyRandom, Strategy::WeightedRandom,
        Strategy::GreedyIntersections, Strategy::GreedyPotential, Strategy::Mix
    };

    double weights[] = {
        greedyAlternative, greedyRandom, weightedRandom,
        greedyIntersections, greedyPotential, mix
    };

    double p = uniformDistribution(generator) * getTotal();
    double sum = 0;
    int last = 0;

    for (int i = 0; i < 6; i++) {
        if (weights[i] == 0)
            continue;

        sum += weights[i];
        last = i;

        if (p <= sum)
            return strategies[i];
    }

    return strategies[last];
}


AdditionReducer::AdditionReducer() : uniformDistribution(0.0, 1.0), boolDistribution(0, 1) {
    realVariables = 0;
    naiveAdditions = 0;
    maxCount = 0;
    strategy = Strategy::Greedy;
    scale = 0;
    alpha = 0;
}

bool AdditionReducer::addExpression(const std::vector<int> &expression) {
    std::unordered_set<int> parsed;
    int variables = expression.size();

    for (int i = 0; i < variables; i++) {
        if (expression[i] == 0)
            continue;

        if (expression[i] == 1)
            parsed.insert(i + 1);
        else if (expression[i] == -1)
            parsed.insert(-i - 1);
        else
            return false;
    }

    expressions.push_back(parsed);
    naiveAdditions += parsed.size() - 1;

    if (variables > realVariables)
        realVariables = variables;

    return true;
}

void AdditionReducer::setStrategy(Strategy strategy) {
    this->strategy = strategy;
    this->scale = 0;
    this->alpha = 0;
}

void AdditionReducer::partialInitialize(const AdditionReducer &reducer, size_t count) {
    for (size_t index = 0; index < count && index < reducer.freshVariables.size(); index++)
        replaceSubexpression(reducer.freshVariables[index]);
}

void AdditionReducer::copyFrom(const AdditionReducer &reducer) {
    realVariables = reducer.realVariables;
    naiveAdditions = reducer.naiveAdditions;
    maxCount = reducer.maxCount;

    freshVariables = std::vector<std::pair<int, int>>(reducer.freshVariables);
    expressions.clear();

    for (size_t i = 0; i < reducer.expressions.size(); i++)
        expressions.push_back(std::unordered_set<int>(reducer.expressions[i]));
}

void AdditionReducer::reduce(std::mt19937 &generator) {
    scale = uniformDistribution(generator) * 0.5;
    alpha = 0.5 + uniformDistribution(generator) * 0.5;

    while (updateSubexpressions()) {
        std::pair<int, int> subexpression = selectSubexpression(generator);
        replaceSubexpression(subexpression);
    }
}

int AdditionReducer::getNaiveAdditions() const {
    return naiveAdditions;
}

int AdditionReducer::getAdditions() const {
    int additions = freshVariables.size();

    for (size_t i = 0; i < expressions.size(); i++)
        additions += expressions[i].size() - 1;

    return additions;
}

int AdditionReducer::getFreshVars() const {
    return freshVariables.size();
}

std::string AdditionReducer::getStrategy() const {
    if (strategy == Strategy::Greedy)
        return "g";

    if (strategy == Strategy::GreedyAlternative)
        return "ga";

    if (strategy == Strategy::WeightedRandom)
        return "wr";

    if (strategy == Strategy::Mix)
        return "mix";

    std::stringstream ss;
    if (strategy == Strategy::GreedyRandom)
        ss << "gr (" << int(scale * 100) << ")";
    else if (strategy == Strategy::GreedyIntersections)
        ss << "gi (" << int(scale * 100) << ")";
    else if (strategy == Strategy::GreedyPotential)
        ss << "gp (" << int(scale * 100) << ")";

    return ss.str();
}

void AdditionReducer::write(std::ostream &os, const std::string &name, const std::string &indent) const {
    os << indent << "\"" << name << "_fresh\": [" << std::endl;

    for (size_t i = 0; i < freshVariables.size(); i++) {
        int index1 = abs(freshVariables[i].first) - 1;
        int value1 = freshVariables[i].first > 0 ? 1 : -1;

        int index2 = abs(freshVariables[i].second) - 1;
        int value2 = freshVariables[i].second > 0 ? 1 : -1;

        os << indent << indent << "[{\"index\": " << index1 << ", \"value\": " << value1 << "}, {\"index\": " << index2 << ", \"value\": " << value2 << "}]";

        if (i < freshVariables.size() - 1)
            os << ",";

        os << std::endl;
    }

    os << indent << "]," << std::endl;
    os << indent << "\"" << name << "\": [" << std::endl;

    for (size_t i = 0; i < expressions.size(); i++) {
        os << indent << indent << "[";

        for (auto j = expressions[i].begin(); j != expressions[i].end(); j++) {
            int index = abs(*j) - 1;
            int value = *j > 0 ? 1 : -1;

            if (j != expressions[i].begin())
                os << ", ";

            os << "{\"index\": " << index << ", \"value\": " << value << "}";
        }

        os << "]";

        if (i < expressions.size() - 1)
            os << ",";

        os << std::endl;
    }

    os << indent << "]";
}

bool AdditionReducer::updateSubexpressions() {
    subexpressions.clear();

    for (size_t index = 0; index < expressions.size(); index++) {
        for (auto it1 = expressions[index].begin(); it1 != expressions[index].end(); it1++) {
            for (auto it2 = std::next(it1); it2 != expressions[index].end(); it2++) {
                int i = *it1;
                int j = *it2;
                canonizeSubexpression(i, j);

                auto result = subexpressions.find({i, j});
                if (result == subexpressions.end())
                    subexpressions[{i, j}] = 1;
                else
                    result->second++;
            }
        }
    }

    maxCount = 0;
    for (auto it = subexpressions.begin(); it != subexpressions.end(); it++)
        maxCount = std::max(maxCount, it->second);

    return maxCount > 1;
}

void AdditionReducer::canonizeSubexpression(int &i, int &j) const {
    if (abs(i) > abs(j))
        std::swap(i, j);

    if (i < 0) {
        i = -i;
        j = -j;
    }
}

std::pair<int, int> AdditionReducer::selectSubexpression(std::mt19937 &generator) {
    Strategy strategy = getStepStrategy(generator);

    if (strategy == Strategy::GreedyAlternative)
        return selectSubexpressionGreedyAlternative(generator);

    if (strategy == Strategy::GreedyRandom)
        return selectSubexpressionGreedyRandom(generator);

    if (strategy == Strategy::GreedyIntersections)
        return selectSubexpressionGreedyIntersections(generator);

    if (strategy == Strategy::WeightedRandom)
        return selectSubexpressionWeightedRandom(generator);

    if (strategy == Strategy::GreedyPotential)
        return selectSubexpressionGreedyPotential(generator);

    return selectSubexpressionGreedy();
}

std::pair<int, int> AdditionReducer::selectSubexpressionGreedy() {
    auto it = subexpressions.begin();

    while (it != subexpressions.end() && it->second != maxCount)
        it++;

    return it->first;
}

std::pair<int, int> AdditionReducer::selectSubexpressionGreedyAlternative(std::mt19937 &generator) {
    std::vector<std::pair<int, int>> top;

    for (auto it = subexpressions.begin(); it != subexpressions.end(); it++)
        if (it->second == maxCount)
            top.push_back(it->first);

    std::uniform_int_distribution<int> dist(0, top.size() - 1);
    return top[dist(generator)];
}

std::pair<int, int> AdditionReducer::selectSubexpressionGreedyRandom(std::mt19937 &generator) {
    bool top = uniformDistribution(generator) < scale;
    std::vector<std::pair<int, int>> pairs;

    for (auto it = subexpressions.begin(); it != subexpressions.end(); it++)
        if ((top && it->second == maxCount) || (!top && it->second > 1))
            pairs.push_back(it->first);

    std::uniform_int_distribution<int> dist(0, pairs.size() - 1);
    return pairs[dist(generator)];
}

std::pair<int, int> AdditionReducer::selectSubexpressionGreedyIntersections(std::mt19937 &generator) {
    double maxScore = 0;
    std::pair<int, int> best = {0, 0};

    for (auto it = subexpressions.begin(); it != subexpressions.end(); it++) {
        if (it->second == 1)
            continue;

        double intScore = 0;

        for (auto jt = subexpressions.begin(); jt != subexpressions.end(); jt++) {
            if (it == jt || jt->second == 1)
                continue;

            if (isIntersects(it->first, jt->first) ^ boolDistribution(generator))
                intScore += alpha * (jt->second - 1);
            else
                intScore += (1 - alpha) * (jt->second - 1);
        }

        double score = it->second - 1 + scale * intScore;

        if (score > maxScore) {
            maxScore = score;
            best = it->first;
        }
    }

    return best;
}

std::pair<int, int> AdditionReducer::selectSubexpressionGreedyPotential(std::mt19937 &generator) {
    double maxScore = 0;
    std::pair<int, int> best = {0, 0};
    int varIndex = realVariables + freshVariables.size() + 1;

    for (auto it = subexpressions.begin(); it != subexpressions.end(); it++) {
        if (it->second == 1)
            continue;

        std::pair<int, int> pair = it->first;
        int i = pair.first;
        int j = pair.second;

        std::unordered_set<std::pair<int, int>, PairHash> potentialSubexpressions;
        int potential = 0;

        for (size_t index = 0; index < expressions.size(); index++) {
            auto end = expressions[index].end();
            int sign = 0;

            if (expressions[index].find(i) != end && expressions[index].find(j) != end) {
                expressions[index].erase(i);
                expressions[index].erase(j);
                expressions[index].insert(varIndex);
                sign = 1;
            }
            else if (expressions[index].find(-i) != end && expressions[index].find(-j) != end) {
                expressions[index].erase(-i);
                expressions[index].erase(-j);
                expressions[index].insert(-varIndex);
                sign = -1;
            }

            for (auto it1 = expressions[index].begin(); it1 != expressions[index].end(); it1++) {
                for (auto it2 = std::next(it1); it2 != expressions[index].end(); it2++) {
                    int si = *it1;
                    int sj = *it2;
                    canonizeSubexpression(si, sj);
                    potentialSubexpressions.insert({si, sj});
                    potential++;
                }
            }

            if (sign) {
                expressions[index].insert(i * sign);
                expressions[index].insert(j * sign);
                expressions[index].erase(varIndex * sign);
            }
        }

        double score = it->second - 1 + scale * (potential - potentialSubexpressions.size());

        if (score > maxScore) {
            maxScore = score;
            best = pair;
        }
    }

    return best;
}

std::pair<int, int> AdditionReducer::selectSubexpressionWeightedRandom(std::mt19937 &generator) {
    double total = 0;

    for (auto it = subexpressions.begin(); it != subexpressions.end(); it++)
        total += it->second - 1;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double p = uniform(generator) * total;
    double sum = 0;

    auto it = subexpressions.begin();
    while (std::next(it) != subexpressions.end() && p > sum) {
        sum += it->second - 1;
        it++;
    }

    return it->first;
}

void AdditionReducer::replaceSubexpression(const std::pair<int, int> &subexpression) {
    int varIndex = realVariables + freshVariables.size() + 1;
    int i = subexpression.first;
    int j = subexpression.second;

    for (size_t index = 0; index < expressions.size(); index++) {
        auto end = expressions[index].end();

        if (expressions[index].find(i) != end && expressions[index].find(j) != end) {
            expressions[index].erase(i);
            expressions[index].erase(j);
            expressions[index].insert(varIndex);
        }
        else if (expressions[index].find(-i) != end && expressions[index].find(-j) != end) {
            expressions[index].erase(-i);
            expressions[index].erase(-j);
            expressions[index].insert(-varIndex);
        }
    }

    freshVariables.push_back({i, j});
}

Strategy AdditionReducer::getStepStrategy(std::mt19937 &generator) {
    if (strategy == Strategy::Mix)
        return strategyWeights.select(generator);

    return strategy;
}

bool AdditionReducer::isIntersects(const std::pair<int, int> pair1, const std::pair<int, int> &pair2) const {
    int i1 = pair1.first;
    int j1 = pair1.second;

    int i2 = pair2.first;
    int j2 = pair2.second;

    return i1 == i2 || i1 == abs(j2) || abs(j1) == i2 || j1 == j2;
}
