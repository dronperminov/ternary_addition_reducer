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

    for (const auto& expression: expressions) {
        for (auto it1 = expression.begin(); it1 != expression.end(); it1++) {
            for (auto it2 = std::next(it1); it2 != expression.end(); it2++) {
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
    for (auto it = subexpressions.begin(); it != subexpressions.end(); ) {
        if (it->second == 1) {
            it = subexpressions.erase(it);
        } else {
            maxCount = std::max(maxCount, it->second);
            it++;
        }
    }

    return maxCount > 0;
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

    for (const auto& pair: subexpressions)
        if (pair.second == maxCount)
            top.push_back(pair.first);

    std::uniform_int_distribution<int> dist(0, top.size() - 1);
    return top[dist(generator)];
}

std::pair<int, int> AdditionReducer::selectSubexpressionGreedyRandom(std::mt19937 &generator) {
    if (uniformDistribution(generator) < scale)
        return selectSubexpressionGreedyAlternative(generator);

    return selectSubexpressionWeightedRandom(generator);
}

std::pair<int, int> AdditionReducer::selectSubexpressionGreedyIntersections(std::mt19937 &generator) {
    double maxScore = 0;
    std::pair<int, int> best = {0, 0};

    for (const auto& pair1: subexpressions) {
        double intScore = 0;

        for (const auto &pair2: subexpressions) {
            if (pair1 == pair2)
                continue;

            if (isIntersects(pair1.first, pair2.first) ^ boolDistribution(generator))
                intScore += alpha * (pair2.second - 1);
            else
                intScore += (1 - alpha) * (pair2.second - 1);
        }

        double score = pair1.second - 1 + scale * intScore;

        if (score > maxScore) {
            maxScore = score;
            best = pair1.first;
        }
    }

    return best;
}

std::pair<int, int> AdditionReducer::selectSubexpressionGreedyPotential(std::mt19937 &generator) {
    double maxScore = 0;
    std::pair<int, int> best = {0, 0};
    int varIndex = realVariables + freshVariables.size() + 1;

    for (const auto &pair: subexpressions) {
        std::pair<int, int> subexpression = pair.first;
        int i = subexpression.first;
        int j = subexpression.second;

        std::unordered_set<std::pair<int, int>, PairHash> potentialSubexpressions;
        int potential = 0;

        for (auto& expression: expressions) {
            const auto end = expression.end();
            int sign = 0;

            if (expression.find(i) != end && expression.find(j) != end) {
                expression.erase(i);
                expression.erase(j);
                expression.insert(varIndex);
                sign = 1;
            }
            else if (expression.find(-i) != end && expression.find(-j) != end) {
                expression.erase(-i);
                expression.erase(-j);
                expression.insert(-varIndex);
                sign = -1;
            }

            for (auto it1 = expression.begin(); it1 != expression.end(); it1++) {
                for (auto it2 = std::next(it1); it2 != expression.end(); it2++) {
                    int si = *it1;
                    int sj = *it2;
                    canonizeSubexpression(si, sj);
                    potentialSubexpressions.insert({si, sj});
                    potential++;
                }
            }

            if (sign) {
                expression.insert(i * sign);
                expression.insert(j * sign);
                expression.erase(varIndex * sign);
            }
        }

        double score = pair.second - 1 + scale * (potential - potentialSubexpressions.size());

        if (score > maxScore) {
            maxScore = score;
            best = subexpression;
        }
    }

    return best;
}

std::pair<int, int> AdditionReducer::selectSubexpressionWeightedRandom(std::mt19937 &generator) {
    double total = 0;
    for (const auto &pair: subexpressions)
        total += pair.second - 1;

    double p = uniformDistribution(generator) * total;
    double sum = 0;
    std::pair<int, int> last;

    for (const auto& pair: subexpressions) {
        sum += pair.second - 1;

        if (p <= sum)
            return pair.first;

        last = pair.first;
    }

    return last;
}

void AdditionReducer::replaceSubexpression(const std::pair<int, int> &subexpression) {
    int varIndex = realVariables + freshVariables.size() + 1;
    int i = subexpression.first;
    int j = subexpression.second;

    for (auto& expression: expressions) {
        if (expression.size() < 2)
            continue;

        const auto end = expression.end();

        if (expression.find(i) != end && expression.find(j) != end) {
            expression.erase(i);
            expression.erase(j);
            expression.insert(varIndex);
        }
        else if (expression.find(-i) != end && expression.find(-j) != end) {
            expression.erase(-i);
            expression.erase(-j);
            expression.insert(-varIndex);
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
