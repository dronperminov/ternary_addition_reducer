#include "addition_reducer.h"

AdditionReducer::AdditionReducer() : uniformDistribution(0.0, 1.0), boolDistribution(0, 1), modeDistribution(0, MIX_MODE - 1) {
    realVariables = 0;
    naiveAdditions = 0;
    maxCount = 0;
    mode = GREEDY_MODE;
    scale = -1;
    alpha = -1;
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

void AdditionReducer::setMode(int mode) {
    this->mode = mode;
    this->scale = -1;
    this->alpha = -1;
    this->beta = -1;
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
    alpha = -1 + uniformDistribution(generator) * 2;
    beta = 0.5 + uniformDistribution(generator) * 0.5;

    std::uniform_int_distribution<int> modeDistribution(0, MIX_MODE - 1);

    while (updateSubexpressions()) {
        int stepMode = mode == MIX_MODE ? modeDistribution(generator) : mode;
        std::pair<int, int> subexpression = selectSubexpression(stepMode, generator);
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

std::string AdditionReducer::getMode() const {
    if (mode == GREEDY_MODE)
        return "g";

    if (mode == GREEDY_ALTERNATIVE_MODE)
        return "ga";

    if (mode == GREEDY_RANDOM_MODE)
        return "gr" + std::to_string(int(scale * 100));

    if (mode == GREEDY_INTERSECTIONS_MODE)
        return "gi" + std::to_string(int(scale * 100));

    if (mode == WEIGHTED_RANDOM_MODE)
        return "wr";

    if (mode == GREEDY_POTENTIAL_MODE)
        return "gp" + std::to_string(int(scale * 100));;

    return "mix";
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

std::pair<int, int> AdditionReducer::selectSubexpression(int mode, std::mt19937 &generator) {
    if (mode == GREEDY_MODE)
        return selectSubexpressionGreedy();

    if (mode == GREEDY_ALTERNATIVE_MODE)
        return selectSubexpressionGreedyAlternative(generator);

    if (mode == GREEDY_RANDOM_MODE)
        return selectSubexpressionGreedyRandom(generator);

    if (mode == GREEDY_INTERSECTIONS_MODE)
        return selectSubexpressionGreedyIntersections(generator);

    if (mode == WEIGHTED_RANDOM_MODE)
        return selectSubexpressionWeightedRandom(generator);

    if (mode == GREEDY_POTENTIAL_MODE)
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
    std::uniform_real_distribution<double> uniform(0.0, 0.5);
    bool top = uniform(generator) < scale;

    std::vector<std::pair<int, int>> pairs;

    for (auto it = subexpressions.begin(); it != subexpressions.end(); it++)
        if ((top && it->second == maxCount) || (!top && it->second > 1))
            pairs.push_back(it->first);

    std::uniform_int_distribution<int> dist(0, pairs.size() - 1);
    return pairs[dist(generator)];
}

std::pair<int, int> AdditionReducer::selectSubexpressionGreedyIntersections(std::mt19937 &generator) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(subexpressions.size());

    for (auto it = subexpressions.begin(); it != subexpressions.end(); it++)
        if (it->second > 1)
            pairs.push_back(it->first);

    int size = pairs.size();
    std::vector<bool> intersections(size * size, false);

    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            bool intersects = isIntersects(pairs[i], pairs[j]);

            if (boolDistribution(generator))
                intersects = !intersects;

            intersections[i * size + j] = intersects;
            intersections[j * size + i] = intersects;
        }
    }

    double maxScore = 0;
    int imax = 0;

    for (int i = 0; i < size; i++) {
        int overlapped = 0;
        int other = 0;

        for (int j = 0; j < size; j++) {
            if (i == j)
                continue;

            if (intersections[i * size + j])
                overlapped += subexpressions.at(pairs[j]) - 1;
            else
                other += subexpressions.at(pairs[j]) - 1;
        }

        double score = subexpressions.at(pairs[i]) - 1 + scale * (other + alpha + overlapped * beta);

        if (score > maxScore) {
            maxScore = score;
            imax = i;
        }
    }

    return pairs[imax];
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

bool AdditionReducer::isIntersects(const std::pair<int, int> pair1, const std::pair<int, int> &pair2) const {
    int i1 = pair1.first;
    int j1 = pair1.second;

    int i2 = pair2.first;
    int j2 = pair2.second;

    return i1 == i2 || i1 == abs(j2) || abs(j1) == i2 || j1 == j2;
}
