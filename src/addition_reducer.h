#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <bitset>
#include <unordered_set>
#include <unordered_map>

enum SelectSubexpressionMode {
    GREEDY_MODE = 0,
    GREEDY_ALTERNATIVE_MODE = 1,
    GREEDY_RANDOM_MODE = 2,
    GREEDY_INTERSECTIONS_MODE = 3,
    WEIGHTED_RANDOM_MODE = 4,
    GREEDY_POTENTIAL_MODE = 5,
    MIX_MODE = 6
};

struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        return h1 ^ (h2 << 1); 
    }
};

class AdditionReducer {
    int realVariables;
    int naiveAdditions;
    int maxCount;
    int mode;
    double scale;
    double alpha;
    double beta;

    std::vector<std::unordered_set<int>> expressions;
    std::vector<std::pair<int,int>> freshVariables;
    std::unordered_map<std::pair<int, int>, int, PairHash> subexpressions;

    std::uniform_real_distribution<double> uniformDistribution;
    std::uniform_int_distribution<int> boolDistribution;
    std::uniform_int_distribution<int> modeDistribution;
public:
    AdditionReducer();

    bool addExpression(const std::vector<int> &expression);
    void setMode(int mode);
    void partialInitialize(const AdditionReducer &reducer, size_t count);

    void copyFrom(const AdditionReducer &reducer);
    void reduce(std::mt19937 &generator);
    void write(std::ostream &os, const std::string &name, const std::string &indent) const;

    int getNaiveAdditions() const;
    int getAdditions() const;
    int getFreshVars() const;
    std::string getMode() const;
private:
    bool updateSubexpressions();
    void canonizeSubexpression(int &i, int &j) const;
    std::pair<int, int> selectSubexpression(int mode, std::mt19937 &generator);
    void replaceSubexpression(const std::pair<int, int> &subexpression);
    void evaluatePotentialParams();

    std::pair<int, int> selectSubexpressionGreedy();
    std::pair<int, int> selectSubexpressionGreedyAlternative(std::mt19937 &generator);
    std::pair<int, int> selectSubexpressionGreedyRandom(std::mt19937 &generator);
    std::pair<int, int> selectSubexpressionGreedyIntersections(std::mt19937 &generator);
    std::pair<int, int> selectSubexpressionWeightedRandom(std::mt19937 &generator);
    std::pair<int, int> selectSubexpressionGreedyPotential(std::mt19937 &generator);

    bool isIntersects(const std::pair<int, int> pair1, const std::pair<int, int> &pair2) const;
};
