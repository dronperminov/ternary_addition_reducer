#pragma once

#include <iostream>
#include <vector>

struct Scheme {
    int dimension[3];
    int elements[3];
    int rank;
    std::vector<int> uvw[3];

    Scheme(int n1, int n2, int n3, int rank);

    bool validate() const;
    bool read(std::istream &is, bool check = true);
private:
    bool validateEquation(int i, int j, int k) const;
};
