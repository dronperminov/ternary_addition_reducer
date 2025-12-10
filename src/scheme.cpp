#include "scheme.h"

Scheme::Scheme(int n1, int n2, int n3, int rank) {
    this->dimension[0] = n1;
    this->dimension[1] = n2;
    this->dimension[2] = n3;
    this->rank = rank;

    this->elements[0] = n1 * n2;
    this->elements[1] = n2 * n3;
    this->elements[2] = n3 * n1;

    for (int i = 0; i < 3; i++)
        this->uvw[i] = std::vector<int>(rank * elements[i], 0);
}

bool Scheme::validate() const {
    bool valid = true;

    for (int i = 0; i < elements[0] && valid; i++)
        for (int j = 0; j < elements[1] && valid; j++)
            for (int k = 0; k < elements[2] && valid; k++)
                valid &= validateEquation(i, j, k);

    return valid;
}

bool Scheme::read(std::istream &is, bool check) {
    for (int i = 0; i < 3; i++)
        for (int index = 0; index < rank; index++)
            for (int j = 0; j < elements[i]; j++)
                is >> uvw[i][index * elements[i] + j];

    return !check || validate();
}

bool Scheme::validateEquation(int i, int j, int k) const {
    int i1 = i / dimension[1];
    int i2 = i % dimension[1];
    int j1 = j / dimension[2];
    int j2 = j % dimension[2];
    int k1 = k / dimension[0];
    int k2 = k % dimension[0];

    int target = (i2 == j1) && (i1 == k2) && (j2 == k1);
    int equation = 0;

    for (int index = 0; index < rank; index++)
        equation += uvw[0][index * elements[0] + i] * uvw[1][index * elements[1] + j] * uvw[2][index * elements[2] + k];

    return equation == target;
}
