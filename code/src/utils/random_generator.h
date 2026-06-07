#pragma once

#include <vector>
#include <random>

class RandomGenerator {
    private:
        std::mt19937 generator;
        std::normal_distribution<double> distribution;

        int dimension;
    
    public:

        RandomGenerator(int dim);

        std::vector<double> get_normal_vector();
        double get_normal();

        std::vector<int> get_poisson_vector(double lambda, double T);
        int get_poisson(double lambda, double T);
};