#include "random_generator.h"

#include <random>

RandomGenerator::RandomGenerator(int dim) 
    : generator(std::random_device{}()), 
      distribution(0.0, 1.0), 
      dimension(dim) 
{}

std::vector<double> RandomGenerator::get_normal_vector() {
    std::vector<double> result(dimension);

    for (int i = 0; i < dimension; ++i) {
        result[i] = distribution(generator);
    }
    return result;
};

double RandomGenerator::get_normal() {
    return distribution(generator);
}

int RandomGenerator::get_poisson(double lambda, double T) {
    std::poisson_distribution<int> dist(lambda * T);
    return dist(generator);
}
