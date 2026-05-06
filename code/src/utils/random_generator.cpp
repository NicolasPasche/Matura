#include "random_generator.h"

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