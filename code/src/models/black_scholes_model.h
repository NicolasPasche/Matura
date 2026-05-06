#pragma once

#include "model.h"

#include <cmath>

class BlackScholesModel : public Model {
    private:
        double S0; 
        double r; 
        double d; 
        double sigma;

    public:
    BlackScholesModel(double S0_, double r_, double d_, double sigma_) 
        : S0(S0_), r(r_), d(d_), sigma(sigma_) {}

    double simulate_terminal_price(double S0, double T, RandomGenerator& rng) const override {
        double Z = rng.get_normal_vector()[0];

        return S0 * std::exp((r - d - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
    }
};
