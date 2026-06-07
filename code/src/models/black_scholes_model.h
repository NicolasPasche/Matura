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

    // Getter functions for model parameters (needed for analytical pricer)
    double getS() const override {return S0;}
    double getR() const override {return r;}
    double getD() const override {return d;}
    double getSigma() const override {return sigma;}

    double simulate_terminal_price(RandomGenerator& rng, double T) const override {
        double Z = rng.get_normal();

        return S0 * std::exp((r - d - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
    }
};
