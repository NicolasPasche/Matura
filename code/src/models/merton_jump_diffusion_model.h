#pragma once

#include "model.h"

#include <cmath>

class MertonJumpDiffusionModel : public Model {
    private:
        double S0;
        double r;
        double d;
        double sigma;
        double lambda;

        double muJ = 0.0; // mean of jump size (in log) standart assumption for mjd (try around later for fun)
        double sigmaJ = 0.2; // std of jump size (in log)

    public:
        MertonJumpDiffusionModel(double S0_, double r_, double d_, double sigma_, double lambda_)
            : S0(S0_), r(r_), d(d_), sigma(sigma_), lambda(lambda_) {}

        // Not sure if i'll need these. Maybe if i'll build a semi-analytical pricer
        double getS() const override {return S0;}
        double getR() const override {return r;}
        double getD() const override {return d;}
        double getSigma() const override {return sigma;}
        double getLambda() const override {return lambda;}

        double simulate_terminal_price(RandomGenerator& rng, double T) const override {
            double Z = rng.get_normal();
            int N = rng.get_poisson(lambda, T);

            double k = std::exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0;

            double drift = (r - d - lambda * k - 0.5 * sigma * sigma) * T;
            double diffusion = sigma * std::sqrt(T) * Z;

            double jump_sum = 0.0;

            for (int i = 0; i < N; ++i) {
                double J = rng.get_normal(); 
                double log_jump = muJ + sigmaJ * J;
                jump_sum += log_jump;
            }

            return S0 * std::exp(drift + diffusion + jump_sum);

        }
};