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

        // First I gave default values, but like this it's more flexible to build new stuff
        // on top of it later...
        double muJ; // mean of jump size (in log) standart assumption for mjd (try around later for fun)
        double sigmaJ; // std of jump size (in log)

    public:
        MertonJumpDiffusionModel(double S0_, double r_, double d_, double sigma_, double lambda_, double muJ_, double sigmaJ_)
            : S0(S0_), r(r_), d(d_), sigma(sigma_), lambda(lambda_), muJ(muJ_), sigmaJ(sigmaJ_) {}

        // Not sure if I'll need the jump parameters. Maybe for semi analytical approach later?
        double getS() const override {return S0;}
        double getR() const override {return r;}
        double getD() const override {return d;}
        double getSigma() const override {return sigma;}
        double getLambda() const override {return lambda;}
        double getMuJ() const override {return muJ;}
        double getSigmaJ() const override {return sigmaJ;}

        // In this case, seperate expressions, because in my oneliner, I had some mistakes...
        double simulate_terminal_price(RandomGenerator& rng, double T) const override {
            // WARNING FOR FUTURE USERS!!!
            // This random draw only works for terminal lognormal models, nto for pathwise models
            // e.g. Barrier Options -- I'll maybe customize it sometime, but not now.
            double Z = rng.get_normal();
            int N = rng.get_poisson(lambda, T);

            double k = std::exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0;

            // My drift was wrong in earlier versions, so I built it in seperate steps
            double drift = (r - d - lambda * k - 0.5 * sigma * sigma) * T;
            double diffusion = sigma * std::sqrt(T) * Z;

            double jump_sum = 0.0;

            for (int i = 0; i < N; ++i) {
                double J = rng.get_normal(); 
                double log_jump = muJ + sigmaJ * J;
                jump_sum += log_jump;
            }

            // FINALLY (._.)
            return S0 * std::exp(drift + diffusion + jump_sum);

        }
};