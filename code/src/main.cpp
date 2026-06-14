#include "models/black_scholes_model.h"
#include "models/merton_jump_diffusion_model.h"

#include "utils/random_generator.h"

#include <iostream>
#include <cmath>

int main() {

    // Parameters
    double S0 = 100.0;
    double r = 0.05;
    double d = 0.00;
    double sigma = 0.20;
    double T = 1.0;

    int N = 1000000;

    RandomGenerator rng(42);

    // =====================================================
    // TEST 1: lambda = 0  --> MJD should equal BS
    // =====================================================

    BlackScholesModel bs_model(S0, r, d, sigma);

    MertonJumpDiffusionModel mjd_model(
        S0,
        r,
        d,
        sigma,
        0.0,   // lambda
        0.0,   // muJ
        0.2    // sigmaJ
    );

    double bs_sum = 0.0;
    double mjd_sum = 0.0;

    for (int i = 0; i < N; ++i) {
        bs_sum += bs_model.simulate_terminal_price(rng, T);
        mjd_sum += mjd_model.simulate_terminal_price(rng, T);
    }

    double bs_mean = bs_sum / N;
    double mjd_mean = mjd_sum / N;

    std::cout << "=========================\n";
    std::cout << "TEST 1: lambda = 0\n";
    std::cout << "=========================\n";

    std::cout << "BS mean ST  = " << bs_mean << '\n';
    std::cout << "MJD mean ST = " << mjd_mean << '\n';
    std::cout << "Difference  = " << std::abs(bs_mean - mjd_mean) << "\n\n";


    // =====================================================
    // TEST 2: Martingale property
    // =====================================================

    double lambda = 1.0;

    MertonJumpDiffusionModel mjd_model2(
        S0,
        r,
        d,
        sigma,
        lambda,
        0.0,   // muJ
        0.2    // sigmaJ
    );

    double st_sum = 0.0;

    for (int i = 0; i < N; ++i) {
        st_sum += mjd_model2.simulate_terminal_price(rng, T);
    }

    double simulated_mean = st_sum / N;
    double theoretical_mean = S0 * std::exp((r - d) * T);

    std::cout << "=========================\n";
    std::cout << "TEST 2: Martingale\n";
    std::cout << "=========================\n";

    std::cout << "Simulated E[ST]  = " << simulated_mean << '\n';
    std::cout << "Theoretical E[ST] = " << theoretical_mean << '\n';
    std::cout << "Difference        = "
              << std::abs(simulated_mean - theoretical_mean)
              << '\n';

    return 0;
}