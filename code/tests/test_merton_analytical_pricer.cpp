#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

#include "models/black_scholes_model.h"
#include "models/merton_jump_diffusion_model.h"

#include "derivatives/call_option.h"

#include "pricers/analytical_pricer.h"
#include "pricers/merton_analytical_pricer.h"
#include "pricers/monte_carlo_pricer.h"

#include "utils/random_generator.h"


bool check(
    const std::string& name,
    double actual,
    double expected,
    double tolerance
) {
    const bool passed =
        std::isfinite(actual) &&
        std::abs(actual - expected) <= tolerance;

    std::cout
        << (passed ? "[PASS] " : "[FAIL] ")
        << std::left << std::setw(30) << name
        << "Actual: " << std::setw(15) << actual
        << "Expected: " << expected
        << '\n';

    return passed;
}


int main() {

    const double S0 = 100.0;
    const double K = 100.0;
    const double T = 1.0;

    const double r = 0.05;
    const double d = 0.02;
    const double sigma = 0.20;

    const double lambda = 0.40;
    const double muJ = -0.08;
    const double sigmaJ = 0.15;

    MertonAnalyticalPricer merton_pricer;
    AnalyticalPricer bs_pricer;
    MonteCarloPricer mc_pricer;

    CallOption call(K);

    MertonJumpDiffusionModel merton_model(
        S0, r, d, sigma, lambda, muJ, sigmaJ
    );

    BlackScholesModel bs_model(
        S0, r, d, sigma
    );

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Merton Analytical Pricer Tests\n\n";

    int passed = 0;
    int total = 0;


    // Compare against a separately calculated Merton series value.
    const double merton_price =
        merton_pricer.price(merton_model, call, r, T);

    total++;
    passed += check(
        "Merton Reference Price",
        merton_price,
        10.09899938344718,
        1e-8
    );


    // Merton must reduce to Black-Scholes when jumps are disabled.
    MertonJumpDiffusionModel no_jump_model(
        S0, r, d, sigma, 0.0, muJ, sigmaJ
    );

    const double no_jump_price =
        merton_pricer.price(no_jump_model, call, r, T);

    const double bs_price =
        bs_pricer.price(bs_model, call, T);

    total++;
    passed += check(
        "No Jumps: Merton vs BS",
        no_jump_price,
        bs_price,
        1e-9
    );


    // Jumps with zero size must not change the stock price.
    MertonJumpDiffusionModel zero_jump_size_model(
        S0, r, d, sigma, lambda, 0.0, 0.0
    );

    total++;
    passed += check(
        "Zero Jump Size",
        merton_pricer.price(zero_jump_size_model, call, r, T),
        bs_price,
        1e-9
    );


    // At maturity, the analytical price must equal the payoff.
    total++;
    passed += check(
        "At Maturity",
        merton_pricer.price(merton_model, call, r, 0.0),
        call.payoff(S0),
        1e-12
    );


    // Without jumps or volatility, the final stock price is deterministic.
    MertonJumpDiffusionModel deterministic_model(
        S0, r, d, 0.0, 0.0, muJ, sigmaJ
    );

    const double ST =
        S0 * std::exp((r - d) * T);

    const double expected_deterministic =
        std::exp(-r * T) * call.payoff(ST);

    total++;
    passed += check(
        "Deterministic Stock",
        merton_pricer.price(deterministic_model, call, r, T),
        expected_deterministic,
        1e-9
    );


    // Calls with higher strikes must not be more expensive.
    CallOption low_strike(80.0);
    CallOption high_strike(120.0);

    const double low_price =
        merton_pricer.price(merton_model, low_strike, r, T);

    const double high_price =
        merton_pricer.price(merton_model, high_strike, r, T);

    const bool monotonic =
        low_price >= merton_price &&
        merton_price >= high_price;

    total++;
    passed += monotonic;

    std::cout
        << (monotonic ? "[PASS] " : "[FAIL] ")
        << "Call Price Monotonicity\n";


    // Monte Carlo provides an independent check of the jump simulation.
    // Its result is random, so the allowed difference depends on its error.
    RandomGenerator rng(1);

    const int simulations = 200000;

    const Pricer::Result mc_result =
        mc_pricer.price(
            merton_model,
            call,
            rng,
            r,
            T,
            simulations
        );

    const double mc_tolerance =
        4.0 * mc_result.standard_error + 0.01;

    total++;
    passed += check(
        "Merton vs Monte Carlo",
        mc_result.price,
        merton_price,
        mc_tolerance
    );

    std::cout
        << "\nMonte Carlo standard error: "
        << mc_result.standard_error << '\n';

    std::cout
        << "Monte Carlo tolerance: "
        << mc_tolerance << '\n';


    std::cout << "\n----------------------------------------\n";
    std::cout << "Passed: " << passed << " / " << total << '\n';

    if (passed == total) {
        std::cout << "All Merton tests passed.\n";
        return 0;
    }

    std::cout << "Some Merton tests failed.\n";
    return 1;
}