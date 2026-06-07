#include "models/black_scholes_model.h"
#include "models/merton_jump_diffusion_model.h"

#include "derivatives/call_option.h"
#include "derivatives/put_option.h"
#include "derivatives/digital_call_option.h"
#include "derivatives/digital_put_option.h"
#include "derivatives/forward_contract.h"
#include "derivatives/zero_coupon_bond.h"

#include "pricers/analytical_pricer.h"
#include "pricers/monte_carlo_pricer.h"

#include "utils/random_generator.h"

#include <iostream>
#include <typeinfo>

int main() {

    // Model parameters
    double S0 = 100.0;
    double r = 0.05;
    double d = 0.00;
    double sigma = 0.20;
    double T = 1.0;
    double lambda = 0.0;

    int num_simulations = 1000000;

    // Model
    BlackScholesModel bs_model(S0, r, d, sigma);
    MertonJumpDiffusionModel mjd_model(S0, r, d, sigma, lambda);

    // RNG
    RandomGenerator rng(42);

    // Pricers
    AnalyticalPricer analytical_pricer;
    MonteCarloPricer monte_carlo_pricer;

    // Derivatives
    CallOption call(100.0);
    PutOption put(100.0);

    DigitalCallOption digital_call(100.0);
    DigitalPutOption digital_put(100.0);

    ForwardContract forward(100.0);

    ZeroCouponBond zcb;

    
    double call_price_analytical = analytical_pricer.price(bs_model, call, T);
    auto call_price_monte_carlo = monte_carlo_pricer.price(bs_model, call, rng, r, T, num_simulations);
    auto call_price_mjd_monte_carlo = monte_carlo_pricer.price(mjd_model, call, rng, r, T, num_simulations);
    
    double put_price_analytical = analytical_pricer.price(bs_model, put, T);
    auto put_price_monte_carlo = monte_carlo_pricer.price(bs_model, put, rng, r, T, num_simulations);
    auto put_price_mjd_monte_carlo = monte_carlo_pricer.price(mjd_model, put, rng, r, T, num_simulations);

    double digital_call_price_analytical = analytical_pricer.price(bs_model, digital_call, T);
    auto digital_call_price_monte_carlo = monte_carlo_pricer.price(bs_model, digital_call, rng, r, T, num_simulations);
    auto digital_call_price_mjd_monte_carlo = monte_carlo_pricer.price(mjd_model, digital_call, rng, r, T, num_simulations);

    double digital_put_price_analytical = analytical_pricer.price(bs_model, digital_put, T);
    auto digital_put_price_monte_carlo = monte_carlo_pricer.price(bs_model, digital_put, rng, r, T, num_simulations);
    auto digital_put_price_mjd_monte_carlo = monte_carlo_pricer.price(mjd_model, digital_put, rng, r, T, num_simulations);

    double forward_price_analytical = analytical_pricer.price(bs_model, forward, T);
    auto forward_price_monte_carlo = monte_carlo_pricer.price(bs_model, forward, rng, r, T, num_simulations);
    auto forward_price_mjd_monte_carlo = monte_carlo_pricer.price(mjd_model, forward, rng, r, T, num_simulations);

    double zcb_price_analytical = analytical_pricer.price(bs_model, zcb, T);
    auto zcb_price_monte_carlo = monte_carlo_pricer.price(bs_model, zcb, rng, r, T, num_simulations);
    auto zcb_price_mjd_monte_carlo = monte_carlo_pricer.price(mjd_model, zcb, rng, r, T, num_simulations);

    
    std::cout << "\n=== RESULTS ===\n\n";

    std::cout << "Call Option\n";
    std::cout << "  Analytical BS: " << call_price_analytical << '\n';
    std::cout << "  Monte Carlo BS: " << call_price_monte_carlo.price
            << " (SE = " << call_price_monte_carlo.standard_error << ")\n";
    std::cout << "  Monte Carlo MJD: " << call_price_mjd_monte_carlo.price
            << " (SE = " << call_price_mjd_monte_carlo.standard_error << ")\n\n";

    std::cout << "Put Option\n";
    std::cout << "  Analytical BS: " << put_price_analytical << '\n';
    std::cout << "  Monte Carlo BS: " << put_price_monte_carlo.price
            << " (SE = " << put_price_monte_carlo.standard_error << ")\n";
    std::cout << "  Monte Carlo MJD: " << put_price_mjd_monte_carlo.price
            << " (SE = " << put_price_mjd_monte_carlo.standard_error << ")\n\n";

    std::cout << "Digital Call Option\n";
    std::cout << "  Analytical BS: " << digital_call_price_analytical << '\n';
    std::cout << "  Monte Carlo BS: " << digital_call_price_monte_carlo.price
            << " (SE = " << digital_call_price_monte_carlo.standard_error << ")\n";
    std::cout << "  Monte Carlo MJD: " << digital_call_price_mjd_monte_carlo.price
            << " (SE = " << digital_call_price_mjd_monte_carlo.standard_error << ")\n\n";

    std::cout << "Digital Put Option\n";
    std::cout << "  Analytical BS: " << digital_put_price_analytical << '\n';
    std::cout << "  Monte Carlo BS: " << digital_put_price_monte_carlo.price
            << " (SE = " << digital_put_price_monte_carlo.standard_error << ")\n";
    std::cout << "  Monte Carlo MJD: " << digital_put_price_mjd_monte_carlo.price
            << " (SE = " << digital_put_price_mjd_monte_carlo.standard_error << ")\n\n";

    std::cout << "Forward Contract\n";
    std::cout << "  Analytical BS: " << forward_price_analytical << '\n';
    std::cout << "  Monte Carlo BS: " << forward_price_monte_carlo.price
            << " (SE = " << forward_price_monte_carlo.standard_error << ")\n";
    std::cout << "  Monte Carlo MJD: " << forward_price_mjd_monte_carlo.price
            << " (SE = " << forward_price_mjd_monte_carlo.standard_error << ")\n\n";

    std::cout << "Zero Coupon Bond\n";
    std::cout << "  Analytical BS: " << zcb_price_analytical << '\n';
    std::cout << "  Monte Carlo BS: " << zcb_price_monte_carlo.price
     << " (SE = " << zcb_price_monte_carlo.standard_error << ")\n";
    std::cout << "  Monte Carlo MJD: " << zcb_price_mjd_monte_carlo.price
     << " (SE = " << zcb_price_mjd_monte_carlo.standard_error << ")\n";

    return 0;
}