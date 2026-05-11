#include "models/black_scholes_model.h"
#include "derivatives/call_option.h"
#include "derivatives/put_option.h"
#include "derivatives/digital_call_option.h"
#include "derivatives/digital_put_option.h"
#include "derivatives/forward_contract.h"
#include "derivatives/zero_coupon_bond.h"
#include "pricers/monte_carlo_pricer.h"
#include "utils/random_generator.h"

# include <iostream>

int main() {
    // Parameters
    double S0 = 100.0; 
    double r = 0.05; 
    double d = 0.2; 
    double sigma = 0.2; 
    double T = 1.0; 
    int num_simulations = 100000;

    // Create model, derivatives, pricer, and random generator
    BlackScholesModel model(S0, r, d, sigma);
    CallOption call_option(100.0);
    PutOption put_option(100.0);
    DigitalCallOption digital_call_option(100.0);
    DigitalPutOption digital_put_option(100.0);
    ForwardContract forward_contract(100.0);
    ZeroCouponBond zero_coupon_bond;
    
    MonteCarloPricer pricer;
    RandomGenerator rng(1);

    // Price derivatives
    auto call_result = pricer.price(model, call_option, rng, r, T, num_simulations);
    auto put_result = pricer.price(model, put_option, rng, r, T, num_simulations);
    auto digital_call_result = pricer.price(model, digital_call_option, rng, r, T, num_simulations);
    auto digital_put_result = pricer.price(model, digital_put_option, rng, r, T, num_simulations);
    auto forward_contract_result = pricer.price(model, forward_contract, rng, r, T, num_simulations);
    auto zero_coupon_bond_result = pricer.price(model, zero_coupon_bond, rng, r, T, num_simulations);

    // Output results
    std::cout << call_option.getName() << ": " << call_result.price << " (SE: " << call_result.standard_error << ")\n";
    std::cout << put_option.getName() << ": " << put_result.price << " (SE: " << put_result.standard_error << ")\n";
    std::cout << digital_call_option.getName() << ": " << digital_call_result.price << " (SE: " << digital_call_result.standard_error << ")\n";
    std::cout << digital_put_option.getName() << ": " << digital_put_result.price << " (SE: " << digital_put_result.standard_error << ")\n";
    std::cout << forward_contract.getName() << ": " << forward_contract_result.price << " (SE: " << forward_contract_result.standard_error << ")\n";
    std::cout << zero_coupon_bond.getName() << ": " << zero_coupon_bond_result.price << " (SE: " << zero_coupon_bond_result.standard_error << ")\n";

    return 0;
}