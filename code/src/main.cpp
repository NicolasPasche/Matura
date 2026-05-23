#include "models/black_scholes_model.h"
#include "derivatives/call_option.h"
#include "pricers/monte_carlo_pricer.h"
#include "utils/random_generator.h"

# include <iostream>

int main() {
    // Parameters
    double S0 = 100.0; 
    double r = 0.05; 
    double d = 0.0; 
    double sigma = 0.2; 
    double T = 1.0; 
    int num_simulations = 100000;

    // Create model, derivatives, pricer, and random generator
    BlackScholesModel model(S0, r, d, sigma);
    CallOption call_option(100.0);
    
    MonteCarloPricer pricer;
    RandomGenerator rng(1);

    // Price derivatives
    for (int i = 0; i < 5; ++i) {
        num_simulations *=2;
        auto call_result = pricer.price(model, call_option, rng, r, T, num_simulations);
        std::cout << "Simulations: " << num_simulations << " - " << call_option.getName() << ": " << call_result.price << " (SE: " << call_result.standard_error << ")\n";

    }

    return 0;
}