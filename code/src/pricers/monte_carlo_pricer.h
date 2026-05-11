#pragma once

#include "pricer.h"

class MonteCarloPricer : public Pricer {
    public:
        Result price(
            const Model& model, 
            const Derivative& derivative, 
            RandomGenerator& rng, 
            double r, 
            double T, 
            int num_simulations
        ) const override {
            
            double sum = 0.0;
            double sum_sq = 0.0;

            for (int i = 0; i < num_simulations; ++i) {

                double ST = model.simulate_terminal_price(rng, T);
                double payoff = derivative.payoff(ST);
                
                double discounted = std::exp(-r * T) * payoff;

                sum += discounted;
                sum_sq += discounted * discounted;
            }

            double mean = sum / num_simulations;
            double variance = (sum_sq / num_simulations) - (mean * mean);
            if (variance < 0) variance = 0;
            double standard_error = std::sqrt(variance / num_simulations);

            return {mean, standard_error};

        }
};