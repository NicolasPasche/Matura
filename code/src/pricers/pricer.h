#pragma once

#include "../models/model.h"
#include "../derivatives/derivative.h"
#include "../utils/random_generator.h"

class Pricer {
    public:
        struct Result {
            double price;
            double standard_error;
        };

        virtual ~Pricer() = default;

        // Virtual function to price a derivative given a model and a random generator
        virtual Result price(
            const Model& model, 
            const Derivative& derivative, 
            RandomGenerator& rng, 
            double r, 
            double T, 
            int num_simulations
        ) const = 0;
};