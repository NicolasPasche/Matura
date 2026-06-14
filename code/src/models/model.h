#pragma once

#include "../utils/random_generator.h"

class Model {
    public:
        virtual ~Model() = default;

        // Virtual getter functions for model parameters
        virtual double getS() const = 0;
        virtual double getR() const = 0;
        virtual double getD() const = 0;
        virtual double getSigma() const = 0;
        virtual double getLambda() const {return 0.0;} // Just for merton model so far
        virtual double getMuJ() const {return 0.0;} // Just for merton model so far
        virtual double getSigmaJ() const {return 0.0;} // Just for merton model so far

        // Virtual function to simulate the underlying asset price
        virtual double simulate_terminal_price(RandomGenerator& rng, double T) const = 0;
};