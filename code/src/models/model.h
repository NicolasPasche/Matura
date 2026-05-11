#pragma once

#include "../utils/random_generator.h"

class Model {
    public:
        virtual ~Model() = default;

        // Virtual function to simulate the underlying asset price
        virtual double simulate_terminal_price(RandomGenerator& rng, double T) const = 0;
};