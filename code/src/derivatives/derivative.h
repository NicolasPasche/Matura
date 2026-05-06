#pragma once

#include <string>

class Derivative {
    public:
        virtual ~Derivative() = default;

        // Virtual function to calculate the payoff of the derivative given the underlying asset price
        virtual double payoff(double ST) const = 0;

        // Virtual function to get the name of the derivative
        virtual std::string getName() const = 0;

};