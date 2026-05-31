#pragma once

#include <string>

class Derivative {
    public:
        virtual ~Derivative() = default;

        // Virtual function to calculate the payoff of the derivative given the underlying asset price
        virtual double payoff(double ST) const = 0;

        // Virtual function to get the strike price of the derivative (if applicable)
        virtual double getStrike() const { return 0.0; }

        // Virtual function to get the name of the derivative
        virtual std::string getName() const = 0;

        // Virtual function to get the type of the derivative (for pricer selection)
        virtual std::string type() const = 0;

};