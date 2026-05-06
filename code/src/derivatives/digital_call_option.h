#pragma once

#include "derivative.h"

#include <algorithm>


class DigitalCallOption : public Derivative {
    private:
        double K;

    public: 
        explicit DigitalCallOption(double K_) : K(K_) {}

        double payoff(double ST) const override  {
            return ST > K ? 1.0 : 0.0;
        }

        std::string getName() const override {
            return "Digital Call Option";
        }
};