#pragma once

#include "derivative.h"

#include <algorithm>


class PutOption : public Derivative {
    private:
        double K;

    public: 
        explicit PutOption(double K_) : K(K_) {}

        double payoff(double ST) const override  {
            return std::max(K - ST, 0.0);
        }

        std::string getName() const override {
            return "Put Option";
        }
};