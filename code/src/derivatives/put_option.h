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

        double getStrike() const override {
            return K;
        }

        std::string getName() const override {
            return "Put Option";
        }

        std::string type() const override {
            return "Put";
        }
};