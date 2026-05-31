#pragma once

#include "derivative.h"

#include <algorithm>


class CallOption : public Derivative {
    private:
        double K;

    public: 
        explicit CallOption(double K_) : K(K_) {}

        double payoff(double ST) const override  {
            return std::max(ST - K, 0.0);
        }

        double getStrike() const override {
            return K;
        }

        std::string getName() const override {
            return "Call Option";
        }

        std::string type() const override {
            return "Call";
        }
};