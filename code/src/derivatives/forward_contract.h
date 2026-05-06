#pragma once

#include "derivative.h"

#include <algorithm>


class ForwardContract : public Derivative {
    private:
        double K;

    public: 
        explicit ForwardContract(double K_) : K(K_) {}

        double payoff(double ST) const override  {
            return ST - K;
        }

        std::string getName() const override {
            return "Forward Contract";
        }
};