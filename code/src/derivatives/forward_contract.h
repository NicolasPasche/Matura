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
        
        double getStrike() const override {
            return K;
        }

        std::string getName() const override {
            return "Forward Contract";
        }

        std::string type() const override {
            return "Forward";
        }
};