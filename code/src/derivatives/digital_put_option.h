#pragma once

#include "derivative.h"

#include <algorithm>


class DigitalPutOption : public Derivative {
    private:
        double K;

    public: 
        explicit DigitalPutOption(double K_) : K(K_) {}

        double payoff(double ST) const override  {
            return ST < K ? 1.0 : 0.0;
        }

        double getStrike() const override {
            return K;
        }

        std::string getName() const override {
            return "Digital Put Option";
        }

        std::string type() const override {
            return "DigitalPut";
        }
};