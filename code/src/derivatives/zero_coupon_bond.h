#pragma once

#include "derivative.h"

#include <algorithm>


class ZeroCouponBond : public Derivative {
    public: 
        double payoff(double ST) const override  {
            return 1.0;;
        }

        std::string getName() const override {
            return "Zero Coupon Bond";
        }
};