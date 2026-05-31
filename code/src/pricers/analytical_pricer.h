#pragma once

#include "../models/black_scholes_model.h"
#include "../derivatives/derivative.h"

class AnalyticalPricer{
    public:
        double price(
            const BlackScholesModel& model, 
            const Derivative& derivative, 
            double T
        ) const;
};
