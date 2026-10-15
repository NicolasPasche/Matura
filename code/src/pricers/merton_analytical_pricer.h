#pragma once

#include "../models/merton_jump_diffusion_model.h"
#include "../derivatives/call_option.h"

class MertonAnalyticalPricer {
    public:
        double price(
            const MertonJumpDiffusionModel& model, 
            const CallOption& call, 
            double r, 
            double T
        ) const;
};