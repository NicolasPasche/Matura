#include "pricers/ann_pricer.h"
#include "models/merton_jump_diffusion_model.h"
#include "derivatives/call_option.h"
#include "utils/random_generator.h"

#include <iostream>

int main() {

    MertonJumpDiffusionModel model(
        100.0,    // S0
        0.05,     // r
        0.04,     // d
        0.275,    // sigma
        0.75,     // lambda
        -0.05,    // muJ
        0.275     // sigmaJ
    );

    CallOption option(100.0);

    RandomGenerator rng(42);

    ANN_Pricer pricer(
        "../../models/best_model_dividend.pt"
    );

    auto result = pricer.price(
        model,
        option,
        rng,
        0.05,
        1.0,
        0
    );

    std::cout << "ANN price: "
              << result.price
              << std::endl;

    return 0;
}