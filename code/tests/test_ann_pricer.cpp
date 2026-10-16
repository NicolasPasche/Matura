#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <exception>

#include <torch/torch.h>

#include "models/merton_jump_diffusion_model.h"
#include "derivatives/call_option.h"
#include "pricers/ann_pricer.h"
#include "utils/random_generator.h"


bool test_prediction(
    const ANN_Pricer& pricer,
    RandomGenerator& rng,
    double S0,
    double K,
    double T,
    double r,
    double d,
    double sigma,
    double lambda,
    double muJ,
    double sigmaJ
) {
    MertonJumpDiffusionModel model(S0, r, d, sigma, lambda, muJ, sigmaJ);
    CallOption call(K);

    // I want to make sure identical inputs always produce identical predictions.
    const double price1 = pricer.price(model, call, rng, r, T, 0).price;
    const double price2 = pricer.price(model, call, rng, r, T, 0).price;

    const bool finite = std::isfinite(price1);

    const bool reproducible =
        finite && std::isfinite(price2) &&
        std::abs(price1 - price2) < 1e-5;

    // A European call must stay between zero and the discounted stock price.
    const double upper_bound = S0 * std::exp(-d * T);

    const bool valid_range =
        finite && price1 >= 0.0 && price1 <= upper_bound;

    const bool passed = finite && reproducible && valid_range;

    std::cout
        << (passed ? "[PASS] " : "[FAIL] ")
        << "S0=" << std::setw(6) << S0
        << " K=" << std::setw(6) << K
        << " T=" << std::setw(5) << T
        << " Price=" << price1 << '\n';

    if (!finite)
        std::cout << "       Prediction is not finite.\n";

    if (!reproducible)
        std::cout << "       Repeated predictions differ.\n";

    if (!valid_range)
        std::cout << "       Prediction violates call price bounds.\n";

    return passed;
}


int main(int argc, char* argv[]) {

    // I want to switch between the two ANNs without recompiling.
    if (argc < 2 || argc > 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <path_to_model.pt> [analytical]\n";
        return 1;
    }

    const std::string model_path = argv[1];

    const bool analytical =
        argc == 3 && std::string(argv[2]) == "analytical";

    if (argc == 3 && !analytical) {
        std::cerr << "Unknown model type: " << argv[2] << '\n';
        return 1;
    }

    // Both networks share the input scaler, but not the output scaler.
    const double y_mean = analytical
        ? 22.223006564152026
        : 22.22354624;

    const double y_scale = analytical
        ? 22.047419209093864
        : 22.05475308;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "ANN Pricer Test\n\n";

    std::cout
        << "CUDA available: "
        << (torch::cuda::is_available() ? "Yes" : "No") << '\n';

    std::cout
        << "Selected device: "
        << (torch::cuda::is_available() ? "CUDA" : "CPU") << '\n';

    std::cout
        << "ANN type: "
        << (analytical ? "Analytical" : "Monte Carlo") << '\n';

    std::cout << "Model: " << model_path << "\n\n";

    try {
        ANN_Pricer pricer(model_path, y_mean, y_scale);

        std::cout << "Model loaded successfully.\n\n";

        RandomGenerator rng(1);

        const double r = 0.05;
        const double d = 0.02;
        const double sigma = 0.20;
        const double lambda = 0.40;
        const double muJ = -0.08;
        const double sigmaJ = 0.15;

        const double S0 = 100.0;
        const double T = 1.0;

        int passed = 0;
        const int total = 3;

        // Different strikes should produce different option prices.
        passed += test_prediction(
            pricer, rng, S0, 80.0, T, r, d,
            sigma, lambda, muJ, sigmaJ
        );

        passed += test_prediction(
            pricer, rng, S0, 100.0, T, r, d,
            sigma, lambda, muJ, sigmaJ
        );

        passed += test_prediction(
            pricer, rng, S0, 120.0, T, r, d,
            sigma, lambda, muJ, sigmaJ
        );

        std::cout << "\n-----------------------------\n";
        std::cout << "Passed: " << passed << " / " << total << '\n';

        if (passed == total) {
            std::cout << "All ANN tests passed.\n";
            return 0;
        }

        std::cout << "Some ANN tests failed.\n";
        return 1;

    } catch (const std::exception& e) {
        std::cerr << "\nANN test failed:\n" << e.what() << '\n';
        return 1;
    }
}