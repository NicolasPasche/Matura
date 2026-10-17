#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <exception>

#include "models/black_scholes_model.h"
#include "models/merton_jump_diffusion_model.h"

#include "derivatives/call_option.h"

#include "pricers/analytical_pricer.h"
#include "pricers/merton_analytical_pricer.h"
#include "pricers/monte_carlo_pricer.h"
#include "pricers/ann_pricer.h"

#include "utils/random_generator.h"

void print_banner() {
    std::cout << R"(
========================================================================

 ██████╗ ███████╗██████╗ ██╗██╗   ██╗ █████╗ ████████╗██╗██╗   ██╗███████╗███████╗
 ██╔══██╗██╔════╝██╔══██╗██║██║   ██║██╔══██╗╚══██╔══╝██║██║   ██║██╔════╝██╔════╝
 ██║  ██║█████╗  ██████╔╝██║██║   ██║███████║   ██║   ██║██║   ██║█████╗  ███████╗
 ██║  ██║██╔══╝  ██╔══██╗██║╚██╗ ██╔╝██╔══██║   ██║   ██║╚██╗ ██╔╝██╔══╝  ╚════██║
 ██████╔╝███████╗██║  ██║██║ ╚████╔╝ ██║  ██║   ██║   ██║ ╚████╔╝ ███████╗███████║
 ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝╚══════╝

                       P R I C I N G   E N G I N E

========================================================================

                 BLACK-SCHOLES  |  MERTON JUMP DIFFUSION
                  MONTE CARLO   |  NEURAL NETWORK PRICING

------------------------------------------------------------------------
                        MATURA THESIS PROJECT
                               2026
                          Nicolas Pasche
------------------------------------------------------------------------

 I do not take any responsibility for you losing money with this software.

========================================================================

)" << std::flush;
}

int main(int argc, char* argv[]) {

    print_banner();

    try {

        // Option parameters
        const double S0 = 100.0;
        const double K = 100.0;
        const double T = 1.0;
        const double r = 0.05;
        const double d = 0.02;

        // Merton Jump Diffusion parameters
        const double sigma = 0.20;
        const double lambda = 0.50;
        const double muJ = -0.10;
        const double sigmaJ = 0.20;

        constexpr int MC_PATHS = 5000;

        // The TorchScript model can also be passed as an argument.
        const std::string model_path = argc > 1
            ? argv[1]
            : "../models/best_model_analytical.pt";

        if (!std::filesystem::exists(model_path)) {
            std::cerr
                << "ANN model not found: "
                << model_path << '\n';

            return 1;
        }


        // Models
        BlackScholesModel bs_model(
            S0, r, d, sigma
        );

        MertonJumpDiffusionModel mjd_model(
            S0, r, d, sigma,
            lambda, muJ, sigmaJ
        );


        // Derivative
        CallOption call(K);


        // Pricers
        AnalyticalPricer bs_pricer;
        MertonAnalyticalPricer mjd_pricer;
        MonteCarloPricer mc_pricer;

        ANN_Pricer ann_pricer(
            model_path,
            22.223006564152026,
            22.047419209093864
        );

        RandomGenerator rng(1);


        // Calculate option prices
        const double bs_price = bs_pricer.price(
            bs_model,
            call,
            T
        );

        const double mjd_price = mjd_pricer.price(
            mjd_model,
            call,
            r,
            T
        );

        const auto mc_result = mc_pricer.price(
            mjd_model,
            call,
            rng,
            r,
            T,
            MC_PATHS
        );

        const auto ann_result = ann_pricer.price(
            mjd_model,
            call,
            rng,
            r,
            T,
            0
        );


        // Display results
        std::cout << std::fixed << std::setprecision(6);

        std::cout
            << "\n============================================\n"
            << "         DERIVATIVES PRICING ENGINE\n"
            << "============================================\n\n";

        std::cout << "European Call Option\n\n";

        std::cout
            << "S0       = " << S0 << '\n'
            << "K        = " << K << '\n'
            << "T        = " << T << '\n'
            << "r        = " << r << '\n'
            << "d        = " << d << '\n'
            << "sigma    = " << sigma << '\n'
            << "lambda   = " << lambda << '\n'
            << "muJ      = " << muJ << '\n'
            << "sigmaJ   = " << sigmaJ << '\n';

        std::cout
            << "\n--------------------------------------------\n"
            << std::left << std::setw(26) << "Method"
            << std::right << std::setw(16) << "Price"
            << "\n--------------------------------------------\n";

        std::cout
            << std::left << std::setw(26) << "Black-Scholes"
            << std::right << std::setw(16) << bs_price << '\n'

            << std::left << std::setw(26) << "Merton Analytical"
            << std::right << std::setw(16) << mjd_price << '\n'

            << std::left << std::setw(26) << "Monte Carlo"
            << std::right << std::setw(16) << mc_result.price << '\n'

            << std::left << std::setw(26) << "ANN Analytical"
            << std::right << std::setw(16) << ann_result.price << '\n';

        std::cout
            << "--------------------------------------------\n\n";


        // Compare the approximations against the Merton reference.
        std::cout << "Errors relative to analytical Merton:\n\n";

        std::cout
            << "Monte Carlo error: "
            << mc_result.price - mjd_price << '\n'

            << "ANN error:         "
            << ann_result.price - mjd_price << '\n'

            << "MC standard error: "
            << mc_result.standard_error << '\n';


        std::cout << "\nPricing completed successfully.\n\n";

        return 0;

    } catch (const std::exception& e) {

        std::cerr << "Error: " << e.what() << '\n';

        return 1;
    }
}