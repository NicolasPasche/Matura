#include "models/merton_jump_diffusion_model.h"

#include "derivatives/call_option.h"
#include "pricers/monte_carlo_pricer.h"
#include "pricers/ann_pricer.h"
#include "utils/random_generator.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

struct TestResult {
    double mc;
    double mc_se;
    double ann;
    double abs_error;
    double rel_error;
};

int main() {

    RandomGenerator rng(42);

    ANN_Pricer ann_pricer(
        "/home/nicolas/Dokumente/Schule/Matura Thesis/derivatives_pricing_eninge/models/best_model_dividend.pt"
    );

    MonteCarloPricer mc_pricer;

    const int TEST_CASES = 1000;
    const int MC_SIMULATIONS = 50000;

    std::vector<TestResult> results;
    results.reserve(TEST_CASES);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\n========================================\n";
    std::cout << "       ANN vs Monte Carlo Benchmark\n";
    std::cout << "========================================\n\n";

    std::cout << "Test cases: "
              << TEST_CASES << "\n";

    std::cout << "MC simulations per case: "
              << MC_SIMULATIONS << "\n\n";


    for (int i = 0; i < TEST_CASES; ++i) {

        double S0 =
            70.0 + rng.get_normal() * 15.0;

        S0 = std::clamp(S0, 50.0, 150.0);


        double K =
            70.0 + rng.get_normal() * 15.0;

        K = std::clamp(K, 50.0, 150.0);


        double T =
            0.1 + std::abs(rng.get_normal()) * 0.5;

        T = std::clamp(T, 0.1, 2.0);


        double r =
            0.01 + std::abs(rng.get_normal()) * 0.02;

        r = std::clamp(r, 0.01, 0.10);


        double d =
            0.01 + std::abs(rng.get_normal()) * 0.01;

        d = std::clamp(d, 0.01, 0.05);


        double sigma =
            0.15 + std::abs(rng.get_normal()) * 0.08;

        sigma = std::clamp(sigma, 0.15, 0.60);


        double lambda =
            0.2 + std::abs(rng.get_normal()) * 0.3;

        lambda = std::clamp(lambda, 0.1, 1.5);


        double muJ =
            -0.05 - std::abs(rng.get_normal()) * 0.05;

        muJ = std::clamp(muJ, -0.30, -0.01);


        double sigmaJ =
            0.10 + std::abs(rng.get_normal()) * 0.08;

        sigmaJ = std::clamp(sigmaJ, 0.05, 0.40);


        MertonJumpDiffusionModel model(
            S0,
            r,
            d,
            sigma,
            lambda,
            muJ,
            sigmaJ
        );


        CallOption call(K);


        Pricer::Result mc_result =
            mc_pricer.price(
                model,
                call,
                rng,
                r,
                T,
                MC_SIMULATIONS
            );


        Pricer::Result ann_result =
            ann_pricer.price(
                model,
                call,
                rng,
                r,
                T,
                0
            );


        double mc_price =
            mc_result.price;

        double mc_se =
            mc_result.standard_error;

        double ann_price =
            ann_result.price;


        double abs_error =
            std::abs(ann_price - mc_price);


        double rel_error = 0.0;

        if (mc_price > 1.0) {
            rel_error =
                abs_error / mc_price;
        }


        results.push_back({
            mc_price,
            mc_se,
            ann_price,
            abs_error,
            rel_error
        });


        if (i < 10) {

            std::cout
                << "Test "
                << std::setw(3)
                << i + 1

                << " | MC: "
                << std::setw(10)
                << mc_price

                << " | SE: "
                << std::setw(8)
                << mc_se

                << " | ANN: "
                << std::setw(10)
                << ann_price

                << " | Abs: "
                << std::setw(8)
                << abs_error

                << " | Rel: ";

            if (mc_price > 1.0) {
                std::cout
                    << std::setw(8)
                    << rel_error * 100.0
                    << "%";
            }
            else {
                std::cout
                    << "   n/a";
            }

            std::cout << "\n";
        }
    }


    double mae = 0.0;
    double mse = 0.0;
    double bias = 0.0;

    int relative_count = 0;

    double mean_relative_error = 0.0;


    for (const auto& r : results) {

        mae += r.abs_error;

        mse +=
            r.abs_error * r.abs_error;

        bias +=
            r.ann - r.mc;


        if (r.mc > 1.0) {

            mean_relative_error +=
                r.rel_error;

            ++relative_count;
        }
    }


    mae /= TEST_CASES;

    double rmse =
        std::sqrt(mse / TEST_CASES);

    bias /= TEST_CASES;

    mean_relative_error /=
        std::max(relative_count, 1);


    double median_error;

    std::vector<double> errors;

    for (const auto& r : results) {
        errors.push_back(r.abs_error);
    }

    std::sort(
        errors.begin(),
        errors.end()
    );


    if (errors.size() % 2 == 0) {

        median_error =
            (
                errors[errors.size() / 2 - 1] +
                errors[errors.size() / 2]
            ) / 2.0;

    } else {

        median_error =
            errors[errors.size() / 2];
    }


    int within_1_percent = 0;
    int within_2_percent = 0;
    int within_5_percent = 0;


    for (const auto& r : results) {

        if (r.mc <= 1.0)
            continue;


        double percentage =
            r.abs_error / r.mc;


        if (percentage <= 0.01)
            ++within_1_percent;

        if (percentage <= 0.02)
            ++within_2_percent;

        if (percentage <= 0.05)
            ++within_5_percent;
    }


    std::cout
        << "\n\n========================================\n";

    std::cout
        << "          Benchmark results\n";

    std::cout
        << "========================================\n\n";


    std::cout
        << "Mean absolute error (MAE):\n"
        << mae
        << "\n\n";


    std::cout
        << "Root mean squared error (RMSE):\n"
        << rmse
        << "\n\n";


    std::cout
        << "Median absolute error:\n"
        << median_error
        << "\n\n";


    std::cout
        << "Mean signed error (bias):\n"
        << bias
        << "\n\n";


    std::cout
        << "Mean relative error:\n"
        << mean_relative_error * 100.0
        << "%\n";

    std::cout
        << "(Only calls with MC price > 1.00)\n\n";


    std::cout
        << "Relative error <= 1%:\n"
        << within_1_percent
        << " / "
        << relative_count
        << " ("
        << 100.0 * within_1_percent /
               std::max(relative_count, 1)
        << "%)\n\n";


    std::cout
        << "Relative error <= 2%:\n"
        << within_2_percent
        << " / "
        << relative_count
        << " ("
        << 100.0 * within_2_percent /
               std::max(relative_count, 1)
        << "%)\n\n";


    std::cout
        << "Relative error <= 5%:\n"
        << within_5_percent
        << " / "
        << relative_count
        << " ("
        << 100.0 * within_5_percent /
               std::max(relative_count, 1)
        << "%)\n\n";


    int cheap_count = 0;
    int medium_count = 0;
    int expensive_count = 0;

    double cheap_error = 0.0;
    double medium_error = 0.0;
    double expensive_error = 0.0;


    for (const auto& r : results) {

        if (r.mc < 5.0) {

            cheap_error += r.abs_error;
            ++cheap_count;

        }
        else if (r.mc < 15.0) {

            medium_error += r.abs_error;
            ++medium_count;

        }
        else {

            expensive_error += r.abs_error;
            ++expensive_count;
        }
    }


    std::cout
        << "Absolute error by option price:\n\n";


    if (cheap_count > 0) {

        std::cout
            << "MC price < 5:\n"
            << "  Cases: "
            << cheap_count
            << "\n"
            << "  MAE: "
            << cheap_error / cheap_count
            << "\n\n";
    }


    if (medium_count > 0) {

        std::cout
            << "5 <= MC price < 15:\n"
            << "  Cases: "
            << medium_count
            << "\n"
            << "  MAE: "
            << medium_error / medium_count
            << "\n\n";
    }


    if (expensive_count > 0) {

        std::cout
            << "MC price >= 15:\n"
            << "  Cases: "
            << expensive_count
            << "\n"
            << "  MAE: "
            << expensive_error / expensive_count
            << "\n\n";
    }


    std::cout
        << "========================================\n";

    return 0;
}