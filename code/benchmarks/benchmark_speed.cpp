#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "models/merton_jump_diffusion_model.h"
#include "derivatives/call_option.h"
#include "pricers/merton_analytical_pricer.h"
#include "pricers/monte_carlo_pricer.h"
#include "pricers/ann_pricer.h"
#include "utils/random_generator.h"


using Clock = std::chrono::steady_clock;
using Input = std::array<double, 9>;

// I need to use the calculated prices so the compiler cannot
// simply remove the analytical pricing calculations.
volatile double result_sink = 0.0;


std::vector<Input> generate_inputs(int n) {

    std::mt19937 gen(2026);

    std::uniform_real_distribution<double> S(50, 150);
    std::uniform_real_distribution<double> K(50, 150);
    std::uniform_real_distribution<double> T(0.1, 2.0);
    std::uniform_real_distribution<double> r(0, 0.1);
    std::uniform_real_distribution<double> d(0, 0.08);
    std::uniform_real_distribution<double> sigma(0.05, 0.5);
    std::uniform_real_distribution<double> lambda(0, 1.5);
    std::uniform_real_distribution<double> muJ(-0.2, 0.1);
    std::uniform_real_distribution<double> sigmaJ(0.05, 0.5);

    std::vector<Input> inputs;
    inputs.reserve(n);

    // Same parameter ranges as in my training dataset.
    for (int i = 0; i < n; ++i) {
        inputs.push_back({
            S(gen),
            K(gen),
            T(gen),
            r(gen),
            d(gen),
            sigma(gen),
            lambda(gen),
            muJ(gen),
            sigmaJ(gen)
        });
    }

    return inputs;
}


double measure(
    const std::function<void()>& function,
    int iterations,
    int repetitions
) {
    std::vector<double> times;
    times.reserve(repetitions);

    // Warm up before measuring.
    function();

    for (int rep = 0; rep < repetitions; ++rep) {

        const auto start = Clock::now();

        for (int i = 0; i < iterations; ++i) {
            function();
        }

        const auto end = Clock::now();

        double ms = std::chrono::duration<double, std::milli>(
            end - start
        ).count();

        times.push_back(ms / iterations);
    }

    // The median is less sensitive to occasional background activity.
    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}


void save_result(
    std::ofstream& csv,
    int n,
    const std::string& method,
    double ms,
    int iterations,
    int repetitions
) {
    const double throughput = n * 1000.0 / ms;

    std::cout
        << std::left << std::setw(22) << method
        << std::right << std::setw(14) << std::fixed
        << std::setprecision(6) << ms
        << std::setw(18) << std::setprecision(0) << throughput
        << '\n';

    csv << n << ","
        << method << ","
        << std::setprecision(12) << ms << ","
        << throughput << ","
        << iterations << ","
        << repetitions << '\n';

    csv.flush();
}


int main() {

    const std::string model_path =
        "../models/best_model_analytical.pt";

    const double y_mean = 22.223006564152026;
    const double y_scale = 22.047419209093864;

    // I want to find where GPU batch inference becomes faster
    // than analytical pricing.
    const std::vector<int> sizes = {
        1,
        10,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        750,
        1000,
        10000,
        100000,
        1000000
    };

    // Generate enough inputs for the largest batch automatically.
    const int max_n = *std::max_element(
        sizes.begin(), sizes.end()
    );

    std::filesystem::create_directories(
        "../results/benchmarks"
    );

    std::ofstream csv(
        "../results/benchmarks/speed_benchmark.csv"
    );

    if (!csv) {
        std::cerr << "Could not create benchmark CSV.\n";
        return 1;
    }

    csv << "N,method,median_ms,options_per_second,"
           "iterations,repetitions\n";

    std::cout << "Speed Benchmark\n";

    std::cout << "CUDA available: "
              << (torch::cuda::is_available() ? "Yes" : "No")
              << '\n';

    std::cout << "Maximum batch size: "
              << max_n << "\n\n";

    MertonAnalyticalPricer analytical;
    MonteCarloPricer mc;

    RandomGenerator rng(1);

    // Model loading happens before timing.
    ANN_Pricer ann_cpu(
        model_path,
        y_mean,
        y_scale,
        torch::Device(torch::kCPU)
    );

    std::unique_ptr<ANN_Pricer> ann_gpu;

    if (torch::cuda::is_available()) {
        ann_gpu = std::make_unique<ANN_Pricer>(
            model_path,
            y_mean,
            y_scale,
            torch::Device(torch::kCUDA)
        );
    }

    // All methods use the same generated option parameters.
    const auto all_inputs = generate_inputs(max_n);

    // Check that scalar and batch inference return the same result.
    {
        const Input& x = all_inputs.front();

        MertonJumpDiffusionModel model(
            x[0], x[3], x[4], x[5],
            x[6], x[7], x[8]
        );

        CallOption call(x[1]);

        const double scalar = ann_cpu.price(
            model, call, rng, x[3], x[2], 0
        ).price;

        const double batch = ann_cpu.price_batch({
            x
        }).front();

        if (std::abs(scalar - batch) > 1e-4) {
            std::cerr
                << "Scalar and batch ANN predictions differ.\n";
            return 1;
        }

        std::cout << "ANN scalar/batch test passed.\n";
    }

    constexpr int MC_PATHS = 5000;

    for (int n : sizes) {

        // Prevent invalid iterator ranges.
        if (n <= 0 || n > static_cast<int>(all_inputs.size())) {
            std::cerr << "Invalid batch size: " << n << '\n';
            return 1;
        }

        std::vector<Input> inputs(
            all_inputs.begin(),
            all_inputs.begin() + n
        );

        // Larger batches need fewer iterations.
        const int iterations =
            n <= 10 ? 200 :
            n <= 100 ? 50 :
            n <= 1000 ? 10 :
            n <= 10000 ? 3 : 1;

        const int repetitions = 15;

        std::cout << "\nOptions: " << n << '\n';

        std::cout
            << std::left << std::setw(22) << "Method"
            << std::right << std::setw(14) << "Time (ms)"
            << std::setw(18) << "Options / second"
            << '\n';


        // -----------------------------------------
        // Analytical Merton pricing on the CPU.
        // -----------------------------------------

        auto analytical_run = [&]() {

            double sum = 0.0;

            for (const auto& x : inputs) {

                MertonJumpDiffusionModel model(
                    x[0],
                    x[3],
                    x[4],
                    x[5],
                    x[6],
                    x[7],
                    x[8]
                );

                CallOption call(x[1]);

                sum += analytical.price(
                    model,
                    call,
                    x[3],
                    x[2]
                );
            }

            result_sink = sum;
        };

        save_result(
            csv,
            n,
            "Analytical CPU",
            measure(
                analytical_run,
                iterations,
                repetitions
            ),
            iterations,
            repetitions
        );


        // -----------------------------------------
        // Monte Carlo pricing on the CPU.
        // -----------------------------------------

        // Monte Carlo is more expensive, so I only test
        // batches with up to 1000 options.
        if (n <= 1000) {

            auto mc_run = [&]() {

                double sum = 0.0;

                for (const auto& x : inputs) {

                    MertonJumpDiffusionModel model(
                        x[0],
                        x[3],
                        x[4],
                        x[5],
                        x[6],
                        x[7],
                        x[8]
                    );

                    CallOption call(x[1]);

                    sum += mc.price(
                        model,
                        call,
                        rng,
                        x[3],
                        x[2],
                        MC_PATHS
                    ).price;
                }

                result_sink = sum;
            };

            save_result(
                csv,
                n,
                "Monte Carlo CPU",
                measure(mc_run, 1, 3),
                1,
                3
            );
        }


        // -----------------------------------------
        // ANN batch inference on the CPU.
        // -----------------------------------------

        auto cpu_batch = [&]() {

            const auto prices = ann_cpu.price_batch(
                inputs
            );

            result_sink =
                prices.front() + prices.back();
        };

        save_result(
            csv,
            n,
            "ANN CPU Batch",
            measure(
                cpu_batch,
                iterations,
                repetitions
            ),
            iterations,
            repetitions
        );


        // -----------------------------------------
        // ANN single inference on the CPU.
        // -----------------------------------------

        if (n <= 1000) {

            auto cpu_single = [&]() {

                double sum = 0.0;

                for (const auto& x : inputs) {

                    MertonJumpDiffusionModel model(
                        x[0],
                        x[3],
                        x[4],
                        x[5],
                        x[6],
                        x[7],
                        x[8]
                    );

                    CallOption call(x[1]);

                    sum += ann_cpu.price(
                        model,
                        call,
                        rng,
                        x[3],
                        x[2],
                        0
                    ).price;
                }

                result_sink = sum;
            };

            save_result(
                csv,
                n,
                "ANN CPU Single",
                measure(
                    cpu_single,
                    iterations,
                    repetitions
                ),
                iterations,
                repetitions
            );
        }


        // -----------------------------------------
        // ANN inference on the GPU.
        // -----------------------------------------

        if (ann_gpu) {

            // The batch call copies the predictions back
            // to the CPU, so GPU work is completed before
            // the timer stops.
            auto gpu_batch = [&]() {

                const auto prices = ann_gpu->price_batch(
                    inputs
                );

                result_sink =
                    prices.front() + prices.back();
            };

            save_result(
                csv,
                n,
                "ANN GPU Batch",
                measure(
                    gpu_batch,
                    iterations,
                    repetitions
                ),
                iterations,
                repetitions
            );


            // -----------------------------------------
            // ANN single inference on the GPU.
            // -----------------------------------------

            if (n <= 1000) {

                auto gpu_single = [&]() {

                    double sum = 0.0;

                    for (const auto& x : inputs) {

                        MertonJumpDiffusionModel model(
                            x[0],
                            x[3],
                            x[4],
                            x[5],
                            x[6],
                            x[7],
                            x[8]
                        );

                        CallOption call(x[1]);

                        sum += ann_gpu->price(
                            model,
                            call,
                            rng,
                            x[3],
                            x[2],
                            0
                        ).price;
                    }

                    result_sink = sum;
                };

                save_result(
                    csv,
                    n,
                    "ANN GPU Single",
                    measure(
                        gpu_single,
                        iterations,
                        repetitions
                    ),
                    iterations,
                    repetitions
                );
            }
        }
    }

    std::cout << "\nBenchmark finished.\n";

    std::cout
        << "Results saved to "
        << "../results/benchmarks/speed_benchmark.csv\n";

    return 0;
}