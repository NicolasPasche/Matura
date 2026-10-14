#pragma once

#include "pricer.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <string>
#include <array>

class ANN_Pricer : public Pricer {
private:

    mutable torch::jit::script::Module model;
    torch::Device device;

    const std::array<double, 9> x_mean = {
        99.9920428,
        99.9648586,
        1.05086311,
        0.0499629072,
        0.0400024049,
        0.275001120,
        0.749113903,
        -0.0499064365,
        0.275170031
    };

    const std::array<double, 9> x_scale = {
        28.8708281,
        28.8671196,
        0.548299487,
        0.0288809425,
        0.0230830861,
        0.129709634,
        0.433354290,
        0.0866032479,
        0.129855032
    };

    const double y_mean = 22.22354624;
    const double y_scale = 22.05475308;

public:

    explicit ANN_Pricer(const std::string& model_path)
        : device(
            torch::cuda::is_available()
                ? torch::kCUDA
                : torch::kCPU
        ),
          model(torch::jit::load(model_path, torch::kCPU))
    {
        model.to(device);
        model.eval();
    }

    Result price(
        const Model& underlying,
        const Derivative& derivative,
        RandomGenerator&,
        double r,
        double T,
        int
    ) const override {

        // 1. Construct input vector
        std::array<double, 9> input = {
            underlying.getS(),
            derivative.getStrike(),
            T,
            r,
            underlying.getD(),
            underlying.getSigma(),
            underlying.getLambda(),
            underlying.getMuJ(),
            underlying.getSigmaJ()
        };

        // 2. Standardize input
        std::array<float, 9> scaled_input;

        for (int i = 0; i < 9; ++i) {
            scaled_input[i] =
                static_cast<float>(
                    (input[i] - x_mean[i]) / x_scale[i]
                );
        }

        // 3. Convert to tensor
        auto x = torch::from_blob(
            scaled_input.data(),
            {1, 9},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);

        // 4. Neural network prediction
        torch::NoGradGuard no_grad;

        auto output = model.forward({x}).toTensor();

        // 5. Convert prediction back from standardized scale
        double prediction_scaled = output.item<double>();

        double price =
            prediction_scaled * y_scale + y_mean;

        return {price, 0.0};
    }
};