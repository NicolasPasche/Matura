#pragma once

#include "pricer.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>


class ANN_Pricer : public Pricer {
private:

    mutable torch::jit::script::Module model;
    torch::Device device;

    // All three networks use the same input scaler.
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

    // The output scaler depends on which ANN I want to use.
    const double y_mean;
    const double y_scale;

public:

    explicit ANN_Pricer(
        const std::string& model_path,
        double output_mean = 22.223546235054094,
        double output_scale = 22.05475308405872,
        torch::Device selected_device = torch::Device(
            torch::cuda::is_available() ? torch::kCUDA : torch::kCPU
        )
    )
        : model(torch::jit::load(model_path, torch::kCPU)),
          device(selected_device),
          y_mean(output_mean),
          y_scale(output_scale)
    {
        if (device.is_cuda() && !torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available.");
        }

        if (y_scale <= 0.0) {
            throw std::invalid_argument("Output scale must be positive.");
        }

        model.to(device);
        model.eval();
    }


    // Price a single option.
    Result price(
        const Model& underlying,
        const Derivative& derivative,
        RandomGenerator&,
        double r,
        double T,
        int
    ) const override {

        // The order must match the features used during training.
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

        // Standardize the inputs.
        std::array<float, 9> scaled_input;

        for (int i = 0; i < 9; ++i) {
            scaled_input[i] = static_cast<float>(
                (input[i] - x_mean[i]) / x_scale[i]
            );
        }

        // Create a tensor and move it to the selected device.
        auto x = torch::from_blob(
            scaled_input.data(),
            {1, 9},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);

        // I only need predictions, so I can disable gradient tracking.
        torch::NoGradGuard no_grad;

        auto output = model.forward({x}).toTensor();

        // Convert the standardized prediction back into an option price.
        double prediction = output.item<double>();
        double predicted_price = prediction * y_scale + y_mean;

        return {predicted_price, 0.0};
    }


    // Price multiple options with a single forward pass.
    std::vector<double> price_batch(
        const std::vector<std::array<double, 9>>& inputs
    ) const {

        if (inputs.empty()) {
            return {};
        }

        const std::size_t n = inputs.size();

        // I want to avoid calling the GPU separately for every option.
        std::vector<float> scaled(n * 9);

        for (std::size_t i = 0; i < n; ++i) {
            for (int j = 0; j < 9; ++j) {
                scaled[i * 9 + j] = static_cast<float>(
                    (inputs[i][j] - x_mean[j]) / x_scale[j]
                );
            }
        }

        // Instead of a [1, 9] tensor, I now use an [N, 9] tensor.
        auto x = torch::from_blob(
            scaled.data(),
            {static_cast<int64_t>(n), 9},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);

        torch::NoGradGuard no_grad;

        auto output = model.forward({x}).toTensor();

        // Copy the predictions back to the CPU.
        // This also ensures GPU computation has finished.
        output = output.to(torch::kCPU).contiguous().view({-1});

        const float* raw = output.data_ptr<float>();

        std::vector<double> prices(n);

        for (std::size_t i = 0; i < n; ++i) {
            prices[i] = static_cast<double>(raw[i]) * y_scale + y_mean;
        }

        return prices;
    }
};