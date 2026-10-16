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

    // Both ANNs were trained with the same input parameters.
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
        double output_mean = 22.22354624,
        double output_scale = 22.05475308
    )
        : model(torch::jit::load(model_path, torch::kCPU)),
          device(
              torch::cuda::is_available()
                  ? torch::kCUDA
                  : torch::kCPU
          ),
          y_mean(output_mean),
          y_scale(output_scale)
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

        // Construct the input vector in the same order as during training.
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

        // Standardize the inputs using the training data statistics.
        std::array<float, 9> scaled_input;

        for (int i = 0; i < 9; ++i) {
            scaled_input[i] = static_cast<float>(
                (input[i] - x_mean[i]) / x_scale[i]
            );
        }

        // Convert the input into a tensor and move it to the selected device.
        auto x = torch::from_blob(
            scaled_input.data(),
            {1, 9},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);

        // I only need predictions, so there's no reason to track gradients.
        torch::NoGradGuard no_grad;

        auto output = model.forward({x}).toTensor();

        // Convert the standardized prediction back into an option price.
        double prediction_scaled = output.item<double>();

        double predicted_price = prediction_scaled * y_scale + y_mean;

        return {predicted_price, 0.0};
    }
};