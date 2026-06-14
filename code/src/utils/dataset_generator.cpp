#include "../models/merton_jump_diffusion_model.h"
#include "../derivatives/call_option.h"
#include "../utils/random_generator.h"

#include <iostream>
#include <fstream>
#include <random>

int main() {

    RandomGenerator rng(42);

    std::ofstream file("merton_dataset.csv");

    // header
    file << "S0,K,T,r,sigma,lambda,muJ,sigmaJ,price\n";

    int N = 500000;  // dataset size
    int mc_paths = 5000; // per point MC (trade accuracy vs speed)

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> S0_dist(50, 150);
    std::uniform_real_distribution<double> K_dist(50, 150);
    std::uniform_real_distribution<double> T_dist(0.1, 2.0);
    std::uniform_real_distribution<double> r_dist(0.0, 0.1);
    std::uniform_real_distribution<double> sigma_dist(0.05, 0.5);
    std::uniform_real_distribution<double> lambda_dist(0.0, 1.5);
    std::uniform_real_distribution<double> muJ_dist(-0.2, 0.1);
    std::uniform_real_distribution<double> sigmaJ_dist(0.05, 0.5);

    for (int i = 0; i < N; ++i) {

        double S0 = S0_dist(gen);
        double K = K_dist(gen);
        double T = T_dist(gen);
        double r = r_dist(gen);
        double sigma = sigma_dist(gen);
        double lambda = lambda_dist(gen);
        double muJ = muJ_dist(gen);
        double sigmaJ = sigmaJ_dist(gen);

        MertonJumpDiffusionModel model(S0, r, 0.0, sigma, lambda, muJ, sigmaJ);
        CallOption call(K);

        double sum = 0.0;

        for (int j = 0; j < mc_paths; ++j) {
            double ST = model.simulate_terminal_price(rng, T);
            sum += call.payoff(ST);
        }

        double price = std::exp(-r * T) * (sum / mc_paths);

        file << S0 << "," << K << "," << T << "," << r << ","
             << sigma << "," << lambda << "," << muJ << "," << sigmaJ << ","
             << price << "\n";

        if (i % 10000 == 0)
            std::cout << "Generated " << i << " samples\n";
    }

    file.close();

    std::cout << "Dataset done.\n";

    return 0;
}