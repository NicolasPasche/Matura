#include "../models/merton_jump_diffusion_model.h"
#include "../derivatives/call_option.h"
#include "../utils/random_generator.h"

#include <iostream>
#include <fstream>
#include <random>
#include <cmath>


int main() {

    RandomGenerator rng(42);

    std::ofstream file("merton_dataset2.csv");


    // CSV header
    file << "S0,K,T,r,d,sigma,lambda,muJ,sigmaJ,"
         << "price,standard_error,lower_CI,upper_CI\n";


    int N = 500000;        // dataset size
    int mc_paths = 5000;   // Monte Carlo simulations per option


    std::mt19937 gen(42);


    std::uniform_real_distribution<double> S0_dist(50, 150);
    std::uniform_real_distribution<double> K_dist(50, 150);
    std::uniform_real_distribution<double> T_dist(0.1, 2.0);

    std::uniform_real_distribution<double> r_dist(0.0, 0.1);

    // continuous dividend yield
    std::uniform_real_distribution<double> d_dist(0.0, 0.08);

    std::uniform_real_distribution<double> sigma_dist(0.05, 0.5);
    std::uniform_real_distribution<double> lambda_dist(0.0, 1.5);
    std::uniform_real_distribution<double> muJ_dist(-0.2, 0.1);
    std::uniform_real_distribution<double> sigmaJ_dist(0.05, 0.5);



    for (int i = 0; i < N; ++i) {


        double S0 = S0_dist(gen);
        double K = K_dist(gen);

        double T = T_dist(gen);

        double r = r_dist(gen);

        // dividend yield
        double d = d_dist(gen);

        double sigma = sigma_dist(gen);

        double lambda = lambda_dist(gen);

        double muJ = muJ_dist(gen);

        double sigmaJ = sigmaJ_dist(gen);



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



        double sum = 0.0;
        double sum_sq = 0.0;



        for (int j = 0; j < mc_paths; ++j) {


            double ST = model.simulate_terminal_price(
                rng,
                T
            );


            double payoff = call.payoff(ST);


            double discounted_payoff =
                std::exp(-r * T) * payoff;



            sum += discounted_payoff;

            sum_sq += discounted_payoff * discounted_payoff;
        }



        double price = sum / mc_paths;



        double variance =
            (sum_sq / mc_paths)
            -
            (price * price);


        if (variance < 0)
            variance = 0;



        double standard_error =
            std::sqrt(
                variance / mc_paths
            );



        double lower_CI =
            price - 1.96 * standard_error;


        double upper_CI =
            price + 1.96 * standard_error;



        file
            << S0 << ","
            << K << ","
            << T << ","
            << r << ","
            << d << ","
            << sigma << ","
            << lambda << ","
            << muJ << ","
            << sigmaJ << ","
            << price << ","
            << standard_error << ","
            << lower_CI << ","
            << upper_CI
            << "\n";



        if (i % 10000 == 0)
            std::cout
                << "Generated "
                << i
                << " samples\n";
    }



    file.close();


    std::cout << "Dataset done.\n";


    return 0;
}