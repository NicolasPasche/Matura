#include "models/merton_jump_diffusion_model.h"

#include "derivatives/call_option.h"
#include "derivatives/put_option.h"

#include "pricers/monte_carlo_pricer.h"
#include "utils/random_generator.h"

#include <iostream>
#include <iomanip>


int main() {


    RandomGenerator rng(42);


    // Market parameters

    double S0 = 100.0;

    double r = 0.05;        // risk-free interest rate
    double d = 0.02;        // dividend yield

    double sigma = 0.25;


    // Merton jump parameters

    double lambda = 0.5;
    double muJ = -0.1;
    double sigmaJ = 0.2;



    // Option parameters

    double K = 100.0;
    double T = 1.0;



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
    PutOption put(K);



    MonteCarloPricer pricer;



    int simulations = 100000;



    Pricer::Result call_result = pricer.price(
        model,
        call,
        rng,
        r,
        T,
        simulations
    );


    Pricer::Result put_result = pricer.price(
        model,
        put,
        rng,
        r,
        T,
        simulations
    );



    std::cout << std::fixed
              << std::setprecision(6);



    std::cout << "\n===== Merton Jump Diffusion =====\n\n";


    std::cout << "Underlying:\n";
    std::cout << "S0 = " << S0 << "\n";


    std::cout << "\nOption:\n";
    std::cout << "K  = " << K << "\n";
    std::cout << "T  = " << T << "\n";


    std::cout << "\nMarket:\n";
    std::cout << "r = " << r << "\n";
    std::cout << "d = " << d << "\n";


    std::cout << "\nVolatility:\n";
    std::cout << "sigma = " << sigma << "\n";


    std::cout << "\nJump parameters:\n";
    std::cout << "lambda = " << lambda << "\n";
    std::cout << "muJ = " << muJ << "\n";
    std::cout << "sigmaJ = " << sigmaJ << "\n";



    std::cout << "\nMonte Carlo results:\n";


    std::cout << "\nCall:\n";
    std::cout << "Price: "
              << call_result.price
              << "\n";

    std::cout << "Standard Error: "
              << call_result.standard_error
              << "\n";


    std::cout << "\nPut:\n";
    std::cout << "Price: "
              << put_result.price
              << "\n";

    std::cout << "Standard Error: "
              << put_result.standard_error
              << "\n";


    return 0;
}