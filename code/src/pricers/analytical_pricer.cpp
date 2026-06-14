/* 
Ugliest code EVER written...

I really couldn't find an elegant way to build an analytical pricer, because I want my derivatives to be in 
seperte classes, and I don't want to build a seperate pricer for each derivative.

So I had to go down the if statement path...

I had major problems building this pricer, which is visible in the out-dated comments. Just don't 
get confused when you don't get them. I'll maybe clean up later, but not now.
*/
#include "../pricers/analytical_pricer.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

static double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double AnalyticalPricer::price(
    const BlackScholesModel& model,
    const Derivative& derivative,
    double T
) const {

    double S0 = model.getS();
    double r = model.getR();
    double d = model.getD();
    double sigma = model.getSigma();

    std::string type = typeid(derivative).name();

    // std::cout << "[DEBUG] type = " << type << std::endl;

    // -------------------------
    // FORWARD
    // -------------------------
    

    // -------------------------
    // ZERO COUPON BOND
    // -------------------------
    if (type == "14ZeroCouponBond") {
        return std::exp(-r * T);
    }

    // -------------------------
    // STRIKE-BASED INSTRUMENTS
    // -------------------------
    double K = derivative.getStrike();

    if (K <= 0.0) {
        throw std::invalid_argument("Invalid strike for option-type derivative");
    }

    double sqrtT = std::sqrt(T);

    double d1 = (std::log(S0 / K) + (r - d + 0.5 * sigma * sigma) * T)
                / (sigma * sqrtT);

    double d2 = d1 - sigma * sqrtT;

    // CALL
    if (type == "10CallOption") {

        // Printing model parameters for debugging
        // std::cout << "d1 = " << d1 << '\n';
        // std::cout << "d2 = " << d2 << '\n';
        // std::cout << "N(d1) = " << norm_cdf(d1) << '\n';
        // std::cout << "N(d2) = " << norm_cdf(d2) << '\n';

        return S0 * std::exp(-d * T) * norm_cdf(d1)
             - K * std::exp(-r * T) * norm_cdf(d2);
    }

    if (type == "15ForwardContract") {
        double K = derivative.getStrike();
        return S0 * std::exp(-d * T) - K * std::exp(-r * T);
    }

    // PUT
    if (type == "9PutOption") {
        return K * std::exp(-r * T) * norm_cdf(-d2)
             - S0 * std::exp(-d * T) * norm_cdf(-d1);
    }

    // DIGITAL CALL
    if (type == "17DigitalCallOption") {
        return std::exp(-r * T) * norm_cdf(d2);
    }

    // DIGITAL PUT
    if (type == "16DigitalPutOption") {
        return std::exp(-r * T) * norm_cdf(-d2);
    }

    throw std::invalid_argument("Unsupported type: " + type);
}