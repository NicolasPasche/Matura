#include "analytical_pricer.h"

#include "../derivatives/call_option.h"
#include "../derivatives/put_option.h"
#include "../derivatives/digital_call_option.h"
#include "../derivatives/digital_put_option.h"
#include "../derivatives/forward_contract.h"
#include "../derivatives/zero_coupon_bond.h"

#include <cmath>
#include <stdexcept>


namespace {

// Using erfc gives better numerical accuracy for very small probabilities.
double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

}


double AnalyticalPricer::price(
    const BlackScholesModel& model,
    const Derivative& derivative,
    double T
) const {

    const double S0 = model.getS();
    const double r = model.getR();
    const double d = model.getD();
    const double sigma = model.getSigma();

    if (!std::isfinite(S0) || !std::isfinite(r) ||
        !std::isfinite(d) || !std::isfinite(sigma) ||
        !std::isfinite(T) ||
        S0 <= 0.0 || sigma < 0.0 || T < 0.0) {

        throw std::invalid_argument("Invalid Black-Scholes parameters");
    }

    // Bonds and forwards do not require the Black-Scholes option formulas.
    if (dynamic_cast<const ZeroCouponBond*>(&derivative)) {
        return std::exp(-r * T);
    }

    if (dynamic_cast<const ForwardContract*>(&derivative)) {
        const double K = derivative.getStrike();

        if (!std::isfinite(K) || K <= 0.0) {
            throw std::invalid_argument("Invalid forward strike");
        }

        return S0 * std::exp(-d * T)
             - K * std::exp(-r * T);
    }

    // The derivative classes are kept separate, but share one pricer.
    const bool is_call =
        dynamic_cast<const CallOption*>(&derivative) != nullptr;

    const bool is_put =
        dynamic_cast<const PutOption*>(&derivative) != nullptr;

    const bool is_digital_call =
        dynamic_cast<const DigitalCallOption*>(&derivative) != nullptr;

    const bool is_digital_put =
        dynamic_cast<const DigitalPutOption*>(&derivative) != nullptr;

    if (!is_call && !is_put && !is_digital_call && !is_digital_put) {
        throw std::invalid_argument("Unsupported derivative type");
    }

    const double K = derivative.getStrike();

    if (!std::isfinite(K) || K <= 0.0) {
        throw std::invalid_argument("Invalid option strike");
    }

    // Without time or volatility, the terminal stock price is known.
    // The existing payoff function handles these cases without division by zero.
    if (T == 0.0 || sigma == 0.0) {
        const double ST = S0 * std::exp((r - d) * T);

        return std::exp(-r * T) * derivative.payoff(ST);
    }

    // All four option types use the same d1 and d2 values.
    const double sqrtT = std::sqrt(T);

    const double d1 =
        (std::log(S0 / K) + (r - d + 0.5 * sigma * sigma) * T)
        / (sigma * sqrtT);

    const double d2 = d1 - sigma * sqrtT;

    const double discounted_stock = S0 * std::exp(-d * T);
    const double discounted_strike = K * std::exp(-r * T);
    const double discounted_unit = std::exp(-r * T);

    if (is_call) {
        return discounted_stock * normal_cdf(d1)
             - discounted_strike * normal_cdf(d2);
    }

    if (is_put) {
        return discounted_strike * normal_cdf(-d2)
             - discounted_stock * normal_cdf(-d1);
    }

    if (is_digital_call) {
        return discounted_unit * normal_cdf(d2);
    }

    if (is_digital_put) {
        return discounted_unit * normal_cdf(-d2);
    }

    throw std::logic_error("Unexpected derivative type");
}