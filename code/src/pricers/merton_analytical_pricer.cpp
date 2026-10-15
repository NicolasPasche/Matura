#include "merton_analytical_pricer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace {

double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

}


double MertonAnalyticalPricer::price(
    const MertonJumpDiffusionModel& model,
    const CallOption& option,
    double r,
    double T
) const {

    const double S0 = model.getS();
    const double K = option.getStrike();
    const double d = model.getD();
    const double sigma = model.getSigma();
    const double lambda = model.getLambda();
    const double muJ = model.getMuJ();
    const double sigmaJ = model.getSigmaJ();

    // Invalid parameters would make the probability model undefined.
    if (
        !std::isfinite(S0) ||
        !std::isfinite(K) ||
        !std::isfinite(r) ||
        !std::isfinite(d) ||
        !std::isfinite(sigma) ||
        !std::isfinite(lambda) ||
        !std::isfinite(muJ) ||
        !std::isfinite(sigmaJ) ||
        !std::isfinite(T) ||
        S0 <= 0.0 ||
        K <= 0.0 ||
        T < 0.0 ||
        sigma < 0.0 ||
        lambda < 0.0 ||
        sigmaJ < 0.0
    ) {
        throw std::invalid_argument("Invalid Merton model parameters");
    }

    // At maturity, there is no remaining uncertainty to price.
    if (T == 0.0) {
        return option.payoff(S0);
    }

    constexpr double TOLERANCE = 1e-10;
    constexpr int MAX_TERMS = 1000;

    // The jump compensator keeps the expected stock growth risk-neutral.
    const double jump_multiplier =
        std::exp(muJ + 0.5 * sigmaJ * sigmaJ);

    const double kappa = jump_multiplier - 1.0;

    const double poisson_mean = lambda * T;

    // The stock-related part of the formula has adjusted Poisson weights.
    const double adjusted_mean =
        poisson_mean * jump_multiplier;

    if (
        !std::isfinite(adjusted_mean) ||
        !std::isfinite(poisson_mean)
    ) {
        throw std::invalid_argument("Jump parameters are too large");
    }

    double poisson_weight = std::exp(-poisson_mean);
    double adjusted_weight = std::exp(-adjusted_mean);

    // The recursive weights must start above zero to work correctly.
    if (poisson_weight == 0.0 || adjusted_weight == 0.0) {
        throw std::runtime_error("Poisson mean is too large");
    }

    double poisson_sum = 0.0;
    double adjusted_sum = 0.0;

    double call_price = 0.0;

    const double discounted_stock =
        S0 * std::exp(-d * T);

    const double discounted_strike =
        K * std::exp(-r * T);

    const double log_moneyness =
        std::log(S0 / K);

    const double base_log_return =
        log_moneyness +
        (r - d - lambda * kappa - 0.5 * sigma * sigma) * T;

    const double jump_variance =
        sigmaJ * sigmaJ;

    // Each term represents one possible number of jumps.
    for (int n = 0; n < MAX_TERMS; ++n) {

        const double variance =
            sigma * sigma * T + n * jump_variance;

        const double log_return =
            base_log_return + n * muJ;

        double nd1;
        double nd2;

        if (variance > 0.0) {

            const double std_dev = std::sqrt(variance);

            const double d2 = log_return / std_dev;
            const double d1 = d2 + std_dev;

            nd1 = normal_cdf(d1);
            nd2 = normal_cdf(d2);

        } else {

            // With no variance, the stock price is known for this jump count.
            const double in_the_money =
                log_return > 0.0 ? 1.0 : 0.0;

            nd1 = in_the_money;
            nd2 = in_the_money;
        }

        const double stock_term =
            discounted_stock * adjusted_weight * nd1;

        const double strike_term =
            discounted_strike * poisson_weight * nd2;

        call_price += stock_term - strike_term;

        poisson_sum += poisson_weight;
        adjusted_sum += adjusted_weight;

        // The remaining probability mass bounds the omitted terms.
        // This avoids choosing an arbitrary fixed number of jumps.
        const double remaining_error =
            discounted_stock *
                std::max(0.0, 1.0 - adjusted_sum)
            +
            discounted_strike *
                std::max(0.0, 1.0 - poisson_sum);

        if (remaining_error < TOLERANCE) {
            return std::max(0.0, call_price);
        }

        // Recursive Poisson probabilities avoid factorial calculations.
        const double next_n = static_cast<double>(n + 1);

        poisson_weight *= poisson_mean / next_n;
        adjusted_weight *= adjusted_mean / next_n;
    }

    throw std::runtime_error(
        "Merton series did not converge within MAX_TERMS"
    );
}