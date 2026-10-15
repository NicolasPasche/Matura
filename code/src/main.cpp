#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

#include "models/black_scholes_model.h"

#include "derivatives/call_option.h"
#include "derivatives/put_option.h"
#include "derivatives/digital_call_option.h"
#include "derivatives/digital_put_option.h"
#include "derivatives/forward_contract.h"
#include "derivatives/zero_coupon_bond.h"

#include "pricers/analytical_pricer.h"


// A small tolerance is necessary because floating-point calculations
// cannot always represent mathematical results exactly.
bool approximately_equal(
    double actual,
    double expected,
    double tolerance = 1e-9
) {
    return std::isfinite(actual)
        && std::isfinite(expected)
        && std::abs(actual - expected) <= tolerance;
}


// Keeping the tests in one function makes failures easier to identify.
bool check(
    const std::string& name,
    double actual,
    double expected,
    double tolerance = 1e-9
) {
    const bool passed =
        approximately_equal(actual, expected, tolerance);

    std::cout
        << (passed ? "[PASS] " : "[FAIL] ")
        << std::left << std::setw(30) << name
        << "Actual: " << std::setw(15) << actual
        << "Expected: " << expected
        << '\n';

    return passed;
}


int main() {

    const double S0 = 100.0;
    const double K = 100.0;
    const double r = 0.05;
    const double d = 0.0;
    const double sigma = 0.20;
    const double T = 1.0;

    BlackScholesModel model(S0, r, d, sigma);

    CallOption call(K);
    PutOption put(K);

    DigitalCallOption digital_call(K);
    DigitalPutOption digital_put(K);

    ForwardContract forward(K);
    ZeroCouponBond bond;

    AnalyticalPricer pricer;

    int passed = 0;
    int total = 0;

    std::cout << std::fixed << std::setprecision(10);

    std::cout << "Black-Scholes Analytical Pricer Tests\n\n";


    // These reference prices are known for the standard Black-Scholes example.
    const double call_price = pricer.price(model, call, T);
    const double put_price = pricer.price(model, put, T);

    total++;
    passed += check(
        "European Call",
        call_price,
        10.4505835722
    );

    total++;
    passed += check(
        "European Put",
        put_price,
        5.5735260223
    );


    // Put-call parity provides an independent consistency check.
    const double parity_left = call_price - put_price;

    const double parity_right =
        S0 * std::exp(-d * T)
        - K * std::exp(-r * T);

    total++;
    passed += check(
        "Put-Call Parity",
        parity_left,
        parity_right
    );


    // A digital call and put must together pay one unit at maturity,
    // except at the strike if their payoff definitions overlap or leave a gap.
    // With positive volatility, the probability of ending exactly at K is zero.
    const double digital_call_price =
        pricer.price(model, digital_call, T);

    const double digital_put_price =
        pricer.price(model, digital_put, T);

    total++;
    passed += check(
        "Digital Call + Put",
        digital_call_price + digital_put_price,
        std::exp(-r * T)
    );


    // The forward price should match the expression in put-call parity.
    const double forward_price =
        pricer.price(model, forward, T);

    total++;
    passed += check(
        "Forward Contract",
        forward_price,
        parity_right
    );


    // A zero-coupon bond paying one unit has a known discounted value.
    const double bond_price =
        pricer.price(model, bond, T);

    total++;
    passed += check(
        "Zero-Coupon Bond",
        bond_price,
        std::exp(-r * T)
    );


    // At maturity, an option is worth its immediate payoff.
    total++;
    passed += check(
        "Call at Maturity",
        pricer.price(model, call, 0.0),
        call.payoff(S0)
    );

    total++;
    passed += check(
        "Put at Maturity",
        pricer.price(model, put, 0.0),
        put.payoff(S0)
    );


    // Zero volatility removes randomness from the stock price.
    // This also tests the special case that would otherwise divide by zero.
    const double zero_sigma = 0.0;

    BlackScholesModel deterministic_model(
        S0, r, d, zero_sigma
    );

    const double deterministic_ST =
        S0 * std::exp((r - d) * T);

    const double expected_call =
        std::exp(-r * T) * call.payoff(deterministic_ST);

    const double expected_put =
        std::exp(-r * T) * put.payoff(deterministic_ST);

    total++;
    passed += check(
        "Call with Zero Volatility",
        pricer.price(deterministic_model, call, T),
        expected_call
    );

    total++;
    passed += check(
        "Put with Zero Volatility",
        pricer.price(deterministic_model, put, T),
        expected_put
    );


    // The same parity relation must also hold when dividends are included.
    const double dividend_yield = 0.03;

    BlackScholesModel dividend_model(
        S0, r, dividend_yield, sigma
    );

    const double dividend_call =
        pricer.price(dividend_model, call, T);

    const double dividend_put =
        pricer.price(dividend_model, put, T);

    const double dividend_parity =
        S0 * std::exp(-dividend_yield * T)
        - K * std::exp(-r * T);

    total++;
    passed += check(
        "Parity with Dividends",
        dividend_call - dividend_put,
        dividend_parity
    );


    // Report the overall result so a failed test cannot go unnoticed.
    std::cout << "\n----------------------------------------\n";

    std::cout
        << "Passed: " << passed
        << " / " << total << '\n';

    if (passed == total) {
        std::cout << "All tests passed.\n";
        return 0;
    }

    std::cout << "Some tests failed.\n";
    return 1;
}