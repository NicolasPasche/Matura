#include "../models/merton_jump_diffusion_model.h"
#include "../derivatives/call_option.h"
#include "../pricers/merton_analytical_pricer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <array>
#include <filesystem>
#include <cmath>
#include <stdexcept>


int main(int argc, char* argv[]) {

    // Input and output paths are arguments so the original dataset
    // does not need to be moved or modified.
    if (argc != 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <input.csv> <output.csv>\n";

        return 1;
    }

    const std::filesystem::path input_path = argv[1];
    const std::filesystem::path output_path = argv[2];

    try {

        std::ifstream input(input_path);

        if (!input.is_open()) {
            throw std::runtime_error(
                "Could not open input dataset: " + input_path.string()
            );
        }

        // Avoid accidentally overwriting an existing dataset.
        if (std::filesystem::exists(output_path)) {
            throw std::runtime_error(
                "Output file already exists: " + output_path.string()
            );
        }

        const std::array<std::string, 9> columns = {
            "S0",
            "K",
            "T",
            "r",
            "d",
            "sigma",
            "lambda",
            "muJ",
            "sigmaJ"
        };

        std::string header;

        if (!std::getline(input, header)) {
            throw std::runtime_error("Input dataset is empty");
        }

        // The two datasets must use the same input columns in the same order.
        std::stringstream header_stream(header);
        std::string column;

        for (const std::string& expected : columns) {

            if (!std::getline(header_stream, column, ',') ||
                column != expected) {

                throw std::runtime_error(
                    "Unexpected CSV header. Expected column: " + expected
                );
            }
        }

        if (!std::getline(header_stream, column, ',') ||
            column != "price") {

            throw std::runtime_error(
                "Expected the original Monte Carlo price column"
            );
        }

        std::ofstream output(output_path);

        if (!output.is_open()) {
            throw std::runtime_error(
                "Could not create output dataset"
            );
        }

        // Keeping the target column name unchanged simplifies
        // the training pipeline for the second neural network.
        output << "S0,K,T,r,d,sigma,lambda,muJ,sigmaJ,price\n";

        // The analytical prices should not lose precision when saved.
        output << std::setprecision(17);

        MertonAnalyticalPricer pricer;

        std::string line;

        std::size_t count = 0;
        std::size_t line_number = 1;

        while (std::getline(input, line)) {

            ++line_number;

            if (line.empty()) {
                continue;
            }

            std::stringstream row(line);

            std::array<std::string, 9> fields;
            std::array<double, 9> values;

            // Reusing the original input values makes the two
            // training datasets directly comparable.
            for (int j = 0; j < 9; ++j) {

                if (!std::getline(row, fields[j], ',')) {
                    throw std::runtime_error(
                        "Missing input value at line " +
                        std::to_string(line_number)
                    );
                }

                try {
                    std::size_t parsed = 0;

                    values[j] = std::stod(fields[j], &parsed);

                    if (parsed != fields[j].size()) {
                        throw std::invalid_argument(
                            "Unexpected characters"
                        );
                    }

                } catch (const std::exception&) {

                    throw std::runtime_error(
                        "Invalid input value at line " +
                        std::to_string(line_number)
                    );
                }
            }

            const double S0 = values[0];
            const double K = values[1];
            const double T = values[2];
            const double r = values[3];
            const double d = values[4];
            const double sigma = values[5];
            const double lambda = values[6];
            const double muJ = values[7];
            const double sigmaJ = values[8];

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

            const double price =
                pricer.price(model, call, r, T);

            if (!std::isfinite(price) || price < 0.0) {
                throw std::runtime_error(
                    "Invalid analytical price at line " +
                    std::to_string(line_number)
                );
            }

            // Preserve the original inputs exactly as they appear
            // in the Monte Carlo dataset.
            for (const std::string& field : fields) {
                output << field << ",";
            }

            output << price << "\n";

            ++count;

            if (count % 10000 == 0) {
                std::cout
                    << "Generated " << count
                    << " analytical prices\n";
            }
        }

        if (input.bad() || !output.good()) {
            throw std::runtime_error("Error reading or writing dataset");
        }

        input.close();
        output.close();

        if (!output) {
            throw std::runtime_error("Failed to finish writing dataset");
        }

        std::cout << "\nDataset generation completed.\n";
        std::cout << "Total samples: " << count << "\n";
        std::cout << "Output: " << output_path << "\n";

        return 0;

    } catch (const std::exception& e) {

        std::cerr << "Error: " << e.what() << "\n";

        return 1;
    }
}