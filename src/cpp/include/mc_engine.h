#pragma once

#include <vector>
#include <cstddef>

struct MCResult {
    std::vector<double> paths;
    std::size_t n_sims = 0;
    std::size_t horizon = 0;
    std::size_t n_assets = 0;
};

struct Summary {
    double var = 0.0;
    double cvar = 0.0;
    double q95 = 0.0;
};

MCResult simulate_paths(
    const std::vector<double>& mu,
    const std::vector<double>& sigma_flat,
    std::size_t n_sims,
    std::size_t horizon
);

Summary summarize_paths(
    const std::vector<double>& paths,
    std::size_t n_sims,
    std::size_t horizon,
    std::size_t n_assets,
    double alpha
);
