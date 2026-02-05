#pragma once

#include <vector>
#include <cstddef>

struct MCResult {
    /// Flat array of simulated paths in (sim, time, asset) order.
    std::vector<double> paths;
    std::size_t n_sims = 0;
    std::size_t horizon = 0;
    std::size_t n_assets = 0;
};

struct Summary {
    /// Value-at-Risk at the requested alpha.
    double var = 0.0;
    /// Conditional Value-at-Risk at the requested alpha.
    double cvar = 0.0;
    /// 95th percentile of the PnL distribution.
    double q95 = 0.0;
};

/// Simulate multivariate Gaussian paths for a given mean/covariance.
MCResult simulate_paths(
    const std::vector<double>& mu,
    const std::vector<double>& sigma_flat,
    std::size_t n_sims,
    std::size_t horizon
);

/// Summarize simulated paths into VaR/CVaR and q95 metrics.
Summary summarize_paths(
    const std::vector<double>& paths,
    std::size_t n_sims,
    std::size_t horizon,
    std::size_t n_assets,
    double alpha
);
