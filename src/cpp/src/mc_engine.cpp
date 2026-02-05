#include "mc_engine.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

// Compute a lower-triangular Cholesky factor for a symmetric matrix.
static std::vector<double> cholesky_lower(
    const std::vector<double>& a,
    std::size_t n
) {
    std::vector<double> l(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                sum += l[i * n + k] * l[j * n + k];
            }
            if (i == j) {
                double diag = a[i * n + i] - sum;
                l[i * n + j] = diag > 0.0 ? std::sqrt(diag) : 0.0;
            } else {
                double denom = l[j * n + j];
                l[i * n + j] = denom > 0.0 ? (a[i * n + j] - sum) / denom : 0.0;
            }
        }
    }
    return l;
}

// Simulate multivariate Gaussian paths using Cholesky decomposition.
MCResult simulate_paths(
    const std::vector<double>& mu,
    const std::vector<double>& sigma_flat,
    std::size_t n_sims,
    std::size_t horizon
) {
    if (mu.empty()) {
        throw std::invalid_argument("mu is empty");
    }
    std::size_t n_assets = mu.size();
    if (sigma_flat.size() != n_assets * n_assets) {
        throw std::invalid_argument("sigma_flat size mismatch");
    }
    if (n_sims == 0 || horizon == 0) {
        throw std::invalid_argument("n_sims and horizon must be positive");
    }

    std::vector<double> l = cholesky_lower(sigma_flat, n_assets);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    MCResult out;
    out.n_sims = n_sims;
    out.horizon = horizon;
    out.n_assets = n_assets;
    out.paths.resize(n_sims * horizon * n_assets, 0.0);

    for (std::size_t s = 0; s < n_sims; ++s) {
        for (std::size_t t = 0; t < horizon; ++t) {
            std::vector<double> z(n_assets, 0.0);
            for (std::size_t i = 0; i < n_assets; ++i) {
                z[i] = dist(rng);
            }
            for (std::size_t i = 0; i < n_assets; ++i) {
                double x = mu[i];
                for (std::size_t k = 0; k <= i; ++k) {
                    x += l[i * n_assets + k] * z[k];
                }
                std::size_t idx = (s * horizon + t) * n_assets + i;
                out.paths[idx] = x;
            }
        }
    }

    return out;
}

// Compute an empirical quantile for a vector of values.
static double quantile(std::vector<double> v, double q) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    double pos = q * (static_cast<double>(v.size() - 1));
    std::size_t idx = static_cast<std::size_t>(pos);
    double frac = pos - static_cast<double>(idx);
    if (idx + 1 < v.size()) {
        return v[idx] * (1.0 - frac) + v[idx + 1] * frac;
    }
    return v[idx];
}

// Summarize simulated paths into VaR/CVaR and q95.
Summary summarize_paths(
    const std::vector<double>& paths,
    std::size_t n_sims,
    std::size_t horizon,
    std::size_t n_assets,
    double alpha
) {
    if (paths.empty()) {
        throw std::invalid_argument("paths is empty");
    }
    if (n_sims == 0 || horizon == 0 || n_assets == 0) {
        throw std::invalid_argument("invalid dimensions");
    }

    std::vector<double> pnl(n_sims, 0.0);
    for (std::size_t s = 0; s < n_sims; ++s) {
        double sum = 0.0;
        for (std::size_t t = 0; t < horizon; ++t) {
            for (std::size_t i = 0; i < n_assets; ++i) {
                std::size_t idx = (s * horizon + t) * n_assets + i;
                sum += paths[idx];
            }
        }
        pnl[s] = sum;
    }

    Summary out;
    out.var = quantile(pnl, alpha);
    double tail_sum = 0.0;
    std::size_t tail_count = 0;
    for (double v : pnl) {
        if (v <= out.var) {
            tail_sum += v;
            tail_count += 1;
        }
    }
    out.cvar = tail_count ? (tail_sum / static_cast<double>(tail_count)) : 0.0;
    out.q95 = quantile(pnl, 0.95);
    return out;
}
