#include <vector>
#include <cstddef>

// Minimal MC engine skeleton.
// TODO: add RNG, distributions (gaussian/t), and math utilities.

struct MCResult {
    std::vector<double> paths;
    std::size_t n_sims = 0;
    std::size_t horizon = 0;
    std::size_t n_assets = 0;
}

MCResult simulate_paths(
    const std::vector<double>& mu,
    const std::vector<double>& sigma_flat,
    std::size_t n_sims,
    std::size_t horizon
) {
    // TODO: implement multivariate simulation.
    
    MCResult out;
    out.n_sims = n_sims;
    out.horizon = horizon;
    out.n_assets = mu.size()
    out.paths
}