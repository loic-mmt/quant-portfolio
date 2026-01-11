#include "pybind11/detail/common.h"
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

// TODO: include mc_engine header when you split declarations.
// For now, keep declarations in this TU or move to a header.

struct MCResult {
    std::vector<double> paths;
    std::size_t n_sims = 0;
    std:: size_t horizon = 0;
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

namespace py = pybind11;

PYBIND11_MODULE(quant_mc, m) {
    m.doc() = "Minimal MC engine building";

    py::class_<MCResult>(m, "MCResult")
        .def_readonly("paths", &MCResult::paths)
        .def_readonly("n_sims", &MCResult::n_sims)
        .def_readonly("horizon", &MCResult::horizon)
        .def_readonly("n_assets", &MCResult::n_assets);

    py::class_<Summary>(m, "Summary")
        .def_readonly("var", &Summary::var)
        .def_readonly("cvar", &Summary::cvar)
        .def_readonly("q95", &Summary::q95);
    
    m.def("simulate_paths", &simulate_paths, "Simulate MC paths");
    m.def("summarize_paths", &summarize_paths, "Summarize MC paths");
}