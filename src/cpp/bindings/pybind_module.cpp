#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mc_engine.h"

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
