#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// unsigned int fibinacci(const unsigned int n);

namespace py = pybind11;

py::array_t<double> echo2darray(py::array_t<double> array);

PYBIND11_MODULE(test_module, mod) {
    mod.def("echo2darray", &echo2darray, "Return a 2d array same as input.");
}
// "function name in python", &function name in c++, "docstring"