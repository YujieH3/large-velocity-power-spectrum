#include "test.h"
#include <iostream>

py::array_t<double> echo2darray(py::array_t<double> array) {
	size_t n = array.shape(0);
	size_t m = array.shape(1);

	py::buffer_info buf = array.request();
	double *ptr = (double *) buf.ptr;
	
	for (size_t i = 0; i < n; i++) {
		std::cout << "(" << ptr[i*m];
		for (size_t j = 1; j < m; j++) {
			std::cout << ", " << ptr[i*m + j];
		}
		std::cout << ")\n";
	}

	return array;
}

