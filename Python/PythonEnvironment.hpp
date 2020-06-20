#ifndef PYTHON_PYTHONENVIRONMENT_HPP_
#define PYTHON_PYTHONENVIRONMENT_HPP_

#include <CXX/Python2/Objects.hxx>
#include <string>

namespace pyx {

auto run(std::string const& script) -> Py::Object;
auto eval(std::string const& expression) -> Py::Object;
auto import(std::string const& moduleName) -> Py::Module;

auto globals() -> Py::Dict;

void initialize();
void finalize();
}
#endif /* PYTHON_PYTHONENVIRONMENT_HPP_ */
