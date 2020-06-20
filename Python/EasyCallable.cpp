#include <Python/EasyCallable.hpp>
#include <Python/PythonFunction.hpp>
#include <Python/PythonEnvironment.hpp>

using namespace pyx;

pyx::EasyCallable::
EasyCallable(Py::Callable inner)
    : inner(inner){}

pyx::EasyCallable::
EasyCallable(
        std::string const& moduleName,
        std::string const& funname)
    : inner(Py::Module(moduleName).getAttr(funname)){}

pyx::EasyCallable::
EasyCallable(
        std::string const& funname)
    : inner(pyx::globals().getAttr(funname)){}

auto pyx::EasyCallable::
operator ()(
        PositionalArgs const& positionalParams,
        KeywordArgs const& kwargs) const -> EasyObject
{
    return inner.apply(positionalParams, kwargs);
}

EasyCallable::
operator Py::Callable() const
{
    return inner;
}
