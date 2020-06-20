#include <Python/PythonFunction.hpp>
#include <CXX/Extensions.hxx>
#include <CXX/Objects.hxx>
#include <CXX/Python2/PythonType.hxx>

using namespace pyx;

bool pyx::FunctionAdapter::initialized = false;

pyx::FunctionAdapter::
FunctionAdapter(std::function<Py::Object(Py::Tuple const&, Py::Dict const&)> stdFun)
    : stdFun(stdFun)
{
    init_type();
}

auto pyx::FunctionAdapter::
call(
        Py::Object const& positionalArgs,
        Py::Object const& keywordArgs
        )  -> Py::Object
{
    return stdFun(
            positionalArgs.isNone() ? Py::Tuple() : Py::Tuple(positionalArgs),
            keywordArgs.isNone() ? Py::Dict() : Py::Dict(keywordArgs));
}

void pyx::FunctionAdapter::
init_type()
{
    if (!initialized)
    {
        behaviors().name("CppFunctionAdapter");
        behaviors().supportCall();
        initialized = true;
    }
}

