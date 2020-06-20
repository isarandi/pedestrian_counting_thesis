#ifndef PYTHON_EASYCALLABLE_HPP_
#define PYTHON_EASYCALLABLE_HPP_

#include <Python/EasyObject.hpp>
#include <Python/PythonFunction.hpp>
#include <CXX/Extensions.hxx>
#include <CXX/Objects.hxx>
#include <functional>
#include <string>

namespace pyx {

// Purpose:
//  1) has operator() so it can be naturally called in C++
//  2) has nice constructors
class EasyCallable
{
public:
    EasyCallable(Py::Callable inner);

    // The given PyFunctor type needs to be convertible to an
    // std::function<Py::Object(Py::Tuple const&, Py::Dict const&)>
    template <typename PyFunctor>
    EasyCallable(PyFunctor fun)
        : inner(
                Py::ExtensionObject<pyx::FunctionAdapter>(
                    new pyx::FunctionAdapter(fun))){}

    explicit EasyCallable(
            std::string const& moduleName,
            std::string const& functionName);
    explicit EasyCallable(
            std::string const& globalFunctionName);

    auto operator ()(
            PositionalArgs const& positionalParams = PositionalArgs(),
            KeywordArgs const& kwargs = KeywordArgs()
            ) const -> pyx::EasyObject;

    template <typename... Ts>
    auto operator ()(Ts&&... params) const -> pyx::EasyObject
    {
    	return (*this)({std::forward<Ts>(params)...});
    }

    operator Py::Callable() const;

private:
    Py::Callable inner;
};

}

#endif /* PYTHON_EASYCALLABLE_HPP_ */
