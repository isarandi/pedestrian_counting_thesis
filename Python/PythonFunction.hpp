#ifndef PYTHON_PYTHONFUNCTION_HPP_
#define PYTHON_PYTHONFUNCTION_HPP_

#include "EasyObject.hpp"
#include "PythonEnvironment.hpp"
#include <CXX/Extensions.hxx>
#include <CXX/Objects.hxx>
#include <cstddef>
#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>

namespace pyx {

namespace details {

    // Metaprogramming: generate increasing indices
    template <std::size_t... Is>
    struct indices {};

    template <std::size_t N, std::size_t... Is>
    struct build_indices : build_indices<N-1, N-1, Is...> {};

    template <std::size_t... Is>
    struct build_indices<0, Is...> : indices<Is...> {};
    //----

    template <typename R, typename... Ts, std::size_t... Is>
    auto _call(
            std::function<R(Ts...)> fun,
            Py::Tuple const& positionalArgs,
            details::indices<Is...> is
            ) -> Py::Object
    {
        return pyx::EasyObject(fun(static_cast<Ts>(pyx::EasyObject(positionalArgs[Is]))...));
    }

    template <typename... Ts, std::size_t... Is>
    auto _asCppTuple(pyx::EasyObject const& pyIndexable, indices<Is...> is) -> std::tuple<Ts...>
    {
        return std::make_tuple(static_cast<Ts>(pyIndexable[Is])...);
    }
}

// Wrapper around std::function, so it can be used in the Python universe
// This is the memory content of the object itself, unlike Py::Object which is actually
// the abstraction of an object _reference_.
class FunctionAdapter : public Py::PythonExtension<FunctionAdapter>
{
public:
    FunctionAdapter(
            std::function<Py::Object(Py::Tuple const&, Py::Dict const&)> stdFun);
    virtual ~FunctionAdapter(){}

    // override
    virtual auto call(
            Py::Object const& positionalArgs,
            Py::Object const& keywordArgs
            )  -> Py::Object;

    static void init_type();

private:
    std::function<Py::Object(Py::Tuple const&, Py::Dict const&)> stdFun;
    static bool initialized;
};

// Convert an indexable Python object to a C++ tuple
template <typename... Ts>
auto asCppTuple(EasyObject const& pyIndexable) -> std::tuple<Ts...>
{
    return details::_asCppTuple<Ts...>(pyIndexable, details::build_indices<sizeof...(Ts)>());
}

template <typename... Ts>
auto asDecayedCppTuple(EasyObject const& pyIndexable) -> std::tuple<typename std::decay<Ts>::type...>
{
    return asCppTuple<typename std::decay<Ts>::type...>(pyIndexable);
}

template <typename R, typename... Ts>
auto callCppFunctionWithPythonArguments(
        std::function<R(Ts...)> fun,
        Py::Tuple const& positionalArgs
        ) -> Py::Object
{
    return details::_call(fun, positionalArgs, details::build_indices<sizeof...(Ts)>());
}



}

#endif /* PYTHON_PYTHONFUNCTION_HPP_ */
