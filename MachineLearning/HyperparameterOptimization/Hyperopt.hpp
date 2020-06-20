#ifndef MACHINELEARNING_HYPERPARAMETEROPTIMIZATION_HYPEROPT_HPP_
#define MACHINELEARNING_HYPERPARAMETEROPTIMIZATION_HYPEROPT_HPP_

#include "Python/EasyCallable.hpp"
#include "Python/EasyObject.hpp"
#include "Python/PythonFunction.hpp"
#include "Python/PythonEnvironment.hpp"
#include "cvextra/strings.hpp"
#include <tuple>
#include <iostream>
#include <type_traits>

namespace hyperopt
{

template <typename PyFunctor>
static auto fminPy(
        Py::Object const& space,
        PyFunctor objective,
        pyx::KeywordArgs const& kwargs
        ) -> Py::Object
{
    auto bestVariableSubstitution = pyx::moduleCall("hyperopt" , "fmin", {pyx::EasyCallable(objective), space}, kwargs);
    auto bestArguments = pyx::moduleCall("hyperopt", "space_eval", {space, bestVariableSubstitution});
    return bestArguments;
}

template <typename PyFunctor>
static auto fmin(
        std::string const& spaceStr,
        PyFunctor objective,
        std::string const& otherArgs
        ) -> Py::Object
{
    pyx::import("hyperopt");
    pyx::run("import numpy");
    pyx::run("import math");
    pyx::run("from hyperopt import tpe, hp");
    auto space = pyx::eval(cvx::str::replace(spaceStr, "\n", "\\\n"));
    return fminPy(space, objective, otherArgs);
}

template <typename PyFunctor>
static auto fminSimple(
        std::string const& spaceStr,
        PyFunctor objective,
        std::string const& otherArgs
        ) -> Py::Object
{
    pyx::import("hyperopt");
    pyx::run("import numpy");
    pyx::run("import math");
    pyx::run("from hyperopt import tpe, hp");

    pyx::EasyCallable fullObjective{
        [objective](Py::Tuple const& posi, Py::Dict const& keyw)
        {
            std::cout << "Trying " << posi[0] << "... ";
            std::cout.flush();

            auto result = Py::Float{objective(posi[0])};
            std::cout << result << std::endl;
            return result;
        }
    };

    auto space = pyx::eval(cvx::str::replace(spaceStr, "\n", "\\\n"));
    return fminPy(space, fullObjective, otherArgs);
}

template <typename R, typename... Args>
static auto fminCpp(
        std::string const& searchSpaceDescription,
        std::function<R(Args...)> objective,
        std::string const& kwargs
        ) -> std::tuple<typename std::decay<Args>::type...>
{
    pyx::import("hyperopt");
    pyx::run("import numpy");
    pyx::run("import math");
    pyx::run("from hyperopt import tpe, hp");

    pyx::EasyCallable objectivePy{
        [objective](Py::Tuple const& posi, Py::Dict const& keyw)
        {
            return pyx::callCppFunctionWithPythonArguments(objective, posi[0]);
        }
    };

    return pyx::asDecayedCppTuple<Args...>(
            fminPy(
                    pyx::eval(searchSpaceDescription),
                    objectivePy,
                    kwargs));
}

template <typename R, typename... Args>
static auto fminCpp2(
        std::string const& searchSpaceDescription,
        R (*objective)(Args...),
        std::string const& kwargs
        ) -> std::tuple<typename std::decay<Args>::type...>
{
    return fminCpp(searchSpaceDescription, std::function<R(Args...)>(objective), kwargs);
}


} /* namespace crowd */

#endif /* MACHINELEARNING_HYPERPARAMETEROPTIMIZATION_HYPEROPT_HPP_ */
