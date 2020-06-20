#include <Python/EasyCallable.hpp>
#include <Python/EasyObject.hpp>
#include <Python/PythonFunction.hpp>
#include <Python/PythonEnvironment.hpp>
#include <cvextra/strings.hpp>
#include <opencv2/core/mat.hpp>
#include <utility>

using namespace pyx;

std::ostream& pyx::
operator <<(std::ostream& os, EasyObject const& ob)
{
    return os << Py::Object(ob);
}

auto pyx::
moduleCall(
        std::string const& moduleName,
        std::string const& funname,
        PositionalArgs const& positionalParams,
        KeywordArgs const& kwargs
        ) -> pyx::EasyObject
{
    return EasyCallable(moduleName, funname)(positionalParams, kwargs);
}

auto pyx::
globalCall(
        std::string const& funname,
        PositionalArgs const& positionalParams,
        KeywordArgs const& kwargs
        ) -> pyx::EasyObject
{
    return EasyCallable(funname)(positionalParams, kwargs);
}

//=======================================================
pyx::EasyObject::
EasyObject(cv::InputArray arr)
{
    inner = Py::List();
    //pyx::moduleCall("numpy", "array", )
    Py::List list = inner;
    for (double val : cv::Mat1d(arr.getMat()))
    {
        list.append(Py::Float(val));
    }
}

//---------------------------------------------------
pyx::EasyObject::
operator double() const
{
    return Py::Float(inner);
}

pyx::EasyObject::
operator float() const
{
    return Py::Float(inner);
}

pyx::EasyObject::
operator int() const
{
    return static_cast<long>(Py::Long(inner));
}

pyx::EasyObject::
operator long() const
{
    return Py::Long(inner);
}

pyx::EasyObject::
operator bool() const
{
    return static_cast<bool>(Py::Boolean(inner));
}

pyx::EasyObject::
operator std::string() const
{
    return Py::String(inner).as_std_string();
}

pyx::EasyObject::
operator Py::Object() const {return inner;}

//---------------------------------------------------
auto pyx::EasyObject::
operator [](EasyObject const& key) const -> EasyObject
{
    return EasyObject(inner.getItem(key));
}

auto pyx::EasyObject::
operator [](char const* key) const -> EasyObject
{
    return EasyObject(inner.getItem(Py::String(key)));
}

//---------------------------------------------------

auto pyx::EasyObject::
operator()() const -> EasyObject
{
    return Py::Callable(inner).apply(Py::Tuple(), Py::Dict());
}

auto pyx::EasyObject::
operator()(
        PositionalArgs const& positionalParams
        ) const -> EasyObject
{
    return Py::Callable(inner).apply(positionalParams, Py::Dict());
}

auto pyx::EasyObject::
operator()(
        PositionalArgs const& positionalParams,
        KeywordArgs const& kwargs
        ) const -> EasyObject
{
    return Py::Callable(inner).apply(positionalParams, kwargs);
}

//--------------------------------------------------

auto pyx::EasyObject::
attr(std::string const& attrName) const -> EasyObject
{
    return inner.getAttr(attrName);
}

auto pyx::EasyObject::operator [](EasyObject const& key) -> EasyObjectItemRef<EasyObject>
{
    return {*this, key, inner.getItem(key)};
}

auto pyx::EasyObject::operator [](char const* key) -> EasyObjectItemRef<EasyObject>
{
    return {*this, Py::String(key), inner.getItem(Py::String(key))};
}

auto pyx::EasyObject::attr(std::string const& attrName) -> EasyObjectAttrRef<EasyObject>
{
    return {*this, attrName, inner.getAttr(attrName)};
}

//----------------------------------------------------

auto pyx::EasyObject::
call(
        std::string const& funname,
        PositionalArgs const& positionalParams,
        KeywordArgs const& kwargs) -> EasyObject
{
    return inner.callMemberFunction(funname, positionalParams, kwargs);
}

auto pyx::EasyObject::
call(
        std::string const& funname,
        PositionalArgs const& positionalParams) -> EasyObject
{
    return call(funname, positionalParams, KeywordArgs());
}

auto pyx::EasyObject::
call(
        std::string const& funname) -> EasyObject
{
    return call(funname, PositionalArgs(), KeywordArgs());
}

//=======================================================

pyx::PositionalArgs::
PositionalArgs(std::initializer_list<SinglePositionalArg> args)
    : objects(args.size())
{
    int i=0;
    for (auto&& arg : args)
    {
        objects.setItem(i++, arg.value);
    }
}

pyx::PositionalArgs::
PositionalArgs(Py::Tuple const& t) : objects(t){}

pyx::PositionalArgs::
operator Py::Tuple() const
{
    return objects;
}

//=======================================================

pyx::KeywordArgs::
KeywordArgs(std::initializer_list<SingleKeywordArg> args)
{
    for (auto&& arg : args)
    {
        kwargs.setItem(arg.key, arg.value);
    }
}

pyx::KeywordArgs::KeywordArgs(Py::Dict const& dict) : kwargs(dict){}

pyx::KeywordArgs::
KeywordArgs(std::string const& s)
{
    initFromString(s);
}

pyx::KeywordArgs::
KeywordArgs(char const* s)
{
    initFromString(s);
}

void pyx::KeywordArgs::
initFromString(std::string const& s)
{
    if (s != "")
    {
        kwargs = pyx::eval(cvx::str::format("dict(%s)", s));
    }
}

pyx::KeywordArgs::
operator Py::Dict() const
{
    return kwargs;
}

//==========================================================

auto pyx::
subst(std::string const& expression, PositionalArgs const& args) -> EasyObject
{
    std::stringstream ss;
    auto tuple = ((Py::Tuple)args);
    int nArgs = tuple.size();

    if (nArgs == 0)
    {
        return pyx::eval(expression);
    }

    for (int i=0; i<nArgs; ++i)
    {
        ss << "_" << i;
        if (i < nArgs-1)
        {
            ss << ",";
        }
    }
    pyx::run("import pickle");
    std::string lambdaCode = "lambda ("+ss.str()+") : " + expression;
    EasyObject lambdaObject = pyx::eval(lambdaCode);
    return lambdaObject({tuple});
}



