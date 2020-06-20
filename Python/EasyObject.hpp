#ifndef PYTHON_EASYOBJECT_HPP_
#define PYTHON_EASYOBJECT_HPP_

#include "PythonEnvironment.hpp"
#include <CXX/Extensions.hxx>
#include <CXX/Objects.hxx>
#include <object.h>
#include <opencv2/core/core.hpp>
#include <functional>
#include <utility>
#include <map>
#include <string>
#include <vector>
#include <type_traits>
#include <iostream>

namespace pyx {

class PositionalArgs;
class KeywordArgs;


template <typename E>
class EasyObjectItemRef : public E
{
public:
    EasyObjectItemRef(
            Py::Object const& boss,
            Py::Object const& key,
            Py::Object const& inner)
        : E(inner), boss(boss), key(key){}

    EasyObjectItemRef(EasyObjectItemRef const& other)
        : E(other), boss(other.boss), key(other.key){}

    virtual ~EasyObjectItemRef(){}

    auto operator =(E const& other) -> EasyObjectItemRef&
    {
        if (Py::Object(*this).ptr() == Py::Object(other).ptr())
        {
            return *this;
        }

        E::operator =(other);
        PyObject_SetItem(boss.ptr(), key.ptr(), Py::Object(other).ptr());
        return *this;
    }

    void del()
    {
        boss.delItem(key);
    }

protected:
    Py::Object boss;
    Py::Object key;

};

template <typename E>
class EasyObjectAttrRef : public E
{
public:
    EasyObjectAttrRef(Py::Object const& boss, std::string const& name, Py::Object const& inner)
        : E(inner), boss(boss), name(name){}

    EasyObjectAttrRef(EasyObjectAttrRef const& other)
        : E(other), boss(other.boss), name(other.name){}

    virtual ~EasyObjectAttrRef(){}

    auto operator=(E const& other) -> EasyObjectAttrRef&
    {
        if (Py::Object(*this).ptr() == Py::Object(other).ptr())
        {
            return *this;
        }

        E::operator =(other);
        boss.setAttr(name, *this);
        return *this;
    }

    void del()
    {
        boss.delAttr(name);
    }

protected:
    Py::Object boss;
    std::string name;
};

// Purpose:
// 1) Nice constructors that also work implicitly
// 2) Nice conversion operators
// 3) Subscript operator always available (but may lead to runtime error)
// 4) Calling member functions is simple
class EasyObject
{
public:
    EasyObject(Py::Object const& obj) : inner(obj){}
    EasyObject(PyObject* pobj) : inner(Py::new_reference_to(pobj)){}

    EasyObject(std::string const& str) : inner(Py::String(str)){}
    EasyObject(char const* str) : inner(Py::String(std::string(str))){}

    EasyObject(long long v) : inner(Py::Long(static_cast<long>(v))){}
    EasyObject(unsigned long long v) : inner(Py::Long(static_cast<long>(v))){}

    EasyObject(long v) : inner(Py::Long(v)){}
    EasyObject(unsigned long v) : inner(Py::Long(static_cast<long>(v))){}

    EasyObject(int v) : inner(Py::Long(v)){}
    EasyObject(unsigned int v) : inner(Py::Long(static_cast<long>(v))){}

    EasyObject(short v) : inner(Py::Long(v)){}
    EasyObject(unsigned short v) : inner(Py::Long(static_cast<long>(v))){}

    EasyObject(char v) : inner(Py::Long(v)){}
    EasyObject(unsigned char v) : inner(Py::Long(static_cast<long>(v))){}

    EasyObject(double v) : inner(Py::Float(v)){}
    EasyObject(float v) : inner(Py::Float(v)){}

    EasyObject(bool v) : inner(Py::Boolean(v)){}

    EasyObject(cv::InputArray arr);
    virtual ~EasyObject(){}

    auto operator [](EasyObject const& key) const -> EasyObject;
    auto operator [](char const* key) const -> EasyObject;

    auto operator [](EasyObject const& key) -> EasyObjectItemRef<EasyObject>;
    auto operator [](char const* key) -> EasyObjectItemRef<EasyObject>;

    template <typename T>
    auto operator [](T key) -> EasyObjectItemRef<EasyObject>{return (*this)[EasyObject(key)];}

    auto operator ()() const -> EasyObject;
    auto operator ()(
            PositionalArgs const& positionalParams
            ) const -> EasyObject;
    auto operator ()(
            PositionalArgs const& positionalParams,
            KeywordArgs const& kwargs
            ) const -> EasyObject;

    auto attr(std::string const& attrName) const -> EasyObject;
    auto attr(std::string const& attrName) -> EasyObjectAttrRef<EasyObject>;

    operator double() const;
    operator float() const;
    operator int() const;
    operator long() const;
    operator bool() const;
    operator std::string() const; //otherwise the subscript operator is ambiguous
    operator Py::Object() const;

    auto call(
            std::string const& funname
            ) -> pyx::EasyObject;

    auto call(
            std::string const& funname,
            PositionalArgs const& positionalParams
            ) -> pyx::EasyObject;

    auto call(
            std::string const& funname,
            PositionalArgs const& positionalParams,
            KeywordArgs const& kwargs
            ) -> pyx::EasyObject;

protected:
    Py::Object inner;
};

std::ostream& operator <<(std::ostream& os, EasyObject const& ob);

//#define PARAMETER_PACK_UNORDERED_FOREACH(expr) do {int a[]={((expr), 0)...};} while(false)

// Reason for this class: implicit user-defined conversions are single-step in C++
// By this trick, Value gets explicitly converted to EasyObject, so there is room for an extra
// implicit conversion inbetween from Value to SomeType where SomeType is known in the constructor
// of EasyObject.
class ImplicitEasyObject
{
public:
    template <typename Value>
    ImplicitEasyObject(Value value) : value(EasyObject(value)){}

    operator Py::Object() const {return value;}
private:
    Py::Object const value;
};

//---------------------------------------------------------

template <typename IterableSizeable>
auto makeTuple(IterableSizeable iterableSizable) -> Py::Tuple
{
    Py::Tuple result(iterableSizable.size());
    int i=0;
    for (auto&& arg : iterableSizable)
    {
        result.setItem(i++, ImplicitEasyObject(arg));
    }

    return result;
}

template <typename T>
auto makeTuple(std::initializer_list<T> l) -> Py::Tuple
{
    Py::Tuple result(l.size());
    int i=0;
    for (auto&& arg : l)
    {
        result.setItem(i++, ImplicitEasyObject(arg));
    }
    return result;
}

inline
auto makeTuple(std::initializer_list<ImplicitEasyObject> l) -> Py::Tuple
{
    Py::Tuple result(l.size());
    int i=0;
    for (auto&& arg : l)
    {
        result.setItem(i++, arg);
    }
    return result;
}

template <typename... Ts>
auto tuple(Ts&&... ts) -> Py::List
{
    return makeTuple({std::forward<Ts>(ts)...});
}

//-------------------------------------------------

template <typename IterableSizeable>
auto makeList(IterableSizeable iterableSizable) -> Py::List
{
    Py::List result(iterableSizable.size());
    int i=0;
    for (auto&& arg : iterableSizable)
    {
        result.setItem(i++,ImplicitEasyObject(arg));
    }

    return result;
}

template <typename T>
auto makeList(std::initializer_list<T> l) -> Py::List
{
    Py::List result(l.size());
    int i=0;
    for (auto&& arg : l)
    {
        result.setItem(i++,ImplicitEasyObject(arg));
    }
    return result;
}

inline
auto makeList(std::initializer_list<ImplicitEasyObject> l) -> Py::List
{
    Py::List result(l.size());
    int i=0;
    for (auto&& arg : l)
    {
        result.setItem(i++, arg);
    }
    return result;
}

template <typename... Ts>
auto list(Ts&&... ts) -> Py::List
{
    return makeList({std::forward<Ts>(ts)...});
}

class ImplicitEasyObjectPair
{
public:
    template <typename Key, typename Value>
    ImplicitEasyObjectPair(Key const& key, Value const& value)
        : first(EasyObject(key)), second(EasyObject(value)){}

    operator std::pair<Py::Object, Py::Object>() const {return {first,second};}

private:
    Py::Object const first;
    Py::Object const second;
};

template <typename IterablePairConvertible>
auto makeDict(IterablePairConvertible iterablePairConvertible) -> Py::Dict
{
    Py::Dict result;
    int i=0;
    for (auto&& arg : iterablePairConvertible)
    {
        std::pair<Py::Object, Py::Object> pair(arg);
        result.setItem(pair.first, pair.second);
    }
    return result;
}

template <typename T>
auto makeDict(std::initializer_list<T> l) -> Py::Dict
{
    Py::Dict result;
    int i=0;
    for (auto&& arg : l)
    {
        std::pair<Py::Object, Py::Object> pair(arg);
        result.setItem(pair.first, pair.second);
    }
    return result;
}

inline
auto makeDict(std::initializer_list<ImplicitEasyObjectPair> l) -> Py::Dict
{
    Py::Dict result;
    int i=0;
    for (auto&& arg : l)
    {
        std::pair<Py::Object, Py::Object> pair(arg);
        result.setItem(pair.first, pair.second);
    }
    return result;
}

//------------------------------
// Type for use as a function parameter, for C++ functions
// with Python-like interface (mostly wrappers around Python functions)
class PositionalArgs
{
    class SinglePositionalArg
    {
    public:
        template <typename Value>
        SinglePositionalArg(Value value) : value(value){}

        EasyObject const value;
    };

public:
    PositionalArgs(){}
    PositionalArgs(std::initializer_list<SinglePositionalArg> args);
    PositionalArgs(Py::Tuple const& m);

    operator Py::Tuple() const;
private:
    Py::Tuple objects;
};


//--------------------------------
// Type for use as a function parameter, for C++ functions
// with Python-like interface (mostly wrappers around Python functions)
// has the advantage of implicit conversion from a string representation
// such as "key1=value1, key2=value2"
// equivalent to {{"key",pyx::evalString("value1")}, {"key",pyx::evalString("value2")}}
class KeywordArgs
{
    class SingleKeywordArg
    {
    public:
        template <typename Value>
        SingleKeywordArg(std::string const& key, Value value)
            : key(key), value(value){}

        std::string const key;
        EasyObject const value;
    };

public:
    KeywordArgs(){}
    KeywordArgs(std::initializer_list<SingleKeywordArg> args);
    KeywordArgs(Py::Dict const& m);

    // Parameter needs to have the format "key1=value1, key2=value2"
    KeywordArgs(std::string const& s);

    // Needed because otherwise two-steps would be needed in implicit conversion
    // (char const*) -> std::string -> KeywordArgs
    KeywordArgs(char const* s);

    operator Py::Dict() const;

private:
    void initFromString(std::string const& s);

    Py::Dict kwargs;
};

//-----------------------------
auto moduleCall(
        std::string const& moduleName,
        std::string const& funname,
        PositionalArgs const& positionalParams = PositionalArgs(),
        KeywordArgs const& kwargs = KeywordArgs()
        ) -> EasyObject;

auto globalCall(
        std::string const& funname,
        PositionalArgs const& positionalParams = PositionalArgs(),
        KeywordArgs const& kwargs = KeywordArgs()
        ) -> EasyObject;

auto subst(std::string const& expression, PositionalArgs const& args) -> EasyObject;

template <typename... Ts>
auto subst(std::string const& expression, Ts&&... args) -> EasyObject
{
    return pyx::subst(expression, {std::forward<Ts>(args)...});
}



}

#endif /* PYTHON_EASYOBJECT_HPP_ */
