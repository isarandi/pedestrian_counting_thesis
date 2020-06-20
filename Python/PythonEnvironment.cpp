#include <CXX/Python2/Exception.hxx>
#include <CXX/Python2/Objects.hxx>
#include <import.h>
#include <moduleobject.h>
#include <object.h>
#include <pyerrors.h>
#include <pythonrun.h>
#include <Python/PythonEnvironment.hpp>

using namespace std;
using namespace pyx;

bool pythonInitialized = false;

void pyx::
initialize()
{
    if (pythonInitialized)
        return;

    Py_Initialize();
    pythonInitialized = true;

    if (PyErr_Occurred())
    {
        PyObject* ptype;
        PyObject* pvalue;
        PyObject* ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        Py::Object type = Py::asObject(ptype);
        Py::Object value = Py::asObject(pvalue);
        Py::Object traceback = Py::asObject(ptraceback);
        throw Py::Exception(Py::String(value).as_std_string());
    }

    pyx::import("__main__");
}

void pyx::
finalize()
{
    Py_Finalize();
    pythonInitialized = false;
}

auto pyx::
run(const std::string& script) -> Py::Object
{
    pyx::initialize();
    string str = script + "\n";
    PyObject* returnValue =
            PyRun_String(
                str.c_str(),
                Py_file_input,
                globals().ptr(),
                nullptr);

    if (returnValue != nullptr)
    {
        return Py::asObject(returnValue);
    } else {
        return Py::Object();
    }
}

auto pyx::
import(const std::string& moduleName) -> Py::Module
{
    pyx::initialize();
    return Py::Module(PyImport_ImportModule(moduleName.c_str()), true);
}

auto pyx::
globals() -> Py::Dict
{
    pyx::initialize();
    return Py::Module("__main__").getDict();
}

auto pyx::
eval(const std::string& expression) -> Py::Object
{
    pyx::initialize();
    Py::Object code =
            Py::asObject(
                Py_CompileString(
                   expression.c_str(),
                   "PythonEnvironment.cpp",
                   Py_eval_input));

    return
        Py::asObject(
            PyEval_EvalCode(
                    reinterpret_cast<PyCodeObject*>(code.ptr()),
                    globals().ptr(),
                    Py::Dict().ptr()));
}


