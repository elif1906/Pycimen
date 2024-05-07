#include <Python.h>

class PyCimenModule : public PyCimenObject {
public:
    PyCimenModule(char* moduleName) : PyCimenObject(PyCimenObject::ObjectType::Module, nullptr) {
        this->moduleObject = PyImport_ImportModule(moduleName);
        this->scope = new PyCimenScope();
    }

    PyObject* arrFromIntArray(int* data, int n) {
        return PyArray_SimpleNewFromData(1, &n, NPY_INT);
    }

    inline bool isModule() const override { return true; }

private:
    PyObject* moduleObject;
    PyCimenScope* scope;
};