#include <Python.h>
#include "./PyCimenModuleAttr.hpp"

PyCimenObject* numpy_array(const std::vector<PyCimenObject*>&);
PyCimenObject* numpy_mean(const std::vector<PyCimenObject*>&);
PyCimenObject* numpy_median(const std::vector<PyCimenObject*>&);
PyCimenObject* numpy_std(const std::vector<PyCimenObject*>&);

class PyCimenModule : public PyCimenObject {
public:
    PyCimenModule(char* moduleName) : PyCimenObject(PyCimenObject::ObjectType::Module, nullptr) {
        this->moduleObject = PyImport_ImportModule(moduleName);
        this->scope = new PyCimenScope();
    }

    PyObject* arrFromIntArray(int* data, int n) {
    }

    inline bool isModule() const override { return true; }

    PyCimenScope* getContext() {
        return this->scope;
    }

    bool hasAttr(char* attribute_name) {
        return (bool) PyObject_HasAttrString(this->moduleObject, attribute_name);
    }

    PyCimenModuleAttr* getAttr(const char* attribute_name) {
        PyObject* pythonObject = PyObject_GetAttrString(this->moduleObject, attribute_name);
        return new PyCimenModuleAttr(pythonObject);

    }

private:
    PyObject* moduleObject;
    PyCimenScope* scope;
};
