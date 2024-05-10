#include <Python.h>
#include <numpy/ndarrayobject.h> 

class PyCimenModule : public PyCimenObject {
public:
    PyCimenModule(char* moduleName) : PyCimenObject(PyCimenObject::ObjectType::Module, nullptr) {
        this->moduleObject = PyImport_ImportModule(moduleName);
        this->scope = new PyCimenScope();

        //array

        //std::vector<AstNode*> params = {PyCimenList()};

        //PyCimenObject* array_func = new PyCimenFunction()

        //this->scope->define("array");
    }

    PyObject* arrFromIntArray(int* data, int n) {
        npy_intp dims[1] = {n}; 
        return PyArray_SimpleNewFromData(1, dims, NPY_INT, data); 
    }

    inline bool isModule() const override { return true; }

private:
    PyObject* moduleObject;
    PyCimenScope* scope;
};