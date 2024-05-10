#include <Python.h>
#include <numpy/ndarrayobject.h> 
#include "./PyCimenModuleFunc.hpp"  
#include "./PyCimenNumpyArray.hpp"

PyCimenObject* numpy_array(const std::vector<PyCimenObject*>&);

class PyCimenModule : public PyCimenObject {
public:
    PyCimenModule(char* moduleName) : PyCimenObject(PyCimenObject::ObjectType::Module, nullptr) {
        this->moduleObject = PyImport_ImportModule(moduleName);
        this->scope = new PyCimenScope();

        //array

        this->scope->define("array", new PyCimenModuleFunc("array", numpy_array, 1));
    }

    PyObject* arrFromIntArray(int* data, int n) {
    }

    inline bool isModule() const override { return true; }

    PyCimenScope* getContext() {
        return this->scope;
    }

private:
    PyObject* moduleObject;
    PyCimenScope* scope;
};

PyCimenObject* numpy_array(const std::vector<PyCimenObject*>& args) {

    import_array();

    PyCimenList* pycimen_list = static_cast<PyCimenList*>(args[0]);

    size_t size = pycimen_list->size();

    npy_intp dims[1]; 

    dims[0] = size;

    int c_array[size];

    for(int i = 0; i < size; ++i) {
        const PyCimenObject* element = (*pycimen_list)[i];
        const PyCimenInt* intElement = dynamic_cast<const PyCimenInt*>(element); // use dynamic_cast instead of static_cast
        const ll val = intElement->getInt();
        c_array[i] = val;
    }

    auto numpy_array =  PyArray_SimpleNewFromData(1, dims, NPY_INT, c_array); 

    for(int i = 0; i < size; ++i){
        auto data_ptr = (int*)PyArray_DATA(numpy_array);
    }

    auto pycimennparray = new PyCimenNumpyArray((int*)PyArray_DATA(numpy_array), size); 

    return pycimennparray;
}