#include <Python.h>
#include <numpy/ndarrayobject.h> 
#include "./PyCimenModuleFunc.hpp"  
#include "./PyCimenNumpyArray.hpp"

PyCimenObject* numpy_array(const std::vector<PyCimenObject*>&);
PyCimenObject* numpy_mean(const std::vector<PyCimenObject*>&);
PyCimenObject* numpy_median(const std::vector<PyCimenObject*>&);
class PyCimenModule : public PyCimenObject {
public:
    PyCimenModule(char* moduleName) : PyCimenObject(PyCimenObject::ObjectType::Module, nullptr) {
        this->moduleObject = PyImport_ImportModule(moduleName);
        this->scope = new PyCimenScope();

        //array

        this->scope->define("array", new PyCimenModuleFunc("array", numpy_array, 1));
        this->scope->define("mean", new PyCimenModuleFunc("mean", numpy_mean, 1));
        this->scope->define("median", new PyCimenModuleFunc("median", numpy_median, 1));
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

PyCimenObject* numpy_mean(const std::vector<PyCimenObject*>& args) {
    PyCimenNumpyArray* arr = static_cast<PyCimenNumpyArray*>(args[0]);
    size_t size = arr->getSize();
    int sum = 0;
    for (int i = 0; i < size; i++) {
        const PyCimenInt* intElement = dynamic_cast<const PyCimenInt*>((*arr)[i]);
        if (intElement) {
            sum += intElement->getInt();
        }
    }
    double mean = static_cast<double>(sum) / size;
    return new PyCimenFloat(mean);
}

PyCimenObject* numpy_median(const std::vector<PyCimenObject*>& args) {
    PyCimenNumpyArray* arr = static_cast<PyCimenNumpyArray*>(args[0]);
    size_t size = arr->getSize();

    
    std::vector<int> values;
    for (int i = 0; i < size; i++) {
        const PyCimenInt* intElement = dynamic_cast<const PyCimenInt*>((*arr)[i]);
        if (intElement) {
            values.push_back(intElement->getInt());
        }
    }
    
    
  
    std::sort(values.begin(), values.end());

   

    double median;
    if (size % 2 == 0) {
        // Even number 
        median = (values[size/2 - 1] + values[size/2]) / 2.0;
    } else {
        // Odd number 
        median = values[size/2];
    }
    
    return new PyCimenFloat(median);
}

