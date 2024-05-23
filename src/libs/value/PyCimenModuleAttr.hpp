PyCimenObject* parsePythonObject(PyObject* pythonObject) {
    if (PyFloat_Check(pythonObject)) {
        return new PyCimenFloat((llf)PyFloat_AsDouble(pythonObject));
    } else if (strstr(pythonObject->ob_type->tp_name, "numpy.ndarray")) {
        std::cout << "toListMethod" << std::endl; 
        PyObject* list = PyObject_CallMethod(pythonObject, "tolist", nullptr);
        return parsePythonObject(list);
    } else if (PyList_Check(pythonObject)) {
        size_t size = PyList_Size(pythonObject);

        PyCimenList* list = new PyCimenList(std::vector<PyCimenObject*>());

        for (size_t i = 0; i < size; ++i) {
            PyObject* item = PyList_GetItem(pythonObject, i);

            PyCimenObject* pyCimenObject = parsePythonObject(item);
            list->append(pyCimenObject);
        }

        return list;
    } else if (PyLong_Check(pythonObject)) {
        return new PyCimenInt((ll) PyLong_AsLong(pythonObject));
    }
    else {
        return new PyCimenNone();
    }
} 

class PyCimenModuleAttr : public PyCimenCallable {
public:

    enum class AttrType {
        Function,
        Unknown
    };

    size_t arity() override {
        return 1;
    }


    inline bool isModuleAttr() const override {
        return true;
    }

    PyCimenModuleAttr(PyObject* pythonObject)
        : PyCimenCallable(ObjectType::ModuleAttr, nullptr){

        this->pythonObject = pythonObject;

        const char* typeName = pythonObject->ob_type->tp_name; 

        if(strstr(typeName, "function") 
                || strstr(typeName, "type") 
                || strstr(typeName, "method")) {
            this->attrType = AttrType::Function; 
        } else {
            std::cout << "This is a " << typeName << std::endl;
            this->attrType = AttrType::Unknown; 
        }
    }

    PyCimenObject* getAttr(const char* name) const {
        PyObject* attr = PyObject_GetAttrString(this->pythonObject, name);

        std::cout << "Got attr " << name << ": " << attr << std::endl;
        return new PyCimenModuleAttr(attr);
    }

    PyCimenObject* call(Interpreter* interpreter, const std::vector<PyCimenObject*>& args) override {

        if (this->attrType == AttrType::Function) {

            PyObject* argsTuple = PyTuple_New(args.size());

            for (int i = 0; i < args.size(); ++i) {

                PyObject* pythonObject = args[i]->getPythonObject();

                PyTuple_SetItem(argsTuple, i, pythonObject);

            }
            
            /* PyObject* result = PyObject_CallFunctionObjArgs(this->pythonObject, args[0]->getPythonObject()); */
            PyObject* result = PyObject_CallObject(this->pythonObject, argsTuple);

            std::cout << "Result: " << result << std::endl;
            std::cout << "Result type: " << result->ob_type->tp_name << std::endl;

            if (result == nullptr) {
                PyErr_Print();
            }

            return new PyCimenModuleAttr(result);

        }
        else {
            std::cout << "This is not a function! This is a: " << std::endl;
        }

        return new PyCimenNone();
    }

    void write(std::ostream& out) const override {
        PyObject* representation = PyObject_Repr(this->pythonObject);
        out << PyUnicode_AsUTF8(representation);
    }

private: 
    PyObject* pythonObject;
    AttrType attrType;
};


