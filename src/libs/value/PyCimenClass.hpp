#pragma once

#include "./PyCimenFunction.hpp"
#include "./PyCimenInstance.hpp"

class PyCimenClass : public PyCimenCallable {
public:
    std::string kname;
    
    explicit PyCimenClass(const std::string& kname, class PyCimenScope* closure) 
        : PyCimenCallable(ObjectType::Klass, closure), kname(kname){}
    
    inline bool isKlass() const override { return true; }
    inline bool isTruthy() const override { return true; }    

    PyCimenObject* call(Interpreter* interpreter, const std::vector<PyCimenObject*>& args) override {
        
        PyCimenInstance* instance = new PyCimenInstance(this);
        
        PyCimenFunction* initTarget = findMethod("__init__");
        
        if(initTarget) {
            initTarget->bind(instance)->call(interpreter, args);
        }
        return instance;
    }
    
    PyCimenFunction* findMethod(const char* methodName) {
        try {
            PyCimenObject* target = this->getContext()->get(methodName);
            PyCimenFunction* method = static_cast<PyCimenFunction*>(target);
            return method;
            
        } catch(std::runtime_error&) {
            return nullptr;
        }
    }
    
    size_t arity() override {
        PyCimenFunction* initTarget = findMethod("__init__");
        return (initTarget ? initTarget->arity() : 0);
    }
    
    void write(std::ostream& out) const override {
        out << "<class \'" << kname << "\'>";
    }

private:
    void deleteData() override {
        delete getContext();
    }
};