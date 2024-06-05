#pragma once

#include "./PyCimenCallable.hpp"
#include "../exceptions/runtime_exceptions.hpp"
#include "./PyCimenNone.hpp"

class PyCimenFunction : public PyCimenCallable {
public:
    explicit PyCimenFunction(
                FunctionNode* declaration, 
                class PyCimenScope* closure, 
                PyCimenInstance* bound = nullptr
    ) : PyCimenCallable(ObjectType::Func, closure), 
        declaration(declaration), 
        bound(bound) {}
    
    inline bool isFunc() const override { return true; }
    inline bool isTruthy() const override { return true; }
    
    PyCimenFunction* bind(PyCimenInstance* instance) {
        return new PyCimenFunction(this->declaration, this->getContext(), instance);
    }
    
    PyCimenObject* call(Interpreter* interpreter, const std::vector<PyCimenObject*>& args) override {

        closure = this->getContext();
        fnCallEnv = new class PyCimenScope(closure);
        interpreter->pushContext(fnCallEnv);
        
        const std::vector<AstNode*>& params = this->declaration->getParams();
        
        unsigned short bounded = (this->bound ? 1 : 0);
        
        size_t expectedArgs = params.size() - bounded;
        
        if (args.size() != expectedArgs) {
            throw std::runtime_error("error on positional args");
        }
        
        size_t argIndex = 0;
        if (this->bound) { // If bound, pass 'self' as the first argument
            NameNode* Self = static_cast<NameNode*>(params[argIndex++]);
            const std::string& argName = Self->getLexeme();
            interpreter->defineOnContext(argName, this->bound);
        }
        
        if(expectedArgs) {
            // Loop through parameters and define them in the context
            for(; argIndex < params.size(); ++argIndex) {
                NameNode* param = static_cast<NameNode*>(params[argIndex]);
                const std::string& argName = param->getLexeme();
                interpreter->defineOnContext(argName, args[argIndex - (bounded)]);
            }
        }
        
        PyCimenObject* retVal = new PyCimenNone();
        try {
            // Visit the body of the function, switching back to the interpreter's context
            retVal = declaration->getBody()->accept(interpreter);
        } catch(ReturnException& re) {
            delete retVal;
            retVal = re.value;
        }
        
        interpreter->popContext();
        return retVal;
    }
    
    size_t arity() override {
        return (bound ? 1 : 0) + (*declaration).getParams().size();
    }
    
    void write(std::ostream& out) const override {
        out << "<function \'" << (*declaration).getName() << "\'>";
    }
    
private:
    FunctionNode* declaration = nullptr;
    PyCimenInstance* bound = nullptr;

    class PyCimenScope* closure;
    class PyCimenScope* fnCallEnv;
    
    void deleteData() override {
        delete declaration;
        delete bound;
    }
};
