#pragma once

#include "../value/PyCimenCallable.hpp"
#include <iostream>
#include <string>

class PyCimenStr;

class Input : public PyCimenCallable {
public:
    explicit Input() : PyCimenCallable(ObjectType::Builtin, nullptr){}
    
    size_t arity() override { return 0; }
    
    PyCimenObject* call(Interpreter* interpreter, const std::vector<PyCimenObject*>& args) override {
        
        if(!args.empty()) {
            throw std::runtime_error("input() takes no arguments");        
        }
        std::string inputStr;
        std::getline(std::cin, inputStr);
        
        return new PyCimenStr(inputStr);
    }
    
    void write(std::ostream& out) const override {
        out << "<builtin  function input>";
    }
};
