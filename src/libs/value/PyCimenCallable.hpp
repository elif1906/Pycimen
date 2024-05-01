#pragma once

#include "../interpreter/interpreter.hpp"

class PyCimenCallable : public PyCimenObject {
public:   
    explicit PyCimenCallable(ObjectType type, PyCimenScope* context) 
        : PyCimenObject(type), PyCimenScope(context){}

    inline bool isCallable() const override { return true; }

    virtual size_t arity() = 0;
    virtual PyCimenObject* call(Interpreter*, const std::vector<PyCimenObject*>&) = 0;
    PyCimenScope* getContext() { return this->PyCimenScope; }

    PyCimenScope* PyCimenScope;
};
