#pragma once

#include "../value/PyCimenObject.hpp"
#include "../scope/scope.hpp"

class PyCimenClass;

class PyCimenInstance : public PyCimenObject {
public:
    explicit PyCimenInstance(PyCimenClass* klass);
    inline bool isInstance() const { return true; }
    inline bool isTruthy() const override { return true; }
    
    PyCimenObject* operator==(const PyCimenObject&) const override;
    
    void define(const std::string& name, PyCimenObject* value) {
        return this->fields->define(name, value);
    }
    PyCimenScope* getContext() { return fields; }

    void write(std::ostream& out) const override;

private:
    PyCimenClass* base;
    PyCimenScope* fields;
    
    void deleteData() override {
        delete getContext();
    }
};