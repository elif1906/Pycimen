#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>

#include <Python.h>

using ll = long long int;
using llf = long double;

class PyCimenObject {
public:
    enum class ObjectType {
        None, 
        Int, Float,
        Boolean, String,
        List, Func, 
        Klass, Instance,
        Builtin,
        Module, ModuleAttr,
    };
    PyCimenObject(ObjectType type, void* data = nullptr) 
        : type(type), data(data) {}
        
    ~PyCimenObject() {
        deleteData();
    }
    
    virtual std::string toString() const {return ""; }
    virtual inline bool isInt() const { return false; }
    virtual inline bool isFloat() const { return false; }
    virtual inline bool isStr() const { return false; }
    virtual inline bool isBool() const { return false; }
    virtual inline bool isList() const { return false; }
    virtual inline bool isFunc() const { return false; }
    virtual inline bool isKlass() const { return false; }
    virtual inline bool isInstance() const { return false; }
    virtual inline bool isNone() const { return false; }
    virtual inline bool isCallable() const { return false; }
    virtual inline bool isModule() const { return false; }
    virtual inline bool isModuleAttr() const { return false; }
    virtual inline bool isModuleFunc() const { return false; }
    virtual inline bool isNumpyArray() const { return false; }

    virtual PyObject* getPythonObject() const { return nullptr; }

    virtual inline bool isTruthy() const { 
        throw std::runtime_error("Yet not evaluatable object.");
    }
    virtual PyCimenObject* operator+(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for +.");
    }
    virtual PyCimenObject* operator-(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for -.");
    }
    virtual PyCimenObject* operator*(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for *.");
    }
    virtual PyCimenObject* operator/(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for /.");
    }
    virtual PyCimenObject* __intdiv__(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for //.");
    }
    virtual PyCimenObject* operator%(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for %.");
    }
    virtual PyCimenObject* operator&(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for &.");
    }
    virtual PyCimenObject* operator|(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for |.");
    }
    virtual PyCimenObject* operator^(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for ^.");
    }
    virtual PyCimenObject* operator<<(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for <<.");
    }
    virtual PyCimenObject* operator>>(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for >>.");
    }
    virtual PyCimenObject* operator==(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for ==.");
    }
    virtual PyCimenObject* operator<(const PyCimenObject& other) const {
        throw std::runtime_error("Unsupported operands for <.");
    }
    virtual PyCimenObject* operator>(const PyCimenObject& other) const {
        std::cout << other << std::endl;
        throw std::runtime_error("Unsupported operands for >.");

    }
    virtual PyCimenObject* operator-() const {
        throw std::runtime_error("Unsupported operands for unary -.");
    }
    virtual PyCimenObject* operator~() const {
        throw std::runtime_error("Unsupported operands for unary ~.");
    }
    virtual PyCimenObject* operator!() const {
        throw std::runtime_error("Unsupported operands for unary !.");
    }
        
    ObjectType getType() const {
        return type;
    }
    
    void incRefCount() { ++rc; }
    void decRefCount() { if(rc > 0) --rc; }
    int getRefCount() const { return rc; }
    
    friend std::ostream& operator<<(std::ostream& out, const PyCimenObject& value) {
        value.write(out);
        return out;
    }
    
    virtual void write(std::ostream& out) const {
        throw std::runtime_error("Yet not printable object.");
    }
    
protected:
    // TODO: move Scope here
    void* data;
    ObjectType type;
    int rc; // reference counting
    virtual void deleteData() {}
};
