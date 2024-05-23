#pragma once

#include "./PyCimenObject.hpp"

class PyCimenFloat : public PyCimenObject {
public:
    explicit PyCimenFloat(const std::string& v);
    explicit PyCimenFloat(llf v);
        
    inline bool isFloat() const override { return true; } 
    inline bool isTruthy() const override { return getFloat() != 0.0; }
    
    PyCimenObject* operator+(const PyCimenObject& other) const override;
    PyCimenObject* operator-(const PyCimenObject& other) const override;
    PyCimenObject* operator*(const PyCimenObject& other) const override;
    PyCimenObject* operator/(const PyCimenObject& other) const override;
    PyCimenObject* operator==(const PyCimenObject& other) const override;
    PyCimenObject* operator<(const PyCimenObject& other) const override;
    PyCimenObject* operator>(const PyCimenObject& other) const override;
    PyCimenObject* operator-() const override;
    PyCimenObject* operator!() const override;
    
    llf getFloat() const;
    void write(std::ostream& out) const override;

    PyObject* getPythonObject() const override;
        
private:
    const llf* getFloatData() const;
    void deleteData() override;
};
