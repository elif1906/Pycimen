#pragma once

#include "./PyCimenObject.hpp"

class PyCimenInt : public PyCimenObject {
public:
    explicit PyCimenInt(const std::string& v);
    explicit PyCimenInt(ll v);

    inline bool isInt() const override { return true; }
    inline bool isTruthy() const override { return getInt() != 0L; }
    
    PyCimenObject* operator+(const PyCimenObject& other) const override;
    PyCimenObject* operator-(const PyCimenObject& other) const override;
    PyCimenObject* operator*(const PyCimenObject& other) const override;
    PyCimenObject* operator/(const PyCimenObject& other) const override;
    PyCimenObject* __intdiv__(const PyCimenObject& other) const override;
    PyCimenObject* operator%(const PyCimenObject& other) const override;
    PyCimenObject* operator&(const PyCimenObject& other) const override;
    PyCimenObject* operator|(const PyCimenObject& other) const override;
    PyCimenObject* operator^(const PyCimenObject& other) const override;
    PyCimenObject* operator<<(const PyCimenObject& other) const override;
    PyCimenObject* operator>>(const PyCimenObject& other) const override;
    PyCimenObject* operator==(const PyCimenObject& other) const override;
    PyCimenObject* operator<(const PyCimenObject& other) const override;
    PyCimenObject* operator>(const PyCimenObject& other) const override;
    PyCimenObject* operator-() const override;
    PyCimenObject* operator~() const override;
    PyCimenObject* operator!() const override;
    
    ll getInt() const;
    void write(std::ostream& out) const override;

    PyObject* getPythonObject() const override;
    
private:
    const ll* getIntData() const;
    void deleteData() override;
};

