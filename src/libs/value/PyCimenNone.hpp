#pragma once

#include "./PyCimenObject.hpp"
#include "./PyCimenBool.hpp"

class PyCimenNone : public PyCimenObject {
public:
    explicit PyCimenNone()
        : PyCimenObject(ObjectType::None, nullptr) {}
        
    inline bool isNone() const override { return true; }
    inline bool isTruthy() const override { return false; }

    PyCimenObject* operator==(const PyCimenObject& other) const override {
        return new PyCimenBool(other.isNone() ? true : false); // None is only equal to None
    }

    void write(std::ostream& out) const override {
        out << "None";
    }
};


