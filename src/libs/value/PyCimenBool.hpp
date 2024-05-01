#pragma once

#include "./PyCimenInt.hpp"

class PyCimenBool : public PyCimenInt {
public:
    explicit PyCimenBool(bool v)
        : PyCimenInt(v ? 1L : 0L) {}

    inline bool isBool() const override { return true; }
    inline bool isTruthy() const override { return getBool(); }

    bool getBool() const {
        return getInt() != 0L;
    }
    void write(std::ostream& out) const override {
        out << (getBool() ? "True" : "False");
    }
};