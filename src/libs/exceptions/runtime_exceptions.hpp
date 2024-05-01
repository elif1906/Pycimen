#pragma once

#include "../value/PyCimenObject.hpp"

class ReturnException : public std::exception {
public:
    ReturnException(PyCimenObject* value) : value(value) {}
    PyCimenObject* value;
};

class BreakException : public std::exception {
public:
    BreakException() {}
};

class ContinueException : public std::exception {
public:
    ContinueException() {}
};
