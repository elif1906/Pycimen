#include "./PyCimenFloat.hpp"
#include "./PyCimenInt.hpp"
#include "./PyCimenBool.hpp"

PyCimenFloat::PyCimenFloat(const std::string& v) 
    : PyCimenObject(ObjectType::Float, new llf(std::stold(v))) {}
    
PyCimenFloat::PyCimenFloat(llf v)
    : PyCimenObject(ObjectType::Float, new llf(v)) {}

PyCimenObject* PyCimenFloat::operator+(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenFloat(getFloat() + static_cast<llf>(rhs->getInt()));
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenFloat(getFloat() + rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for +.");
    }
}

PyCimenObject* PyCimenFloat::operator-(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenFloat(getFloat() - static_cast<llf>(rhs->getInt()));
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenFloat(getFloat() - rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for -.");
    }
}

PyCimenObject* PyCimenFloat::operator*(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenFloat(getFloat() * static_cast<llf>(rhs->getInt()));
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenFloat(getFloat() * rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for *.");
    }
}

PyCimenObject* PyCimenFloat::operator/(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        llf rvalue = static_cast<llf>(rhs->getInt());
        if(rvalue == 0.0) throw std::runtime_error("Attempted to divide by zero");
        return new PyCimenFloat(getFloat() / rvalue);
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        llf rvalue = rhs->getFloat();
        if(rvalue == 0.0) throw std::runtime_error("Attempted to divide by zero");
        return new PyCimenFloat(getFloat() / rvalue);
    } else {
        throw std::runtime_error("Unsupported operands for /.");
    }
}

PyCimenObject* PyCimenFloat::__intdiv__(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        ll rvalue = rhs->getInt();
        if(rvalue == 0) throw std::runtime_error("Attempted to divide by zero");
        return new PyCimenFloat(static_cast<ll>(getFloat()) / rvalue);
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        ll rvalue = static_cast<ll>(rhs->getFloat());
        if(rvalue == 0) throw std::runtime_error("Attempted to divide by zero");
        return new PyCimenFloat(static_cast<ll>(getFloat()) / rvalue);
    } else {
        throw std::runtime_error("Unsupported operands for /.");
    }
}

PyCimenObject* PyCimenFloat::operator==(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenBool(getFloat() == static_cast<llf>(rhs->getInt()));
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenBool(getFloat() == rhs->getFloat());
    } else {
        return new PyCimenBool(false);
    }
}

PyCimenObject* PyCimenFloat::operator<(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenBool(getFloat() < static_cast<llf>(rhs->getInt()));
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenBool(getFloat() < rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for <.");
    }
}

PyCimenObject* PyCimenFloat::operator>(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenBool(getFloat() > static_cast<llf>(rhs->getInt()));
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenBool(getFloat() > rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for >.");
    }
}

PyCimenObject* PyCimenFloat::operator-() const {
    return new PyCimenFloat(-getFloat());
}

PyCimenObject* PyCimenFloat::operator!() const {
    return new PyCimenBool(!(this->isTruthy()));
}

llf PyCimenFloat::getFloat() const {
    return *getFloatData();
}

void PyCimenFloat::write(std::ostream& out) const {
    out << getFloat();
}

const llf* PyCimenFloat::getFloatData() const {
    return static_cast<llf*>(data);
}

void PyCimenFloat::deleteData() {
    delete getFloatData();
}

PyObject* PyCimenFloat::getPythonObject() const {
    PyObject* object = PyFloat_FromDouble((long double)this->getFloat());
    Py_INCREF(object);
    return object;
}
