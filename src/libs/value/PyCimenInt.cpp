#include "./PyCimenInt.hpp"
#include "./PyCimenFloat.hpp"
#include "./PyCimenBool.hpp"

PyCimenInt::PyCimenInt(const std::string& v)
    : PyCimenObject(ObjectType::Int, new ll(std::stoll(v))) {}

PyCimenInt::PyCimenInt(ll v)
    : PyCimenObject(ObjectType::Int, new ll(v)) {}

PyCimenObject* PyCimenInt::operator+(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenInt(getInt() + rhs->getInt());
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenFloat((llf)getInt() + rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for +.");
    }
 }

PyCimenObject* PyCimenInt::operator-(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenInt(getInt() - rhs->getInt());
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenFloat((llf)getInt() - rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for -.");
    }
}

PyCimenObject* PyCimenInt::operator*(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenInt(getInt() * rhs->getInt());
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenFloat(static_cast<llf>(getInt()) * rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for *.");
    }
}

PyCimenObject* PyCimenInt::operator/(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        llf rvalue = static_cast<llf>(rhs->getInt());
        if(rvalue == 0.0) throw std::runtime_error("Attempted to divide by zero");
        return new PyCimenFloat(static_cast<llf>(getInt()) / rvalue);
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        llf rvalue = rhs->getFloat();
        if(rvalue == 0.0) throw std::runtime_error("Attempted to divide by zero");
        return new PyCimenFloat(static_cast<llf>(getInt()) / rvalue);
    } else {
        throw std::runtime_error("Unsupported operands for /.");
    }
}

PyCimenObject* PyCimenInt::operator%(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        ll rvalue = rhs->getInt();
        if(rvalue == 0L) throw std::runtime_error("Modulo by zero");
        return new PyCimenInt(getInt() % rvalue);
    } else {
        throw std::runtime_error("Unsupported operands for %.");
    }
}

PyCimenObject* PyCimenInt::operator&(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenInt(getInt() & rhs->getInt());
    } else {
        throw std::runtime_error("Unsupported operands for &.");
    }
}

PyCimenObject* PyCimenInt::operator<<(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenInt(getInt() << rhs->getInt());
    } else {
        throw std::runtime_error("Unsupported operands for <<.");
    }
}

PyCimenObject* PyCimenInt::operator>>(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenInt(getInt() >> rhs->getInt());
    } else {
        throw std::runtime_error("Unsupported operands for >>.");
    }
}

PyCimenObject* PyCimenInt::operator|(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenInt(getInt() | rhs->getInt());
    } else {
        throw std::runtime_error("Unsupported operands for |.");
    }
}

PyCimenObject* PyCimenInt::operator^(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenInt(getInt() ^ rhs->getInt());
    } else {
        throw std::runtime_error("Unsupported operands for ^.");
    }
}

PyCimenObject* PyCimenInt::operator==(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenBool(getInt() == rhs->getInt());
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenBool(static_cast<llf>(getInt()) == rhs->getFloat());
    } else {
        return new PyCimenBool(false);
    }
}

PyCimenObject* PyCimenInt::operator<(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenBool(getInt() < rhs->getInt());
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenBool(static_cast<llf>(getInt()) < rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for <.");
    }
}

PyCimenObject* PyCimenInt::operator>(const PyCimenObject& other) const {
    if (other.isInt()) {
        const PyCimenInt* rhs = dynamic_cast<const PyCimenInt*>(&other);
        return new PyCimenBool(getInt() > rhs->getInt());
    } else if(other.isFloat()) {
        const PyCimenFloat* rhs = dynamic_cast<const PyCimenFloat*>(&other);
        return new PyCimenBool(static_cast<llf>(getInt()) > rhs->getFloat());
    } else {
        throw std::runtime_error("Unsupported operands for >.");
    }
}

PyCimenObject* PyCimenInt::operator-() const {
    return new PyCimenInt(-getInt());
}

PyCimenObject* PyCimenInt::operator~() const {
    return new PyCimenInt(~getInt());
}

PyCimenObject* PyCimenInt::operator!() const {
    return new PyCimenBool(!(this->isTruthy()));
}

ll PyCimenInt::getInt() const {
    return *getIntData();
}

void PyCimenInt::write(std::ostream& out) const {
    out << getInt();
}

const ll* PyCimenInt::getIntData() const {
    return static_cast<const ll*>(data);
}

void PyCimenInt::deleteData() {
    delete getIntData();
}


PyObject* PyCimenInt::getPythonObject() const {
    PyObject* object = PyLong_FromLongLong(this->getInt());
    Py_INCREF(object);
    return object;
}
