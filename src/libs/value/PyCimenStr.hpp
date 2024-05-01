#pragma once

#include "./PyCimenObject.hpp"

class PyCimenStr : public PyCimenObject {
public:
    explicit PyCimenStr(const std::string& v) 
        : PyCimenObject(ObjectType::String, new std::string(v)) {}
    
    inline bool isStr() const override { return true; } 
    inline bool isTruthy() const override { return getStr() != ""; }
     
    PyCimenObject* operator+(const PyCimenObject& other) const override {
        if(other.isStr()) {
            const PyCimenStr* rhs = dynamic_cast<const PyCimenStr*>(&other);
            return new PyCimenStr(getStr() + rhs->getStr());
        } else {
            throw std::runtime_error("Unsupported operands for +.");
        }
    }
    PyCimenObject* operator==(const PyCimenObject& other) const override {
        if(other.isStr()) {
            const PyCimenStr* rhs = dynamic_cast<const PyCimenStr*>(&other);
            return new PyCimenBool(getStr() == rhs->getStr());
        } else {
            return new PyCimenBool(false);
        }
    }
        
    const std::string& getStr() const {
        return *getStrData();
    }
    void write(std::ostream& out) const override {
        out << "\'" << getStr() << "\'";
    }
    
private:
    const std::string* getStrData() const {
        return static_cast<std::string*>(data);
    }
    void deleteData() override {
        delete getStrData();
    }
};
