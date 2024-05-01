#pragma once

#include <string>
#include <unordered_map>
#include <stdexcept>

class PyCimenScope {
public:
    PyCimenScope(PyCimenScope* enclosing = nullptr) : enclosing(enclosing) {}

    ~PyCimenScope() {
        for (auto& [key, value] : values) {
            if (value!= nullptr) {
                value->decRefCount();
            }
        }
    }

    void define(const std::string& name, PyCimenObject* value) {
        if (name.empty()) {
            throw std::runtime_error("Variable name cannot be empty");
        }
        if (value == nullptr) {
            throw std::runtime_error("Value cannot be nullptr");
        }
        auto it = values.find(name);
        if (it!= values.end()) {
            PyCimenObject* temp = it->second;
            if (temp!= nullptr) {
                temp->decRefCount();
            }
            it->second = value;
            value->incRefCount();
        } else {
            values.emplace(name, value);
            value->incRefCount();
        }
    }

    PyCimenObject* get(const std::string& name) {
        if (name.empty()) {
            throw std::runtime_error("Variable name cannot be empty");
        }
        auto it = values.find(name);
        if(it!= values.end()) {
            return it->second;
        } else if (enclosing) {
            return enclosing->get(name);
        } else {
            throw std::runtime_error("Undeclared variable '" + name + "'.");
        }
    }

    PyCimenScope* getEnclosing() {
        return enclosing;
    }

private:
    std::unordered_map<std::string, PyCimenObject*> values;
    PyCimenScope* enclosing;
};