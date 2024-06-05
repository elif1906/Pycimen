#pragma once

#include "./PyCimenObject.hpp"
#include "../ast/ast.hpp"



class NodeVisitor;

class PyCimenList : public PyCimenObject {
public:
    PyCimenList() : PyCimenObject(ObjectType::List, new std::vector<PyCimenObject*>()) {}

    explicit PyCimenList(const std::vector<PyCimenObject*>& v)
        : PyCimenObject(ObjectType::List, new std::vector<PyCimenObject*>(v)) {}


    PyCimenList(const std::vector<AstNode*>& nodes, NodeVisitor* visitor) 
        : PyCimenObject(ObjectType::List, new std::vector<PyCimenObject*>()) {
        for (const auto& node : nodes) {
            append(node->accept(visitor)); 
        }
    }
    
    inline bool isList() const override { return true; }
    inline bool isTruthy() const override { return getList().size() != 0; }

    const std::vector<PyCimenObject*>& getList() const {
        return *getListData();
    }

    void setList(std::vector<PyCimenObject*> list) {
        this->data = new std::vector<PyCimenObject*>(list);
    }

    const PyCimenObject* operator[](size_t index) const {
        const auto& listData = getList();
        if (index < listData.size()) {
            return listData[index];
        }
        throw std::out_of_range("Index out of range in PyCimenList");
    }

    std::size_t size() const {
        return getList().size();
    }
    void append(PyCimenObject* item) {
        getListData()->push_back(item);
    }
    void remove(std::size_t index) {
        if (index < getList().size()) {
            getListData()->erase(getListData()->begin() + index);
        }
    }
    
    void write(std::ostream& out) const override {
        out << '[';
        bool not_first = false;
        for (const auto& elem : getList()) {
            if (not_first) out << ", ";
            elem->write(out);
            not_first = true;
        }
        out << ']';
    }

    PyObject* getPythonObject() const override {
        PyObject* pythonList = PyList_New(0);
        auto list = this->getList();
        for (int i = 0; i < this->size(); i++) {
            PyObject* pythonObject = list[i]->getPythonObject();
            int res = PyList_Append(pythonList, pythonObject);
            if (res == -1) {
                PyErr_Print();
                Py_DECREF(pythonObject);
                Py_DECREF(pythonList);
                return nullptr;
            }
            Py_DECREF(pythonObject);
        }
        return pythonList;
    }    

private:
    std::vector<PyCimenObject*>* getListData() const {
        return static_cast<std::vector<PyCimenObject*>*>(data);
    }
    void deleteData() override {
        delete getListData();
    }
};
