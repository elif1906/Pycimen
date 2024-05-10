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
    
private:
    std::vector<PyCimenObject*>* getListData() const {
        return static_cast<std::vector<PyCimenObject*>*>(data);
    }
    void deleteData() override {
        delete getListData();
    }
};
