#pragma once

#include "../ast/ast.hpp"
#include "../gc/gc.hpp"
#include "../scope/scope.hpp"

class Interpreter : public NodeVisitor {
   
public:
    Interpreter();
        
    PyCimenObject* interpret(ProgramNode* node);
    

    virtual PyCimenObject* visitProgramNode(ProgramNode* node) override { return nullptr; }
    virtual PyCimenObject* visitImportNode(ImportNode* node) override;
    virtual PyCimenObject* visitBlockNode(BlockNode* node) override;
    virtual PyCimenObject* visitPrintNode(PrintNode* node) override;
    virtual PyCimenObject* visitWhileNode(WhileNode* node) override;
    virtual PyCimenObject* visitBreakNode(BreakNode* node) override;
    virtual PyCimenObject* visitContinueNode(ContinueNode* node) override;
    virtual PyCimenObject* visitPassNode(PassNode* node) override;
    virtual PyCimenObject* visitIfNode(IfNode* node) override;
    virtual PyCimenObject* visitAssignNode(AssignNode* node) override;
    virtual PyCimenObject* visitTernaryOpNode(TernaryOpNode* node) override;
    virtual PyCimenObject* visitBinaryOpNode(BinaryOpNode* node) override;
    virtual PyCimenObject* visitUnaryOpNode(UnaryOpNode* node) override;
    virtual PyCimenObject* visitIntNode(IntNode* node) override;
    virtual PyCimenObject* visitFloatNode(FloatNode* node) override;
    virtual PyCimenObject* visitNameNode(NameNode* node) override;
    virtual PyCimenObject* visitStringNode(StringNode* node) override;
    virtual PyCimenObject* visitBooleanNode(BooleanNode* node) override;
    virtual PyCimenObject* visitNullNode(NullNode* expr) override;
    virtual PyCimenObject* visitFunctionNode(FunctionNode* node) override;
    virtual PyCimenObject* visitCallNode(CallNode* node) override;
    virtual PyCimenObject* visitReturnNode(ReturnNode* node) override;
    virtual PyCimenObject* visitClassNode(ClassNode* node) override;
    virtual PyCimenObject* visitPropertyNode(PropertyNode* node) override;

    void pushContext(PyCimenScope* frame) {
        contextStack.push_back(frame);
    }
    
    void popContext() { 
        
        if(!contextStack.empty()) {
            //delete contextStack.back();
            contextStack.pop_back();
        } else {
            throw std::runtime_error("Cannot pop context from empty stack");
        }
    }
    
    PyCimenScope* currentContext() { 
        
        if(!contextStack.empty()) {
            return contextStack.back();
        } else {
            return nullptr;
        }
    }
    
    void defineOnContext(const std::string& name, PyCimenObject* value) {
        
        if(!contextStack.empty()){
            PyCimenScope* lastFrame = contextStack.back();
            lastFrame->define(name, value);
        } else {
            throw std::runtime_error("Cannot define variable outside of context");
        }
    }
    
    PyCimenObject* getFromContext(const std::string& name) {
        
        if(!contextStack.empty()){
            PyCimenScope* lastFrame = contextStack.back();
            return lastFrame->get(name);
        } else {
            throw std::runtime_error("Cannot access variable outside of context");
        }
    }

private:
    GarbageCollector GC;
    std::vector<PyCimenScope*> contextStack;
};

