#include "ast.hpp"

PyCimenObject* AssignNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "AssignNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitAssignNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "AssignNode" << "\n";
    #endif
    return value;
}

PyCimenObject* CallNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "CallNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitCallNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "CallNode" << "\n";
    #endif
    return value;
}

PyCimenObject* BinaryOpNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "BinaryOpNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitBinaryOpNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "BinaryOpNode" << "\n";
    #endif
    return value;
}

PyCimenObject* TernaryOpNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "TernaryOpNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitTernaryOpNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "TernaryOpNode" << "\n";
    #endif
    return value;
}

PyCimenObject* NameNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "NameNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitNameNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "NameNode" << "\n";
    #endif
    return value;
}

PyCimenObject* ImportNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "ImportMode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitImportNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "ImportMode" << "\n";
    #endif
    return value;
}

PyCimenObject* StringNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "StringNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitStringNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "StringNode" << "\n";
    #endif
    return value;
}

PyCimenObject* BooleanNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "BooleanNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitBooleanNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "BooleanNode" << "\n";
    #endif
    return value;
}

PyCimenObject* UnaryOpNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "UnaryOpNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitUnaryOpNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "UnaryOpNode" << "\n";
    #endif
    return value;
}

PyCimenObject* IntNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "IntNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitIntNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "IntNode" << "\n";
    #endif
    return value;
}

PyCimenObject* FloatNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "FloatNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitFloatNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "FloatNode" << "\n";
    #endif
    return value;
}

PyCimenObject* ProgramNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "ProgramNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitProgramNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "ProgramNode" << "\n";
    #endif
    return value;
}

PyCimenObject* PrintNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "PrintNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitPrintNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "PrintNode" << "\n";
    #endif
    return value;
}

PyCimenObject* NullNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "NullNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitNullNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "NullNode" << "\n";
    #endif
    return value;
}

PyCimenObject* BlockNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "BlockNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitBlockNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "BlockNode" << "\n";
    #endif
    return value;
}

PyCimenObject* WhileNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "WhileNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitWhileNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "WhileNode" << "\n";
    #endif
    return value;
}

PyCimenObject* BreakNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "BreakNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitBreakNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "BlockNode" << "\n";
    #endif
    return value;
}

PyCimenObject* ContinueNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "ContinueNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitContinueNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "ContinueNode" << "\n";
    #endif
    return value;
}

PyCimenObject* PassNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "PassNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitPassNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "PassNode" << "\n";
    #endif
    return value;
}

PyCimenObject* IfNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "IfNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitIfNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "IfNode" << "\n";
    #endif
    return value;
}

PyCimenObject* FunctionNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "FunctionNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitFunctionNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "FunctionNode" << "\n";
    #endif
    return value;
}

PyCimenObject* ClassNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "ClassNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitClassNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "ClassNode" << "\n";
    #endif
    return value;
}

PyCimenObject* PropertyNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "PropertyNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitPropertyNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "PropertyNode" << "\n";
    #endif
    return value;
}

PyCimenObject* ReturnNode::accept(NodeVisitor* visitor) {
    #ifdef DEBUG
        std::cout << "visiting " << "ReturnNode" << "\n";
    #endif
    PyCimenObject* value = visitor->visitReturnNode(this);
    #ifdef DEBUG
        std::cout << "exiting " << "ReturnNode" << "\n";
    #endif
    return value;
}

