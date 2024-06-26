#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "../token/token.hpp"
#include "../value/PyCimenInstance.hpp"


using llf = long double;
using lld = long long int;

class NodeVisitor;

// Define the AST node types
enum class AstNodeType {
    // Root
    Program, 
    
    // Group Statements
    Block,
    
    // Statements
    Print, Range
    ,While, Break, Continue,
    If, Function, Return,
    Class, AttrRef,
    Pass,
    Import,For,

    // Expressions
    UnaryOp, BinaryOp, TernaryOp,
    Assign, Call,

    // Literals
    Name, String,
    Int, Float, Boolean, Null, List
};


class AstNode {

public:

    AstNode(AstNodeType type) : type(type) {}
    
    virtual ~AstNode() {}

    bool is_name_node() { return type == AstNodeType::Name; }
    bool is_property_node() { return type == AstNodeType::AttrRef; }
    bool is_block() { return type == AstNodeType::Block; }
    bool is_function() { return type == AstNodeType::Function; }

    virtual PyCimenObject* accept(NodeVisitor* visitor) = 0;

    AstNodeType type;
};

class ListNode : public AstNode {
public:

  ListNode() : AstNode(AstNodeType::List) {
    this->values = {};
  }

  ListNode(std::vector<AstNode*> values) : AstNode(AstNodeType::List) {
    this->values = values;
  }

  void append(AstNode* value){
    values.push_back(value);
  }

  int size() {
      return values.size();
  }

  std::vector<AstNode*> get_values() {
    return values;
  }

  virtual PyCimenObject* accept(NodeVisitor* visitor) override;

private:
  std::vector<AstNode*> values;
};


class PrintNode : public AstNode {

public:

    PrintNode(AstNode* args) 
      : AstNode(AstNodeType::Print),  args(args) {}

    AstNode* args;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class RangeNode : public AstNode {
public:
    RangeNode(AstNode* start, AstNode* end)
        : AstNode(AstNodeType::Range), start(start), end(end) {}

    AstNode* start;
    AstNode* end;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};




class BlockNode : public AstNode {

public:

    BlockNode(const std::vector<AstNode*> statements)
      : AstNode(AstNodeType::Block), statements(statements) {}
      
    std::vector<AstNode*> statements;
    
    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class ProgramNode : public AstNode {

public:

    ProgramNode(BlockNode* body) 
      : AstNode(AstNodeType::Program), body(body) {}
    
    BlockNode* body;
    
    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class TernaryOpNode : public AstNode {

public:

    TernaryOpNode(AstNode* condition, AstNode* left, AstNode* right) 
      : AstNode(AstNodeType::TernaryOp), 
        cond(condition), left(left), right(right) {}

    AstNode *cond, *left, *right;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class BinaryOpNode : public AstNode {

public:

    BinaryOpNode(AstNode* left, Token op, AstNode* right)
      : AstNode(AstNodeType::BinaryOp), 
        left(left), op(op), right(right) {}

    Token op;
    AstNode *left, *right;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class UnaryOpNode : public AstNode {

public:
    
    UnaryOpNode(Token op, AstNode* right) 
      : AstNode(AstNodeType::UnaryOp), 
        op(op), right(right) {}

    Token op;
    AstNode* right;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class IntNode : public AstNode {

public:
    
    IntNode(Token value) 
      : AstNode(AstNodeType::Int), value(value) {}

    Token value; // TODO: implement the BigInt class
    
    const std::string& getLexeme() { return value.lexeme; }

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class FloatNode : public AstNode {

public:
    
    FloatNode(Token value)
      : AstNode(AstNodeType::Float), value(value) {}

    Token value;
    
    const std::string& getLexeme() { return value.lexeme; }

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class NameNode : public AstNode {

public:
    
    NameNode(Token name) 
      : AstNode(AstNodeType::Name), value(name) {}

    Token value;
    
    const std::string& getLexeme() { return value.lexeme; }

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class StringNode : public AstNode {

public:
    
    StringNode(Token value) 
      : AstNode(AstNodeType::String), value(value) {}
    
    Token value;
    
    const std::string& getLexeme() { return value.lexeme; }

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class BooleanNode : public AstNode {

public:

    BooleanNode(bool value) 
      : AstNode(AstNodeType::Boolean), value(value) {}
    
    bool value;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class ReturnNode : public AstNode {

public:

    ReturnNode(Token keyword, AstNode* value) 
      : AstNode(AstNodeType::Return), kwd(keyword), value(value) {}
    
    Token kwd;
    AstNode* value;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class NullNode : public AstNode {

public:
    
    NullNode(Token none) : AstNode(AstNodeType::Null), none(none) {}
    
    Token none;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class CallNode : public AstNode {

public:
    
    CallNode(AstNode* caller, const std::vector<AstNode*>& args) 
      : AstNode(AstNodeType::Call), caller(caller), args(args) {}

    AstNode* caller;
    std::vector<AstNode*> args;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class AssignNode : public AstNode {
public:
    AssignNode(AstNode* name, AstNode* value, Token op) 
      : AstNode(AstNodeType::Assign), name(name), value(value), op(op) {}

    AstNode *name, *value;
    Token op;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class WhileNode : public AstNode {
public:

    WhileNode(AstNode* cond, AstNode* body) 
      : AstNode(AstNodeType::While), cond(cond), body(body) {}

    AstNode *cond, *body;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};

class ForNode : public AstNode {
public:
    ForNode(AstNode* target, AstNode* iterable, AstNode* body, AstNode* elseBranch)
      : AstNode(AstNodeType::For) {
        this->target = target;
        this->iterable = iterable;
        this->body = body;
        this->elseBranch = elseBranch;
      }

    AstNode* target;
    AstNode* iterable; 
    AstNode* body;
    AstNode* elseBranch;
    
    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};

class BreakNode : public AstNode {

public:

    BreakNode(Token keyword) 
      : AstNode(AstNodeType::Break), kwd(keyword) {}
    
    Token kwd;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class ContinueNode : public AstNode {

public:

    ContinueNode(Token keyword) 
      : AstNode(AstNodeType::Continue), kwd(keyword) {}
    
    Token kwd;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class PassNode : public AstNode {

public:

    PassNode(Token keyword) 
      : AstNode(AstNodeType::Pass), kwd(keyword) {}
    
    Token kwd;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class FunctionNode : public AstNode {

public:

    FunctionNode(Token fname, std::vector<AstNode*> params, AstNode* body)
        : AstNode(AstNodeType::Function), 
          fname(fname), params(params), body(body) {}
   
    const std::string& getName() const { return fname.lexeme; };
    const std::vector<AstNode*>& getParams() const { return params; }
    AstNode* getBody() const { return body; }

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;

private:
    Token fname;
    std::vector<AstNode*> params;
    AstNode* body;
};

class ImportNode : public AstNode {
  public:

    ImportNode(std::string moduleName) : AstNode(AstNodeType::Import){
      this ->moduleName = moduleName;
    }

    std::string getModuleName() {
      return this->moduleName;
    }

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;

  private:
    std::string moduleName;
};


class ClassNode : public AstNode {

public:

    ClassNode(Token kname, AstNode* body)
        : AstNode(AstNodeType::Class), kname(kname), body(body) {}
   
    const std::string& getName() const { return kname.lexeme; }
    AstNode* getBody() const { return body; }

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;

private:
    Token kname;
    AstNode* body;
};


class PropertyNode : public AstNode {

public:
    
    PropertyNode(AstNode* object, AstNode* attr)
        : AstNode(AstNodeType::AttrRef), object(object), attribute(attr) {}

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
    
    AstNode *object, *attribute;
};

class IfNode : public AstNode {

public:

    IfNode(AstNode* cond, AstNode* trueBranch, 
           const std::vector<std::pair<AstNode*, AstNode*>>& elifBranches, 
           AstNode* elseBranch) : AstNode(AstNodeType::If), cond(cond), 
          trueBranch(trueBranch), elifBranches(elifBranches), elseBranch(elseBranch) {}

    AstNode *cond, *trueBranch, *elseBranch;;
    std::vector<std::pair<AstNode*, AstNode*>> elifBranches;

    virtual PyCimenObject* accept(NodeVisitor* visitor) override;
};


class NodeVisitor {

public:

    ~NodeVisitor() = default;

    virtual PyCimenObject* visitImportNode(ImportNode* node) = 0;
    virtual PyCimenObject* visitProgramNode(ProgramNode* node) = 0;
    virtual PyCimenObject* visitBlockNode(BlockNode* node) = 0;
    virtual PyCimenObject* visitPrintNode(PrintNode* node) = 0;
    virtual PyCimenObject* visitRangeNode(RangeNode* node) = 0;
    virtual PyCimenObject* visitWhileNode(WhileNode* node) = 0;
    virtual PyCimenObject* visitForNode(ForNode* node) = 0;
    virtual PyCimenObject* visitBreakNode(BreakNode* node) = 0;
    virtual PyCimenObject* visitContinueNode(ContinueNode* node) = 0;
    virtual PyCimenObject* visitPassNode(PassNode* node) = 0;
    virtual PyCimenObject* visitIfNode(IfNode* node) = 0;
    virtual PyCimenObject* visitAssignNode(AssignNode* node) = 0;
    virtual PyCimenObject* visitTernaryOpNode(TernaryOpNode* node) = 0;
    virtual PyCimenObject* visitBinaryOpNode(BinaryOpNode* node) = 0;
    virtual PyCimenObject* visitUnaryOpNode(UnaryOpNode* node) = 0;
    virtual PyCimenObject* visitIntNode(IntNode* node) = 0;
    virtual PyCimenObject* visitFloatNode(FloatNode* node) = 0;
    virtual PyCimenObject* visitNameNode(NameNode* node) = 0;
    virtual PyCimenObject* visitStringNode(StringNode* node) = 0;
    virtual PyCimenObject* visitBooleanNode(BooleanNode* node) = 0;
    virtual PyCimenObject* visitNullNode(NullNode* node) = 0;
    virtual PyCimenObject* visitFunctionNode(FunctionNode* node) = 0;
    virtual PyCimenObject* visitCallNode(CallNode* node) = 0;
    virtual PyCimenObject* visitReturnNode(ReturnNode* node) = 0;
    virtual PyCimenObject* visitClassNode(ClassNode* node) = 0;
    virtual PyCimenObject* visitPropertyNode(PropertyNode* node) = 0;
    virtual PyCimenObject* visitListNode(ListNode* node) = 0;
    
};
