#include <iomanip>
#include <stdexcept>
#include "./interpreter.hpp"
#include "../token/token.hpp"
#include "../token/tokentype.hpp"
#include "../exceptions/runtime_exceptions.hpp" // break, continue, return
#include "../value/primitives.hpp"
#include "../builtin/builtins.hpp"
#include "../value/PyCimenModule.hpp"
#include "../value/AppendMethod.hpp"
#include <sstream>


void todo() {
    throw std::runtime_error("Feature not implemented yet");
}

Interpreter::Interpreter() {
    
    PyCimenScope* builtins = new PyCimenScope();
    pushContext(builtins);
    PyCimenScope* globals = new PyCimenScope(builtins);
    pushContext(globals);
    
    std::string inputRepr("input");    
    builtins->define(inputRepr, new Input());
}

PyCimenObject* Interpreter::interpret(ProgramNode* node) {
    return node->body->accept(this);
}

std::string formatNumber(llf number) {
    
    std::ostringstream oss;
    int intPart = static_cast<int>(number);
    double fracPart = number - intPart;

    if (fracPart == 0.0) {
        oss << intPart << ".0";
    } else {
        oss << number;
    }
    return oss.str();
}

PyCimenObject* Interpreter::visitPrintNode(PrintNode* node) {
    
    PyCimenObject* argValue = nullptr;

    if (node->args != nullptr) {
        argValue = node->args[0].accept(this);
    }
    if (argValue != nullptr) {
        if((*argValue).isStr()) {
            const PyCimenStr* obj = dynamic_cast<const PyCimenStr*>(argValue);
            std::cout << (*obj).getStr();
        } else if((*argValue).isFloat()) {
            const PyCimenFloat* obj = dynamic_cast<const PyCimenFloat*>(argValue);
            std::cout << formatNumber((*obj).getFloat());
        } else {
            std::cout << *argValue;
        }
    }
    std::cout << "\n" << std::flush;

    return new PyCimenNone();
}

PyCimenObject* Interpreter::visitImportNode(ImportNode* node) {
    char* moduleName = node->getModuleName().data();

    PyCimenModule* module = new PyCimenModule(moduleName);

    PyCimenObject* value = module;

    defineOnContext(node->getModuleName(), value);

    GC.pushObject(value);

    return value;
}

PyCimenObject* Interpreter::visitIntNode(IntNode* node){
    
    const std::string& str = node->getLexeme();
    PyCimenObject* value = new PyCimenInt(str);
    GC.pushObject(value);
    return value;
}

PyCimenObject* Interpreter::visitFloatNode(FloatNode* node){
    
    const std::string& str = node->getLexeme();
    PyCimenObject* value = new PyCimenFloat(str);
    GC.pushObject(value);
    return value;
}

PyCimenObject* Interpreter::visitFunctionNode(FunctionNode* node){

    const std::string& fname = node->getName();
    PyCimenFunction* value = new PyCimenFunction(node, this->currentContext());
    defineOnContext(fname, value);

    return new PyCimenNone();
}

PyCimenObject* Interpreter::visitClassNode(ClassNode* node) {
    
    PyCimenScope* closure = this->currentContext();
    PyCimenScope* classEnv = new PyCimenScope(closure);
    
    this->pushContext(classEnv);
    node->getBody()->accept(this);
    this->popContext();
    
    const std::string& kname = node->getName();
    PyCimenClass* value = new PyCimenClass(kname, classEnv);

    closure->define(kname, value);
    
    return new PyCimenNone();
}

PyCimenObject* Interpreter::visitPropertyNode(PropertyNode* node) {
    
    PropertyNode* node_property = node;
    
    PyCimenObject* object = node_property->object->accept(this);

    NameNode* attr = dynamic_cast<NameNode*>(node_property->attribute);
   
    if (!object->isList()) {
        while (attr == nullptr) {
            node_property = dynamic_cast<PropertyNode*>(node_property->attribute);
            object = node_property->object->accept(this);
            attr = dynamic_cast<NameNode*>(node_property->attribute);
        }
    }

    std::string name = std::string("");
    if (attr) {
        name = attr->getLexeme();
    }

   
    
    if(object->isInstance()) {
    
        PyCimenInstance* instance = static_cast<PyCimenInstance*>(object);
        
        PyCimenScope* context = instance->getContext();
        PyCimenObject* value = context->get(name);
        
        if(value->isFunc()) {
            return static_cast<PyCimenFunction*>(value)->bind(instance);
        } else {
            return value;
        }
        
    } else if(object->isFunc()) {
        
        PyCimenFunction* function = static_cast<PyCimenFunction*>(object);
        
        PyCimenScope* context = function->getContext();
        return context->get(name);

    } else if (object->isModule()) {

        PyCimenModule* module = static_cast<PyCimenModule*>(object);

        PyCimenObject* attr = module->getAttr(name.c_str());

        return attr;

    } else if (object->isModuleAttr()) {

        PyCimenModuleAttr* attr = static_cast<PyCimenModuleAttr*>(object);

        PyCimenObject* new_attr = attr->getAttr(name.c_str());

        return new_attr;
    } else if (object->isList()) {

        PyCimenList* list = dynamic_cast<PyCimenList*>(object);

        // size attribute
        if(attr && strstr(name.data(), "size")) {
            return new PyCimenInt(list->size());
        }

        if(attr && strstr(name.data(), "append")) {
            return new AppendMethod(list);
        }

        // index access

        auto indexObject = node->attribute->accept(this);

        auto index = dynamic_cast<PyCimenInt*>(indexObject);

        if(index) {
            //access with int
            auto indexVal = index->getInt();

            if (indexVal < 0) {
                indexVal += list->size();
            }


            if (indexVal >= list->size()) {
                throw std::runtime_error("index out of array bound");
            }

            return list->getList()[indexVal];
        } else {
            auto varName = dynamic_cast<NameNode*>(node->attribute)->getLexeme();
            auto indexObject = node->attribute->accept(this);
            auto indexVal = dynamic_cast<PyCimenInt*>(indexObject)->getInt();

            if (indexVal >= list->size()) {
                throw std::runtime_error("index out of array bound");
            }

            return list->getList()[indexVal];

        }

        return new PyCimenNone();
    }
    
    return new PyCimenNone();
}

PyCimenObject* Interpreter::visitBlockNode(BlockNode* node) {
    
    for(auto statement : node->statements) {
    	statement->accept(this);
    }
    return new PyCimenNone();
}

PyCimenObject* Interpreter::visitWhileNode(WhileNode* node){

    PyCimenObject* cond = node->cond->accept(this);

    while(cond->isTruthy()){
     	try {
     	    node->body->accept(this);
     	} catch(BreakException& be) {
     	    break;
     	} catch(ContinueException& ce) {
     	    ; // goes back to evaluate the condition again
     	}
        cond = node->cond->accept(this);
    }
    return new PyCimenNone();
}
PyCimenObject* Interpreter::visitForNode(ForNode* node) {
    
    std::string varName = static_cast<NameNode*>(node->target)->getLexeme();

    PyCimenObject* iterable = node->iterable->accept(this);

    if (iterable->isList()) {
        PyCimenList* iterableList = static_cast<PyCimenList*>(iterable);
        const std::vector<PyCimenObject*>& values = iterableList->getList();
        for (PyCimenObject* obj : values) {

            this->defineOnContext(varName, obj);
            
            node->body->accept(this);
        }
    } else {
        throw std::runtime_error("Unsupported iterable type");
    }

    
    if (node->elseBranch != nullptr) {
        node->elseBranch->accept(this);
    }

    return new PyCimenNone();
}

PyCimenObject* Interpreter::visitBreakNode(BreakNode* node) {
    throw BreakException();
    return nullptr; // unreachable
}

PyCimenObject* Interpreter::visitContinueNode(ContinueNode* node) {
    throw ContinueException();
    return nullptr; // unreachable
}

PyCimenObject* Interpreter::visitPassNode(PassNode* node) {
    return new PyCimenNone();
}

PyCimenObject* Interpreter::visitListNode(ListNode* node) {
    std::vector<PyCimenObject*> values;
    for (auto& valueNode : node->get_values()) {
        auto object = valueNode->accept(this);
        object->incRefCount();
        values.push_back(object);
    }

    return new PyCimenList(values);
}

PyCimenObject* Interpreter::visitIfNode(IfNode* node) {

    PyCimenObject* cond = node->cond->accept(this);

    if (cond->isTruthy()) {
        return node->trueBranch->accept(this);
    } else {
        for (const auto& elif : node->elifBranches) {
            PyCimenObject* elifCond = elif.first->accept(this);
            if (elifCond->isTruthy()) {
                return elif.second->accept(this);
            }
        }
        if (node->elseBranch != nullptr) {
            return node->elseBranch->accept(this);
        }
    }
    return new PyCimenNone();
}

PyCimenObject* Interpreter::visitTernaryOpNode(TernaryOpNode* node) {

    PyCimenObject* cond = node->cond->accept(this);

    if(cond->isTruthy()) {
        return node->left->accept(this);
    } else {
        return node->right->accept(this);
    }
    return nullptr; // unreachable
}

PyCimenObject* Interpreter::visitBinaryOpNode(BinaryOpNode* node)  {

    PyCimenObject* leftValue = node->left->accept(this);
    leftValue->incRefCount();

    PyCimenObject* rightValue = node->right->accept(this);
    rightValue->incRefCount();

    PyCimenObject* value = nullptr;

    switch(node->op.type) {
        case TokenType::Plus: // TODO: replace with __add__ call
            value = *leftValue + *rightValue;
            GC.pushObject(value);
            break;
        case TokenType::Minus: // TODO: replace with __sub__ call
            value = *leftValue - *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::Star: // TODO: replace with __mul__ call
            value = *leftValue * *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::Slash: // TODO: replace with __truediv__ call
            value = *leftValue / *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::DoubleSlash:
            value = leftValue->__intdiv__(*rightValue);
            GC.pushObject(value); 
            break;
        case TokenType::Ampersand: // TODO: replace with __and__ call
            value = *leftValue & *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::Pipe: // TODO: replace with __or__ call
            value = *leftValue | *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::Caret: // TODO: replace with __xor__ call
            value = *leftValue ^ *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::Mod: // TODO: replace with __mod__ call
            value = *leftValue % *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::EqualEqual: // TODO: replace with __eq__ call
            value = *leftValue == *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::BangEqual: // TODO: replace with __ne__ call
            value = !(*(*leftValue == *rightValue));
            GC.pushObject(value);
            break;
        case TokenType::Less: // TODO: replace with __lt__ call
            value = *leftValue < *rightValue; 
            GC.pushObject(value);
            break;
        case TokenType::Greater: // TODO: replace with __gt__ call
            value = *leftValue > *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::LessEqual: // TODO: replace with __le__ call
            value = !(*(*leftValue > *rightValue));
            GC.pushObject(value);
            break;
        case TokenType::GreaterEqual: // TODO: replace with __ge__ call
            value = !(*(*leftValue < *rightValue));
            GC.pushObject(value);
            break;
        case TokenType::LeftShift: // TODO: replace with __lshift__ call
            value = *leftValue << *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::RightShift: // TODO: replace with __rshift__ call
            value = *leftValue >> *rightValue;
            GC.pushObject(value); 
            break;
        case TokenType::Or:
            /*inter
             *  try to do short-circuit: if after evaluating the left operand, 
             *  the result of the logical expression is known, 
             *  do not evaluate the right operand
            */
            value = leftValue->isTruthy() ? leftValue : rightValue;
            break;
        case TokenType::And:
            /*
             *  try to do short-circuit: if after evaluating the left operand, 
             *  the result of the logical expression is known, 
             *  do not evaluate the right operand
            */
            value = !(leftValue->isTruthy()) ? leftValue : rightValue;
            break;
        default:
            throw std::logic_error("Unsupported binary operator");
    } 
    leftValue->decRefCount();
    rightValue->decRefCount();

    return value;
}

PyCimenObject* Interpreter::visitAssignNode(AssignNode* node) {

    PyCimenObject* value = node->value->accept(this);
    value->incRefCount();
    
    AstNode* targetNode = node->name;
    std::string varName;
    PyCimenScope* currCtx = nullptr;
    
    if(targetNode->is_name_node()) {
        NameNode* name = static_cast<NameNode*>(targetNode);
        varName = name->getLexeme();

        currCtx = this->currentContext();
        
    } else if (targetNode->is_property_node()) {
        PropertyNode* propertyNode = static_cast<PropertyNode*>(targetNode);
        PyCimenObject* object = propertyNode->object->accept(this);

        if (object->isInstance()) {
            PyCimenInstance* instance = static_cast<PyCimenInstance*>(object);
            NameNode* attribute = static_cast<NameNode*>(propertyNode->attribute);
            varName = attribute->getLexeme();
            currCtx = instance->getContext();
        } else if (object->isList()) {
            auto pyCimenList = dynamic_cast<PyCimenList*>(object);
            auto list = pyCimenList->getList();
            auto index = dynamic_cast<IntNode*>(propertyNode->attribute);
            ll indexVal;

            if(index) {
                //access with int
                indexVal = stoi(index->getLexeme());

                if (indexVal >= list.size()) {
                    throw std::runtime_error("index out of array bound");
                }

            } else {
                auto indexObject = propertyNode->attribute->accept(this);
                indexVal = dynamic_cast<PyCimenInt*>(indexObject)->getInt();

                if (indexVal >= list.size()) {
                    throw std::runtime_error("index out of array bound");
                }
            }

            switch(node->op.type) {
                case TokenType::Equals:
                    list[indexVal] = value;
                    break;
                case TokenType::PlusEqual:
                    list[indexVal] = (*list[indexVal] + *value);
                    break;
                case TokenType::MinusEqual:
                    list[indexVal] = *list[indexVal] - *value;
                    break;
                case TokenType::StarEqual:
                    list[indexVal] = *list[indexVal] * *value;
                    break;
                case TokenType::SlashEqual:
                    list[indexVal] = *list[indexVal] / *value;
                    break;
                case TokenType::ModEqual:
                    list[indexVal] = *list[indexVal] % *value;
                    break;
                case TokenType::AndEqual:
                    list[indexVal] = *list[indexVal] & *value;
                    break;
                case TokenType::OrEqual:
                    list[indexVal] = *list[indexVal] | *value;
                    break;
                case TokenType::XorEqual:
                    list[indexVal] = *list[indexVal] ^ *value;
                    break;
                case TokenType::LeftShiftEqual:
                    list[indexVal] = *list[indexVal] << *value;
                    break;
                case TokenType::RightShiftEqual:
                    list[indexVal] = *list[indexVal] >> *value;
                    break;
                default:
                    throw std::runtime_error("Unsupported assignment operator");
            }

            pyCimenList->setList(list);

            return pyCimenList;
        }
        
    } else {
        throw std::runtime_error("Unsupported target expression");
    }
    
    if(node->op.type == TokenType::Equals) {
        currCtx->define(varName, value);
    } else {
        PyCimenObject* targetValue = currCtx->get(varName);
        targetValue->incRefCount();
        
        switch(node->op.type) {
            case TokenType::PlusEqual:
                value = *targetValue + *value;
                break;
            case TokenType::MinusEqual:
                value = *targetValue - *value;
                break;
            case TokenType::StarEqual:
                value = *targetValue * *value;
                break;
            case TokenType::SlashEqual:
                value = *targetValue / *value;
                break;
            case TokenType::ModEqual:
                value = *targetValue % *value;
                break;
            case TokenType::AndEqual:
                value = *targetValue & *value;
                break;
            case TokenType::OrEqual:
                value = *targetValue | *value;
                break;
            case TokenType::XorEqual:
                value = *targetValue ^ *value;
                break;
            case TokenType::LeftShiftEqual:
                value = *targetValue << *value;
                break;
            case TokenType::RightShiftEqual:
                value = *targetValue >> *value;
                break;
            default:
                throw std::runtime_error("Unsupported assignment operator");
        }
        currCtx->define(varName, value);
        targetValue->decRefCount();
    }
    value->decRefCount();

    return value;
}

PyCimenObject* Interpreter::visitNameNode(NameNode* node){
    const std::string& varname = node->getLexeme();
    return getFromContext(varname);;
}

PyCimenObject* Interpreter::visitBooleanNode(BooleanNode* node){
    PyCimenObject* value = new PyCimenBool(node->value);
    GC.pushObject(value);
    return value;
}

PyCimenObject* Interpreter::visitStringNode(StringNode* node){
    const std::string& str = node->getLexeme();
    PyCimenObject* value = new PyCimenStr(str);
    GC.pushObject(value);
    return value;
}

PyCimenObject* Interpreter::visitUnaryOpNode(UnaryOpNode* node){

    PyCimenObject* operandValue = node->right->accept(this);
    PyCimenObject* result = nullptr;

    switch(node->op.type) {
        case TokenType::Minus:
            result = -(*operandValue);
            break;
        case TokenType::Not:
            result = !(*operandValue);
            break;
        case TokenType::Tilde:
            result = ~(*operandValue);
            break;
        default:
            throw std::logic_error("Unsupported unary operator");
    }
    GC.pushObject(result);
    return result;
}

PyCimenObject* Interpreter::visitNullNode(NullNode* expr){
    PyCimenObject* value = new PyCimenNone();
    GC.pushObject(value);
    return value;
}

PyCimenObject* Interpreter::visitCallNode(CallNode* expr) {
    
    PyCimenObject* callee = expr->caller->accept(this);

    if (!callee->isCallable()) {
        throw std::runtime_error("not a callable object");
    }
    PyCimenCallable* callable = static_cast<PyCimenCallable*>(callee);
    
    std::vector<PyCimenObject*> arguments;
    arguments.reserve(expr->args.size());
    
    for(AstNode* argumentNode : expr->args) {
        arguments.push_back(argumentNode->accept(this));
    }
    return (*callable).call(this, arguments);
}


PyCimenObject* Interpreter::visitRangeNode(RangeNode* rangeNode) {

    PyCimenObject* start = rangeNode->start->accept(this);
    PyCimenObject* end = rangeNode->end->accept(this);

    auto startVal = dynamic_cast<PyCimenInt*>(start)->getInt();
    auto endVal = dynamic_cast<PyCimenInt*>(end)->getInt();

    PyCimenList* list = new PyCimenList();

    if (startVal < endVal) {
        for (int i = startVal; i < endVal; ++i) {
            list->append(new PyCimenInt(i));
        }
    } else {
        for (int i = startVal; i > endVal; --i) {
            list->append(new PyCimenInt(i));
        }
    }


    return list;
}


PyCimenObject* Interpreter::visitReturnNode(ReturnNode* node) {
    
    AstNode* value = node->value;
    PyCimenObject* retValue = value ? value->accept(this) : new PyCimenNone();
    throw ReturnException(retValue);
    return nullptr; // unreachable
};

