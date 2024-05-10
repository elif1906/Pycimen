class PyCimenModuleFunc : public PyCimenCallable {
public:

    size_t arity() override {
        return this->n_arity;
    }

    PyCimenModuleFunc(std::string fname, PyCimenObject* (*func)(const std::vector<PyCimenObject*>&), size_t arity)
    : PyCimenCallable(ObjectType::ModuleFunc, nullptr){
        this->func = func;
        this->n_arity = arity;
    }

    PyCimenObject* call(Interpreter* interpreter, const std::vector<PyCimenObject*>& args) override {
        return func(args);
    }

private: 
    PyCimenObject* (*func) (const std::vector<PyCimenObject*>&);
    size_t n_arity;
};
