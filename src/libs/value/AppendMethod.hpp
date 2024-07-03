#pragma once

#include "PyCimenList.hpp"
#include "PyCimenCallable.hpp"

class AppendMethod : public PyCimenCallable {
    public:
        explicit AppendMethod(PyCimenList* list): PyCimenCallable(
                ObjectType::Func,
                new class PyCimenScope()){
            this->list = list;
        }
        inline bool isCallable() const override { return true; }

        size_t arity() override {
            return 2;
        }

        PyCimenObject* call(
                Interpreter* interpreter,
                const std::vector<PyCimenObject*>& args) override{
            if(args.size() != 1) {
                throw std::runtime_error("invalid arguments");
            }
            list->append(args[0]);

            return new PyCimenNone();
        }

    private:
        PyCimenList* list;
};
