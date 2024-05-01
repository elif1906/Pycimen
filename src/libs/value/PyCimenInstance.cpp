#include "./PyCimenInstance.hpp"
#include "./PyCimenClass.hpp"

PyCimenInstance::PyCimenInstance(PyCimenClass* klass)
 : PyCimenObject(ObjectType::Instance), base(klass){
    fields = new PyCimenScope(klass->getContext());
}

PyCimenObject* PyCimenInstance::operator==(const PyCimenObject& other) const {
    return new PyCimenBool(other.isNone() ? false : true);
}

void PyCimenInstance::write(std::ostream& out) const {
    out << "<\'" << base->kname << "\' instance>";
}