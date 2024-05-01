#pragma once

#include "../value/PyCimenObject.hpp"

class GarbageCollector { 

public:
    GarbageCollector() {};
    void freeUnused(); 
    void pushObject(PyCimenObject* value);
        
private:
    std::vector<PyCimenObject*> objects; 
    unsigned int nAllocs = 0;
};
