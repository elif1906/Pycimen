#include "gc.hpp"
#include <algorithm>

void GarbageCollector::freeUnused() {    

    for(auto object : objects) {
                        
        if(object->getRefCount() == 0) {

            std::vector<PyCimenObject*>::iterator position = std::find(objects.begin(), objects.end(), object);
                    
            if(position != objects.end()){
                objects.erase(position);
                delete object;
            }
        }
    }
}

void GarbageCollector::pushObject(PyCimenObject* value) {

    if(nAllocs >= 50){
        freeUnused();
        nAllocs = 0;
    }
    nAllocs++;
    objects.push_back(value); 

    return;
}
