class PyCimenNumpyArray: public PyCimenObject {
public:
    PyCimenNumpyArray(int* data, size_t size): PyCimenObject(ObjectType::NumpyArray, data) {
        this->size = size;

        list = new PyCimenObject*[size];

        for(int i = 0; i < size; i++) {
            list[i] = new PyCimenInt(data[i]);
        }
    }

    inline bool isNumpyArray() const override { return true; }
    inline bool isTruthy() const override { return true; }

    const PyCimenObject* operator[](size_t index) const {
        if (index < size) {
            return list[index];
        }
        throw std::out_of_range("Index out of range in PyCimenList");
    }

    void write(std::ostream& out) const override {
        out << '[';
        bool not_first = false;
        for (int i = 0; i < size; ++i) {
            if (not_first) out << ", ";
            list[i]->write(out);
            not_first = true;
        }
        out << ']';
    }


private:
    size_t size;
    PyCimenObject** list;
};
