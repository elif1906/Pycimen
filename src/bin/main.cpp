#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "../libs/lexer/lexer.hpp"
#include "../libs/parser/parser.hpp"
#include "../libs/ast/ast.hpp"
#include "../libs/interpreter/interpreter.hpp"
#include <string.h>

#pragma GCC optimize("Ofast")
#pragma GCC target("avx,avx2,fma")

using namespace std;

inline void show_tokens(const vector<Token>& tokens) {
    for (const auto& token : tokens) {
        cout << token << '\n';
    }
    cout << flush;
}

int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " [filename].pcl\n";
        return 1;
    }

    if (strcmp(argv[1], "--version") == 0) {
        cout << "version: 1.0.0" << std::endl;
        return 0;
    }

    const char* const filename = argv[1];
    ifstream inputFile(filename);

    if (!inputFile) {
        cerr << "Error: could not open file '" << filename << "'\n";
        return 1;
    }

    string source((istreambuf_iterator<char>(inputFile)), {});

    try {
        Lexer lexer(source);
        vector<Token> tokens = lexer.scanTokens();

        #ifdef DEBUG
            /* To define DEBUG, use `make DEBUG=1` when compiling */
            show_tokens(tokens);
        #endif

        Parser parser(tokens);
        ProgramNode* root = parser.parse();

        Interpreter interpreter;
        interpreter.interpret(root);
        
    } catch (const runtime_error& err) {
        cerr << err.what() << '\n';
        return EXIT_FAILURE;
    }

      

    Py_Initialize();

    PyObject* numpy_module = PyImport_ImportModule("numpy");
    if (!numpy_module) {
        PyErr_Print();
        Py_Finalize();
        return EXIT_FAILURE;
    }

    import_array();  

    int nd = 2;
    npy_intp dims[2] = {6, 1};
    int type_code = NPY_INT;

    PyObject* py_array = PyArray_SimpleNew(nd, dims, type_code);
    if (!py_array) {
        PyErr_Print();
        Py_DECREF(numpy_module);
        Py_Finalize();
        return EXIT_FAILURE;
    }

    int* data = (int*)PyArray_DATA(py_array);
    if (!data) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(numpy_module);
        Py_Finalize();
        return EXIT_FAILURE;
    }

    npy_intp dims_res = *PyArray_DIMS(py_array);
    cout << "dims: " << dims_res << endl;

    // Set array values
    data[0] = 1;
    data[1] = 2;
    // ...

    PyObject* mean_func = PyObject_GetAttrString(numpy_module, "mean");
    if (!mean_func || !PyCallable_Check(mean_func)) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(numpy_module);
        Py_Finalize();
        return EXIT_FAILURE;
    }

    PyObject* arrayRes = PyObject_CallFunction(mean_func, "O", py_array);
    if (!arrayRes) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(numpy_module);
        Py_DECREF(mean_func);
        Py_Finalize();
        return EXIT_FAILURE;
    }

    double mean_value = PyFloat_AsDouble(arrayRes);
    cout << "Mean value: " << mean_value << endl;

    // Clean up
    Py_DECREF(arrayRes);
    Py_DECREF(mean_func);
    Py_DECREF(py_array);
    Py_DECREF(numpy_module);

    Py_Finalize();

    return 0;
}