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
    import_array(); 

   
    npy_intp dims[2] = {3, 4}; 
    PyObject* py_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    if (!py_array) {
        PyErr_Print();
        Py_Finalize();
        return EXIT_FAILURE;
    }

   
    double* data = static_cast<double*>(PyArray_DATA(py_array));
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            *(data + i * dims[1] + j) = (i + 1) * (j + 1); 
        }
    }

    
    std::cout << "Arrays:" << std::endl;
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            std::cout << *(data + i * dims[1] + j) << " ";
        }
        std::cout << std::endl;
    }

    
    PyObject* max_module = PyImport_ImportModule("numpy");
    if (!max_module) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_Finalize();
        return EXIT_FAILURE;
    }

    PyObject* max_func = PyObject_GetAttrString(max_module, "max");
    Py_DECREF(max_module);

    if (!max_func) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_Finalize();
        return EXIT_FAILURE;
    }

    PyObject* max_args = Py_BuildValue("(O)", py_array);
    PyObject* max_result = PyObject_CallObject(max_func, max_args);
    Py_DECREF(max_args);

    if (!max_result) {
        PyErr_Print();
        Py_DECREF(max_func);
        Py_DECREF(py_array);
        Py_Finalize();
        return EXIT_FAILURE;
    }

    double max_value = PyFloat_AsDouble(max_result);
    std::cout << "Max Value: " << max_value << std::endl;

    
    Py_DECREF(max_result);
    Py_DECREF(max_func);
    Py_DECREF(py_array);

    Py_Finalize();

    

    return 0;
}