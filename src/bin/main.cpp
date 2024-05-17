#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <getopt.h>

#include "../libs/lexer/lexer.hpp"
#include "../libs/parser/parser.hpp"
#include "../libs/ast/ast.hpp"
#include "../libs/interpreter/interpreter.hpp"

using namespace std;

void show_tokens(const vector<Token>& tokens) {
    for (const auto& token : tokens) {
        cout << token << '\n';
    }
    cout << flush;
}

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " [options] [filename].pcl\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h,  --help     \n";
    std::cerr << "  -v,  --version  \n";
    std::cerr << "  -s,  --syntax \n";
    std::cerr << "  -d,  --data_types \n";
    std::cerr << "  -o,  --operators \n";
    std::cerr << "  -c,  --control_flow \n";
    std::cerr << "  -f,  --function\n";
    std::cerr << "  -C,  --classes\n";
    std::cerr << "  -I,  --import\n";
    std::cerr << "  -n,  --numpy\n";

}

int main(int argc, char* argv[]) {
    int opt;
    while ((opt = getopt(argc, argv, "hvsdocfCIn")) != -1) {
        switch (opt) {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'v':
                std::cout << "version: 1.0.0" << std::endl;
                return 0;
            
            case 's':
                std::cout << "1.Indentation" << std::endl;
                std::cout << " In Pycimen, code blocks are defined by indentation. " << std::endl;
                std::cout << " Indentation can be created using spaces or tab characters, but mixed use within the same block is not allowed. " << std::endl;
                std::cout << "Correct Usage:" << std::endl;
                std::cout << " if x > 0: " << std::endl;
                std::cout << "     print(\"Positive\")  # Correct " << std::endl;
                std::cout << "     print(\"Value\")" << std::endl;
                std::cout << "Incorrect Usage (Mixed Indentation):" << std::endl;
                std::cout << " if x > 0: " << std::endl;
                std::cout << "     print(\"Positive\")  # Incorrect " << std::endl;
                std::cout << "               print(\"Value\")" << std::endl;
                std::cout << "2.Line Breaks" << std::endl;
                std::cout <<" In Pycimen, many statements can be written on a single line, but for longer statements, multiple lines can be used."<< std::endl;
                std::cout <<" The backslash () character is used for this purpose." << std::endl;
                std::cout << "Example:" << std::endl;
                std::cout << "x = 1 + 2 + \\" << std::endl;
                std::cout << "           3 + 4"<< std::endl;
                std::cout << "3. Comment Lines"<< std::endl;
                std::cout << "Single-line comments start with the # character."<< std::endl;
                std::cout << "4. Multiline Comments/Docstrings"<< std::endl;
                return 0;  

            case 'd':
                std::cout << "Data Types:" << std::endl;
                std::cout << "1.int - Integers" << std::endl;
                std::cout << "2.float - Floating-point numbers" << std::endl;
                std::cout << "3.string - String literals" << std::endl;
                std::cout << "4.boolean - True and False" << std::endl;
                std::cout << "5.None - Equivalent to Python's None" << std::endl;
                return 0;  

            case 'o':
                std::cout << "Operators:" << std::endl;
                std::cout << "Pycimen supports the following operators:" << std::endl;
                std::cout << "1. Arithmetic operators: +, -, *, /, %" << std::endl;
                std::cout << "2. Comparison operators: <, >, ==, !=, <=, >=" << std::endl;
                std::cout << "3. Logical operators: and, or, not" << std::endl;
                std::cout << "4. Bitwise operators: &, |, ^, <<, >>" << std::endl;
                return 0;

            case 'c':
                std::cout << "Control Flow:" << std::endl;
                std::cout << "Pycimen supports the following control flow statements:" << std::endl;
                std::cout << "1. if/elif/else" << std::endl;
                std::cout << "2. while loop" << std::endl;
                std::cout << "3. for loop" << std::endl;
                std::cout << "4. break" << std::endl;
                std::cout << "5. continue" << std::endl;
                std::cout << "6. pass" << std::endl;
                return 0;      
            case 'f':
                std::cout << "Functions:" << std::endl;
                std::cout << "Functions are defined with the def keyword and parameters are specified in parentheses." << std::endl;
                return 0;
            case 'C':
                std::cout << "Classes:" << std::endl;
                std::cout << "Pycimen supports class definition with the class keyword." << std::endl;
                return 0;

            case 'I':
                std::cout << "Import:" << std::endl;
                std::cout << "In Pycimen, functions or classes from other modules can be imported using the import statement. This facilitates code reuse and modularity." << std::endl;
                std::cout << "Note: In Pycimen, user-defined modules can be imported in addition to standard modules. The module name should be used without the file extension." << std::endl;
                return 0;  
            case 'n':
                std::cout << "Numpy:" << std::endl;
                std::cout << "The numpy library used in data analysis in Python is here, you can use it more easily by plus the c ++ speed. Currently, only the numpy library is added. Other libraries will be added in the future. " << std::endl;
                std::cout << "With this library, you can easily perform array ,mean, median, standard deviation operations on int, float type arrays." << std::endl;
            return 0;     
                           
            default:
                std::cerr << "Unknown option: " << opt << std::endl;
                return 1;
        }
    }

    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* const filename = argv[1];
    ifstream inputFile(filename);

    if (!inputFile) {
        std::cerr << "Error: could not open file '" << filename << "': " << strerror(errno) << std::endl;
        return 1;
    }

    string source((istreambuf_iterator<char>(inputFile)), {});

    Py_Initialize();

    import_array();

    try {
        Lexer lexer(source);
        vector<Token> tokens = lexer.scanTokens();

        #ifdef DEBUG
            show_tokens(tokens);
        #endif

        Parser parser(tokens);
        ProgramNode* root = parser.parse();

        Interpreter interpreter;
        interpreter.interpret(root);

    } catch (const runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return EXIT_FAILURE;
    }

    Py_Finalize();

    return 0;
}