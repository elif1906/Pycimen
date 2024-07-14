<div>
  <img src="picture.jpeg" alt="Python Logo" align="left" width="100" height="100">
  <h1>PyCimen Language</h1>
</div>

#Abstract
This work explores the design and implementation of a new hybrid programming language called PyCimen, which com- bines the flexibility of Python with the performance-oriented nature of C++. This study aims to examine the core concepts of programming language design and implementation. It includes the construction of essential components such as the Lexer, Parser, and Abstract syntax tree (AST), offering Python-like syntax executed in a highly optimized and effi- cient C++ runtime. PyCimen’s minimalistic syntax enhances the ease and simplicity of the language. PyCimen provides a convenient developer experience with PyCimen-LSP fea- tures such as code completion, error checking, and syntax highlighting. It includes a package manager and an embed- ded Python interpreter, enabling direct import of Python modules installed with PIP, thus leveraging the rich Python module ecosystem. Language development requires lexical analysis, parsing, and evaluation. Consequently, this work aims to promote the design and implementation of programming lan- guages, supporting their creation by providing fundamental knowledge of language design concepts. It also demonstrates the feasibility of PyCimen combining the strengths of Python and C++ into a single integrated language. The addition of new features will make PyCimen a viable language option in various fields


#PyCimen : A New Hybrid Programming Language
This new hybrid programming language aims to strengthen the basic principles of programming languages by creating a custom interpreter for PyCimen. Remarkably, PyCimen seamlessly integrates Python's vast library ecosystem, enabling effortless use of powerful tools across a variety of domains. Additionally, inheriting the performance-oriented nature of C++, PyCimen ensures efficient execution, succeeding in data analysis tasks where speed and productivity are vital. Furthermore, we can benefit from the features of the C++ language while eliminating its difficulties with the help of PyCimen. This approach offers an easy way to gain a new perspective.Additionally, thanks to this, Python machine learning libraries can be used in C++, and AI integration can be done more easily compared to the existing C++ libraries.


## Project Overview

The Python Interpreter project aims to develop a fully functional interpreter for the Python programming language from scratch, using C++ as the implementation language. This project serves as an impressive demonstration of technical expertise and deep understanding of both Python and C++.

### This project delves into the heart of Python by crafting a custom interpreter built in C++.

   + Language Parsing: The interpreter includes a robust parser capable of parsing Python source code and generating an abstract syntax tree (AST) representation.

   + Lexical Analysis: A comprehensive lexer/tokenizer module is implemented to break down the input source code into tokens, facilitating the subsequent parsing and interpretation stages.

   + Semantic Analysis: The interpreter performs semantic analysis on the parsed AST to detect and report any language-specific errors, such as undeclared variables or type mismatches.

   + Execution Engine: A powerful execution engine is designed to execute the parsed Python code, implementing the language's semantics and executing statements, expressions, and control flow constructs.

   + Standard Library Support: Efforts are made to provide support for a subset of Python's standard library modules, enabling the interpreter to execute a wide range of Python programs.

### Why This Project Matters

   * Become a Language Whisperer: Crafting a Python interpreter from scratch equips you with an intimate understanding of programming languages, their design principles, and the intricate workings of compilers and interpreters. It showcases your prowess in tackling complex projects and makes you a valuable asset to potential employers and collaborators.

   * Technical Virtuosity: This project offers a hands-on playground to hone your skills in language parsing, lexical analysis, semantic analysis, and execution engine design. These are highly sought-after skills in software development, particularly in areas like language design, compiler development, and performance optimization.

   * Level Up Your Curriculum: Adding this project to your resume or portfolio is a bold declaration of your dedication to continuous learning and mastery of advanced programming concepts. It sets you apart from the crowd and signals your passion for exploring the intricacies of computer science.

## Building

```bash
$ git clone https://github.com/elif1906/Pycimen
$ cd pycimen
$ make

```
## Running Tests

```bash
  $ make test
```

## File Running
# for window users
```bash
$ pycimen [filename.pcl]  
$ pycimen --version

```
# for linux/macos users
```bash
$ ./pycimen [filename.pcl]  
$ ./pycimen --version

```


