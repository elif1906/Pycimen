# Get a list of all .cpp files in the root directory and its subdirectories
SRCS := $(shell find . -name "*.cpp")

# Set the output file name
OUTPUT :=  pycimen

# Set the compiler and compiler flags
CC := g++
CFLAGS := -std=c++17
CFLAGS += -I/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11 -Wsign-compare -Wunreachable-code -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk
CFLAGS += -L/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/config-3.11-darwin -lpython3.11 -ldl -framework CoreFoundation
CFLAGS += -I/opt/homebrew/lib/python3.11/site-packages/numpy/core/include
# Add DEBUG macro to CFLAGS
ifdef DEBUG
CFLAGS += -DDEBUG -Wall -Wextra -g
CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-reorder
endif

# Build rule
build: $(OUTPUT)

# Linking rule
$(OUTPUT): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(OUTPUT)

test:
	chmod +x test.sh
	./test.sh

# Clean rule
clean:
	@rm -f $(OUTPUT)
