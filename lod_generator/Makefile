all: lod_generator

CC=c++

UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
LIBS=-fopenmp -I/usr/include/eigen3
CFLAGS=-O3 -pipe -std=c++17 # I'm too lazy to set up C++20 on my machine
DEBUGCFLAGS=-Og -pipe -g -std=c++17
else
LIBS=-Xclang -fopenmp -lomp
CFLAGS=-O3 -pipe -std=c++20 -I/opt/homebrew/include/eigen3
DEBUGCFLAGS=-Og -pipe -g -std=c++20 -I/opt/homebrew/include/eigen3
endif

INPUT=lod_generator.cpp
OUTPUT=lod_generator

RM=/bin/rm

lod_generator: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(CFLAGS)
debug: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(DEBUGCFLAGS)
clean: $(OUTPUT)
	if [ -e $(OUTPUT) ]; then $(RM) $(OUTPUT); fi
