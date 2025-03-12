all: lod_generator

CC=c++

LIBS=-fopenmp -lgomp
CFLAGS=-O3 -pipe -std=c++20
DEBUGCFLAGS=-Og -pipe -g -std=c++20

INPUT=lod_generator.cpp
OUTPUT=lod_generator

RM=/bin/rm

lod_generator: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(CFLAGS)
debug: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(DEBUGCFLAGS)
clean: $(OUTPUT)
	if [ -e $(OUTPUT) ]; then $(RM) $(OUTPUT); fi
