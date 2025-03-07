all: k_means

CC=c++

LIBS=-fopenmp -lomp
CFLAGS=-O3 -pipe -std=c++20
DEBUGCFLAGS=-Og -pipe -g -std=c++20

INPUT=k_means.cpp
OUTPUT=k_means

RM=/bin/rm

k_means: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(CFLAGS)
debug: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(DEBUGCFLAGS)
clean: $(OUTPUT)
	if [ -e $(OUTPUT) ]; then $(RM) $(OUTPUT); fi
