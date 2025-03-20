all: bvh_wasm

CC=em++

UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
SHELL := /bin/bash
EMCC_HOME=/home/benjx/cs_wsl/emsdk
EMCC_ENV=emsdk_env.sh
LIBS=-I/usr/include/eigen3
CFLAGS=-std=c++17 -O3 -DEIGEN_DONT_PARALLELIZE -s ASSERTIONS
 # I'm too lazy to set up C++20 on my machine
EMFLAGS=-s MODULARIZE -s EXPORT_NAME="bvh_WASM_Module" -s ENVIRONMENT=worker --bind -s INITIAL_MEMORY=268435456 -s ALLOW_MEMORY_GROWTH=1
EXPORTS=-s EXPORTED_FUNCTIONS="['_malloc', '_free']"
DEBUGCFLAGS=-Og -pipe -g -std=c++17
else
LIBS=-Xclang -fopenmp -lomp
CFLAGS=-O3 -pipe -std=c++20 -I/opt/homebrew/include/eigen3
DEBUGCFLAGS=-Og -pipe -g -std=c++20 -I/opt/homebrew/include/eigen3
endif

INPUT=bvh_wasm.cpp
OUTPUT=bvh_wasm.js

RM=/bin/rm

# Run this first
env:
	source $(EMCC_HOME)/$(EMCC_ENV)

bvh_wasm: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(CFLAGS) $(EMFLAGS) $(EXPORTS)
debug: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(DEBUGCFLAGS)
clean: $(OUTPUT)
	if [ -e $(OUTPUT) ]; then $(RM) $(OUTPUT); fi
