all: lod

CC=em++

UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
SHELL := /bin/bash
EMCC_HOME=/home/benjx/cs_wsl/emsdk
EMCC_ENV=emsdk_env.sh
LIBS=-fopenmp -I/usr/include/eigen3
CFLAGS=-s MODULARIZE -s EXPORT_NAME="LoD_WASM_Module" -std=c++17 -O3 --bind # I'm too lazy to set up C++20 on my machine
DEBUGCFLAGS=-Og -pipe -g -std=c++17
else
LIBS=-Xclang -fopenmp -lomp
CFLAGS=-O3 -pipe -std=c++20 -I/opt/homebrew/include/eigen3
DEBUGCFLAGS=-Og -pipe -g -std=c++20 -I/opt/homebrew/include/eigen3
endif

INPUT=lod.cpp
OUTPUT=lod.js

RM=/bin/rm

# Run this first
env:
	source $(EMCC_HOME)/$(EMCC_ENV)

lod: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(CFLAGS)
debug: $(INPUT)
	$(CC) $(INPUT) -o $(OUTPUT) $(LIBS) $(DEBUGCFLAGS)
clean: $(OUTPUT)
	if [ -e $(OUTPUT) ]; then $(RM) $(OUTPUT); fi
