CUDA_INSTALL_PATH ?= /usr/local/cuda
GCC_VER = -5

CXX := /usr/bin/g++$(GCC_VER)
CC := /usr/bin/gcc$(GCC_VER)
LINK := $(CXX) -fPIC
CCPATH := ./gcc
NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc -ccbin $(CCPATH)

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Libraries
LIB_CUDA := -L/usr/lib/nvidia-current -lcuda

# Options
NVCCOPTIONS = -arch sm_20 -ptx -Wno-deprecated-gpu-targets
CXXOPTIONS = -std=c++17 -O2

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)
CXXFLAGS += $(COMMONFLAGS) $(CXXOPTIONS)
CFLAGS += $(COMMONFLAGS)

CUDA_OBJS = bfsCUDA.ptx
OBJS = main.cpp.o graph.cpp.o bfsCPU.cpp.o
TARGET = main
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

.SUFFIXES:	.c	.cpp	.cu	.o
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): prepare $(OBJS) $(CUDA_OBJS)
	$(LINKLINE)

clean:
	rm -rf $(TARGET) *.o *.ptx

prepare:
	rm -rf $(CCPATH);\
	mkdir -p $(CCPATH);\
	ln -s $(CXX) $(CCPATH)/g++;\
	ln -s $(CC) $(CCPATH)/gcc;

