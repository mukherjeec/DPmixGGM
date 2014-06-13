################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= DPmixGGM_SSS.exe
# Cuda source files (compiled with cudacc)
CUFILES_sm_13	:= DPmixGGM_SSS_main.cu     # For CUDA compute capability 1.3
#CUFILES_sm_20	:= DPmixGGM_SSS_main.cu		# For CUDA compute capability 2.0

#fastmath := 1
#maxregisters := 1
#ptxas := 1
#dbg := 1

# Rules and targets

ROOTDIR := /usr/local/cuda_sdk/C/common
ROOTBINDIR := .
BINDIR  := .
BINSUBDIR := .

#include common_unicorn.mk
include $(ROOTDIR)/common.mk

LIB += -I/usr/local/include -L/usr/local/lib -lgsl -lgslcblas -lm -lgomp
COMMONFLAGS += -Xcompiler
