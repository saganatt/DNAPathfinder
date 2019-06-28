NVCC        := /usr/local/cuda/bin/nvcc -ccbin g++

NVCCFLAGS   := -m64 -arch=sm_35
CCFLAGS     :=
LDFLAGS     := -rpath=../lib -L ../lib -lcnpy -lz

ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      CCFLAGS += -DDEBUG=2
endif

ifdef print_dbg
      CCFLAGS += -DDEBUG=$(print_dbg)
endif

ALL_CCFLAGS := -std=c++11
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

INCLUDES  := -I ../inc -I .
LIBRARIES :=

all: build

build: particles

filesIO.o:filesIO.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

particleSystem.o:particleSystem.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

kernelWrappers.o:kernelWrappers.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

particles.o:particles.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

particles: filesIO.o particleSystem.o kernelWrappers.o particles.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./particles

clean:
	rm -f particles particleSystem.o kernelWrappers.o particles.o filesIO.o