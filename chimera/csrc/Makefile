CXX ?= clang++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -Werror -fPIC -ffast-math

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    TARGET = libchimera_eval.dylib
    ifeq ($(UNAME_M),arm64)
        CXXFLAGS += -arch arm64
    endif
else
    TARGET = libchimera_eval.so
endif

all: $(TARGET)

$(TARGET): chimera_eval.cpp chimera_eval.hpp
	$(CXX) $(CXXFLAGS) -shared -o $@ chimera_eval.cpp

clean:
	rm -f libchimera_eval.dylib libchimera_eval.so *.o

.PHONY: all clean

