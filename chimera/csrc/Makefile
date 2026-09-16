CXX ?= clang++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -Werror -fPIC -ffast-math -arch arm64
TARGET = libchimera_eval.dylib

all: $(TARGET)

$(TARGET): chimera_eval.cpp chimera_eval.hpp
	$(CXX) $(CXXFLAGS) -shared -o $@ chimera_eval.cpp

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
