CXX = g++
CXXFLAGS = -std=c++20 -O3 -fPIC
INCLUDES = -I/opt/homebrew/include/eigen3 -I.

SRCS = l1_norm.cpp solver.cpp partition_matrix.cpp objectives.cpp
TARGET = lib_cpp.so

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
