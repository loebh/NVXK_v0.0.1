CXX = g++
CXXFLAGS = -O3 -fopenmp -Wall -I.
BENCHFLAGS = $(CXXFLAGS)
TESTFLAGS = -O2 -Wall -I.

all: nvxk.exe my_app.exe

test: nvxk-test.exe
	./nvxk-test.exe

nvxk.exe: main.cpp nvxk.cpp nvxk.h
	$(CXX) $(BENCHFLAGS) -o nvxk.exe main.cpp nvxk.cpp

my_app.exe: my_app.cpp nvxk.cpp nvxk.h
	$(CXX) $(CXXFLAGS) -o my_app.exe my_app.cpp nvxk.cpp

nvxk-test.exe: nvxk_test.cpp nvxk.cpp nvxk.h
	$(CXX) $(TESTFLAGS) -o nvxk-test.exe nvxk_test.cpp nvxk.cpp

.PHONY: all clean test
clean:
	rm -f nvxk.exe my_app.exe nvxk-test.exe