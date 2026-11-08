CXX = g++
CXXFLAGS = -O3 -fopenmp -Wall -I.
BENCHFLAGS = $(CXXFLAGS)

all: nvxk.exe my_app.exe

nvxk.exe: main.cpp nvxk.cpp nvxk.h
	$(CXX) $(BENCHFLAGS) -o nvxk.exe main.cpp nvxk.cpp

my_app.exe: my_app.cpp nvxk.cpp nvxk.h
	$(CXX) $(CXXFLAGS) -o my_app.exe my_app.cpp nvxk.cpp

.PHONY: all clean
clean:
	rm -f nvxk.exe my_app.exe