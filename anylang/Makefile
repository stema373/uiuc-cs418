.PHONY: all build run clean debug

CXX = clang++
CXXFLAGS = -O3 -I.
LDFLAGS = -lpng
SRC = main.cpp uselibpng.c
OBJ = $(SRC:.cpp=.o)

all: build

build: program

program: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	clang $(CXXFLAGS) -c $< -o $@

run: program
	./program $(file)

clean:
	rm -f program *.o *.png

debug: CXXFLAGS=-g -I.
debug: program
