.PHONY: all build run clean debug

CXX = clang++
CXXFLAGS = -O3 -march=native -ffast-math -funroll-loops -flto -fno-exceptions -fno-rtti -I.
CFLAGS = -O3 -march=native -ffast-math -funroll-loops -flto
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
	clang $(CFLAGS) -c $< -o $@

run: program
	./program $(file)

clean:
	rm -f program *.o
	find . -type f -name "*.png" ! -name "earth.png" ! -name "moon.png" -delete

debug: CXXFLAGS=-g -I.
debug: program
