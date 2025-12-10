CXX = g++
FLAGS = -Wall -O3 -std=c++14 -fopenmp
OBJECTS = src/arg_parser.o src/scheme.o src/addition_reducer.o src/scheme_reducer.o

all: ternary_addition_reducer

ternary_addition_reducer: $(OBJECTS)
	$(CXX) $(FLAGS) $(OBJECTS) main.cpp -o ternary_addition_reducer

%.o: %.cpp
	$(CXX) $(FLAGS) -c $< -o $@

clean:
	rm -f src/*.o