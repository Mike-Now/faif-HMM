CC=clang++
INCLUDE_DIRS=-I${CURDIR}/../src/learning  -I/usr/local/boost_1_57_0
CFLAGS = -Wall -Werror -Wmissing-prototypes $(INCLUDE_DIRS) -std=c++11
LDFLAGS=-L/usr/local/boost_1_57_0/stage/lib -lboost_system
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
TARGET=test_mlr
 
all: $(TARGET)
 
$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS)  -o $@ $^
	./test_mlr	 
%.o: %.cpp %.hpp
	$(CC) $(CFLAGS) -c $<
 
%.o: %.cpp
	$(CC) $(CFLAGS) -c $<
 
clean:
	rm -f *.o $(TARGET)
