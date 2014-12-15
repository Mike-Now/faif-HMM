CC=clang++
CFLAGS = -Wall -Werror -Wmissing-prototypes
LDFLAGS=
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
TARGET=TravisTest
 
all: $(TARGET)
 
$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS)  -o $@ $^
 
%.o: %.cpp %.hpp
	$(CC) $(CFLAGS) -c $<
 
%.o: %.cpp
	$(CC) $(CFLAGS) -c $<
 
clean:
	rm -f *.o $(TARGET)C = gcc
