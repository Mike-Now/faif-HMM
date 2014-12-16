CC=clang++
INCLUDE_DIRS=.
CFLAGS = -Wall -Werror -Wmissing-prototypes -I$(INCLUDE_DIRS)
LDFLAGS=
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
	rm -f *.o $(TARGET)C = gcc
