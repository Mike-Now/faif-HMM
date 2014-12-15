CC=clang++
INCLUDE_DIRS=.
CFLAGS = -Wall -Werror -Wmissing-prototypes -I$(INCLUDE_DIRS)
LDFLAGS=
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
TARGET=TravisTest
 
all: $(TARGET)
 
$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS)  -o $@ $^
	./TravisTest	 
%.o: %.cpp %.hpp
	$(CC) $(CFLAGS) -c $<
 
%.o: %.cpp
	$(CC) $(CFLAGS) -c $<
 
clean:
	rm -f *.o $(TARGET)C = gcc
