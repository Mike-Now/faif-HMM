CC=clang++
INCLUDE_DIRS=-I${CURDIR}/../src/learning  -I/usr/local/boost_1_57_0
CFLAGS = -Wall -Werror -Wmissing-prototypes $(INCLUDE_DIRS) -std=c++11
LDFLAGS=-L/usr/local/boost_1_57_0/stage/lib -lboost_system -lboost_serialization
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
TARGET= mlr MlrTest

DEPENDENCIES=$(filter-out $(TARGET:=.o), $(OBJECTS) )

all: $(TARGET)

mlr: $(DEPENDENCIES) mlr.o
	$(CC) $(LDFLAGS)  -o $@ $^
	./$@
MlrTest: $(DEPENDENCIES) MlrTest.o
	$(CC) $(LDFLAGS)  -o $@ $^
	./$@
%.o: %.cpp %.hpp
	$(CC) $(CFLAGS) -c $<

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o $(TARGET)
