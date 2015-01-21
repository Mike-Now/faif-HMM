CC=clang++
INCLUDE_DIRS=-I${CURDIR}/../src/learning
CFLAGS = -Wall -Werror -Wmissing-prototypes $(INCLUDE_DIRS) -std=c++11
LDFLAGS=-lboost_system -lboost_serialization -lboost_unit_test_framework
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
TARGET= mlr MlrTest nbc

DEPENDENCIES=$(filter-out $(TARGET:=.o), $(OBJECTS) )

all: $(TARGET)

mlr: $(DEPENDENCIES) mlr.o
	$(CC) $(LDFLAGS)  -o $@ $^
	./$@
MlrTest: $(DEPENDENCIES) MlrTest.o
	$(CC) $(LDFLAGS)  -o $@ $^
	./$@

nbc: $(DEPENDENCIES) nbc.o
	$(CC) $(LDFLAGS)  -o $@ $^
	./$@
%.o: %.cpp %.hpp
	$(CC) $(CFLAGS) -c $<

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o $(TARGET)
