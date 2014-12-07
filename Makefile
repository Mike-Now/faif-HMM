CC = gcc
CFLAGS = -Wall -Werror -Wmissing-prototypes
OBJS = ./build/examples/dna.o ./build/examples/dtc.o ./build/examples/dtcdb.o ./build/examples/ea.o ./build/examples/hillclimb.o ./build/examples/knn.o ./build/examples/nbc.o ./build/examples/nbcdb.o ./build/examples/random.o ./build/examples/search.o
TARGET = project

all: $(TARGET)

$(TARGET): $(OBJS)
    $(CC) $(CFLAGS) $(OBJS) -o $(TARGET)
