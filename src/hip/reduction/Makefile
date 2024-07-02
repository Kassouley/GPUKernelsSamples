CC := hipcc
TARGET := simple_reduction double_reduction
INCLUDES := -I. -I../../common
CFLAGS := -O3 $(INCLUDES)
SRCDIR := .

SRCS = main.cpp driver.cpp kernel.cpp ../../common/matrix.cpp

all: $(TARGET)

simple_reduction: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ 

double_reduction: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ -D DOUBLE

clean:
	rm -rf $(TARGET)

.PHONY: all clean
