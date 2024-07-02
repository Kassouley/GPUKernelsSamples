CC := hipcc
TARGET := sgemm dgemm
INCLUDES := -I. -I../../common
CFLAGS := -O3 $(INCLUDES) -lrocblas
SRCDIR := .

SRCS = main.cpp driver.cpp ../../common/matrix.cpp

all: $(TARGET)

sgemm: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ 

dgemm: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ -D DOUBLE

clean:
	rm -rf $(TARGET)

.PHONY: all clean
