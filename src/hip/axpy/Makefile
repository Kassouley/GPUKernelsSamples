CC := hipcc
TARGET := saxpy daxpy
INCLUDES := -I. -I../../common
CFLAGS := -O3 $(INCLUDES)
SRCDIR := .

SRCS = main.cpp driver.cpp kernel.cpp ../../common/matrix.cpp


all: $(TARGET)

saxpy: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ 

daxpy: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ -D DOUBLE

clean:
	rm -rf $(TARGET) $(BUILDDIR)

.PHONY: all clean
