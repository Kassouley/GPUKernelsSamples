CC := hipcc
TARGET := hip_api
INCLUDES := -I. -I../../common
CFLAGS := -O3 $(INCLUDES)

SRCS = main.cpp driver.cpp kernel.cpp ../../common/matrix.cpp

all: $(TARGET)

hip_api: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ 

clean:
	rm -rf $(TARGET)

.PHONY: all clean
