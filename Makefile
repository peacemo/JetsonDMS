CC=gcc
CFLAGS=-I/usr/include/
LDFLAGS=-L/usr/lib/x86_64-linux-gnu/
TARGET=dms
SRC=src/main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
