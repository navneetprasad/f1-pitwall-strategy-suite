# Makefile for F1 Embedded C Telemetry & EKF Engine

CC ?= gcc
CFLAGS ?= -std=c99 -Wall -Wextra -Werror -pedantic -O3 -Iinclude -fPIC
LDFLAGS ?= -lm

SRC = src/f1_can_decoder.c src/f1_ekf_core.c
OBJ = $(SRC:.c=.o)
TEST_SRC = tests/test_f1_ecu.c
TARGET_LIB = libf1_ecu.so
TARGET_LIB_WIN = f1_ecu.dll
TEST_BIN = test_f1_ecu

.PHONY: all clean test shared

all: shared test

shared: $(OBJ)
ifeq ($(OS),Windows_NT)
	$(CC) -shared -o $(TARGET_LIB_WIN) $(OBJ) $(LDFLAGS)
else
	$(CC) -shared -o $(TARGET_LIB) $(OBJ) $(LDFLAGS)
endif

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(OBJ) $(TEST_SRC)
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(OBJ) $(LDFLAGS)

clean:
	rm -f $(OBJ) $(TEST_BIN) $(TARGET_LIB) $(TARGET_LIB_WIN) *.exe
