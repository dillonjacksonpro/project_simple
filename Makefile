# Conservative, override-friendly build for C++ + MPI + optional OpenMP.
# Examples:
#   make
#   make MODE=debug
#   make USE_OPENMP=0
#   make MPICXX=mpicxx
#   make MPI_CXXFLAGS='...'

TARGET ?= project_simple
SRC ?= main.cpp

# Use an MPI wrapper by default so MPI include/lib paths are toolchain-managed.
MPICXX ?= mpic++

MODE ?= release
STD ?= -std=c++17

# C++ warning and safety flags (strict, but not forcing -Werror by default).
WARN_CXX ?= \
	-Wall -Wextra -Wpedantic \
	-Wshadow -Wconversion -Wsign-conversion \
	-Wcast-qual -Wnon-virtual-dtor -Wold-style-cast \
	-Woverloaded-virtual -Wnull-dereference -Wdouble-promotion \
	-Wformat=2 -Wundef -Wimplicit-fallthrough

SECURITY_CXX ?= -D_FORTIFY_SOURCE=2 -fstack-protector-strong

# Conservative optimization profile.
ifeq ($(MODE),debug)
	OPT_CXX ?= -O0 -g3
else
	OPT_CXX ?= -O2 -g
endif

# MPI-specific extension points. These remain empty by default because MPICXX
# usually injects MPI paths/libs itself.
MPI_CPPFLAGS ?=
MPI_CXXFLAGS ?=
MPI_LDFLAGS ?=
MPI_LDLIBS ?=

# OpenMP is assumed to be available.
OPENMP_CXXFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?= -fopenmp

CPPFLAGS ?=
CPPFLAGS += $(MPI_CPPFLAGS) $(SECURITY_CXX)

CXXFLAGS ?=
CXXFLAGS += $(STD) $(OPT_CXX) $(WARN_CXX) $(MPI_CXXFLAGS)

LDFLAGS ?=
LDFLAGS += $(MPI_LDFLAGS)

LDLIBS ?=
LDLIBS += $(MPI_LDLIBS)

CXXFLAGS += $(OPENMP_CXXFLAGS)
LDFLAGS += $(OPENMP_LDFLAGS)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)
