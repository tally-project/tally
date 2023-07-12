all: build

build: FORCE
	mkdir -p build
	cd build && cmake -DENABLE_LOGGING=1 .. && make -j

build-verbose: FORCE
	mkdir -p build
	cd build && cmake -DENABLE_LOGGING=1 .. && make -j

FORCE: ;