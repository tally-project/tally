all: build

build: FORCE
	mkdir -p build
	cd build && cmake .. && make -j

build-verbose: FORCE
	mkdir -p build
	cd build && cmake -DENABLE_LOGGING=ON .. && make -j

FORCE: ;