all: build

build: FORCE
	mkdir -p build
	cd build && cmake .. && make -j

FORCE: ;