all: build

build: FORCE
	cd third_party/nccl && make -j src.build
	mkdir -p build
	cd build && cmake .. && make -j

FORCE: ;