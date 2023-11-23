all: build

build: FORCE
	cd third_party/nccl && make -j src.build
	cd third_party/nccl/ext-net/example && make
	mkdir -p build
	cd build && cmake .. && make -j

FORCE: ;