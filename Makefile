all: build

build: FORCE
	mkdir -p build
	cd build && cmake -DENABLE_LOGGING=OFF -DENABLE_PROFILING=OFF .. && make -j

build-profile: FORCE
	mkdir -p build
	cd build && cmake -DENABLE_PROFILING=ON -DRUN_LOCALLY=OFF .. && make -j

build-profile-offine: FORCE
	mkdir -p build
	cd build && cmake -DENABLE_PROFILING=ON -DRUN_LOCALLY=ON .. && make -j

build-verbose: FORCE
	mkdir -p build
	cd build && cmake -DENABLE_LOGGING=ON -DRUN_LOCALLY=OFF .. && make -j

FORCE: ;