all: build

build: FORCE
	mkdir -p build
	cd build && cmake .. && make -j

docker-build-base:
	sudo docker build --no-cache -t tally:base -f Dockerfile_base .

docker-push-base:
	sudo docker tag tally:base wzhao18/tally:base
	sudo docker push wzhao18/tally:base

docker-build:
	sudo docker build -t tally .

FORCE: ;