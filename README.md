# Tally

## Build
Follow the [Dockerfile](https://github.com/wzhao18/tally/blob/master/Dockerfile) as a guide for build instructions.

cuDNN is required and can be downloaded from https://developer.nvidia.com/cudnn.

## Usage

Launch iox-roudi:
```sh
$ ./build/iox-roudi
```

Start tally server:
```sh
$ ./start_server
```

Start a client process:
```sh
$ ./start_client ./build/test/elementwise
```
