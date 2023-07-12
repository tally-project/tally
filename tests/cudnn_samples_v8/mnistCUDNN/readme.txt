Before users can build the samples, they should download and build "FreeImage" first. "FreeImage" can be downloaded from https://freeimage.sourceforge.io/download.html. 
With the FreeImage object code and header file, users can start building the samples after placing the files under the directory specified below.

* please place "FreeImage.h" under "./FreeImage/include"
* please place the object code under the directory described below

	* Linux / x86-64
	    ./FreeImage/lib/linux/x86_64/libfreeimage.a
	* Linux / POWER9
	    ./FreeImage/lib/linux/ppc64le/libfreeimage.a 
	* Linux / Arm64
	    ./FreeImage/lib/linux/aarch64/libfreeimage.a

After completing the steps above, users can build the samples by using the following command.

* Linux / x86-64
    make all TARGET_ARCH=x86_64
* Linux / POWER9
    make all TARGET_ARCH=ppc64le  
* Linux / Arm64
    make all TARGET_ARCH=aarch64

If users encounter undefined reference to `png_init_filter_functions_vsx`  or undefined reference to `png_init_filter_functions_neon` while compiling the samples, they should re-compile libfreeimage.a after adding the additional option to CFLAGS.

* If the error is "undefined reference to `png_init_filter_functions_vsx`"
	* Add -DPNG_POWERPC_VSX_OPT=0 to CFLAGS before compiling libfreeimage.a
* If the error is "undefined reference to `png_init_filter_functions_neon`"
	* Add -DPNG_ARM_NEON_OPT=0 to CFLAGS before compiling libfreeimage.a

This sample demonstrates how to use cuDNN library to implement forward pass
given a trained network.

The sample is based on "Training LeNet on MNIST with Caffe" tutorial, located
at http://caffe.berkeleyvision.org/. The network is identical with the exception 
of addition of LRN layer. All the network weights are obtained and exported
using Caffe.

Network layer topology:

1. Convolution
2. Pooling
3. Convolution
4. Pooling
5. Fully connected
6. Relu
7. LRN
8. Fully Connected
9. SoftMax

By default, the sample will classify three images, located in "data" directory
using precomputed network weights:
1) Two convolution layers and their bias: conv1.bias.bin conv1.bin conv2.bias.bin conv2.bin
2) Two fully connected layers and their bias: ip1.bias.bin ip1.bin ip2.bias.bin ip2.bin

Supported platforms: identical to cuDNN

How to run:

mnistCUDNN {<options>}
help                   : display this help
device=<int>           : set the device to run the sample
image=<name>           : classify specific image

New in version 3 release
fp16 (three ways of conversion: on host, on device using cuDNN, on device using CUDA)
Local Response Normalization (LRN)
Find fastest config (cudnnFindConvolutionForwardAlgorithm)
FFT convolution
Demonstrate Nd API (first available in cuDNN v2)
