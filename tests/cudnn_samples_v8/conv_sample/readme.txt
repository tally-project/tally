This example demonstrates how to use CUDNN library calls cudnnConvolutionForward,
cudnnConvolutionBackwardData, and cudnnConvolutionBackwardFilter with the option
to enable Tensor Cores on Volta with cudnnSetConvolutionMathType.

1. Make sure cuda and cudnn are installed in the same directory.

2. Run make from the directory of the sample specifying the cuda installation path:
        make CUDA_PATH=<cuda installation path>

3. Use the following arguments to run sample with different convolution parameters:

        -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
        -c512 -h28 -w28 -k128 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
        -c512 -h28 -w28 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
        -c512 -h28 -w28 -k256 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
        -c256 -h14 -w14 -k256 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
        -c256 -h14 -w14 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
        -c1024 -h14 -w14 -k256 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
        -c1024 -h14 -w14 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
        -c1024 -h14 -w14 -k512 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
        -c512 -h7 -w7 -k512 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
        -c512 -h7 -w7 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
        -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1

4. Use the following arguments to run sample with int8x4 and int8x32 benchmarks:

          -mathType1 -filterFormat2 -n1 -c512 -h100 -w100 -k64 -r8 -s8 -pad_h0 -pad_w0 -u1 -v1 -b
          -mathType1 -filterFormat2 -n1 -c4096 -h64 -w64 -k64 -r4 -s4 -pad_h1 -pad_w1 -u1 -v1 -b
          -mathType1 -filterFormat2 -n1 -c512 -h100 -w100 -k64 -r8 -s8 -pad_h1 -pad_w1 -u1 -v1 -b
          -mathType1 -filterFormat2 -n1 -c512 -h128 -w128 -k64 -r13 -s13 -pad_h1 -pad_w1 -u1 -v1 -b

5. Use the following additional arguments to run the layer with a different setup:
        -mathType1     : enable Tensor Cores.
        -dataType0     : Data is represented as FLOAT
        -dataType1     : Data is represented as HALF
        -dataType2     : Data is represented as INT8x4
        -dataType3     : Data is represented as INT8x32
        -dgrad         : run cudnnConvolutionBackwardData() instead of cudnnConvolutionForward().
        -wgrad         : run cudnnConvolutionBackwardFilter() instead of cudnnConvolutionForward().
        -n<int>        : mini batch size. (use -b with large n)
        -b             : benchmark mode. Bypass the CPU correctness check.
        -filterFormat0 : Use tensor format CUDNN_TENSOR_NCHW (Default).
        -filterFormat1 : Use tensor format CUDNN_TENSOR_NHWC.
        -filterFormat2 : Use tensor format CUDNN_TENSOR_NCHW_VECT_C. Using this
                         format switches to int8x4 and int8x32 testing

6. Note that changing the "-filterFormat" flag will automatically switch to valid data types for
    that format. CUDNN_TENSOR_NCHW and CUDNN_TENSOR_NHWC support single and half precision
    tests, while CUDNN_TENSOR_NCHW_VECT_C supports int8x4 and int8x32 tests.

7. "-fold" flag is useful for strided cases, FFT algorithm is chosen for demo purposes, but it can be applied to
   other algorithms as well

8. Use the following arguments to run INT8x4 and INT8x32 convolution with reordered filter matrices.
          -mathType1 -filterFormat2 -dataType3 -n5 -c32 -h16 -w16 -k32 -r5 -s5 -pad_h0 -pad_w0 -u1 -v1 -b
          -mathType1 -filterFormat2 -dataType3 -n5 -c64 -h16 -w16 -k32 -r5 -s5 -pad_h0 -pad_w0 -u1 -v1 -b
          -mathType1 -filterFormat2 -dataType3 -n5 -c128 -h16 -w16 -k32 -r5 -s5 -pad_h0 -pad_w0 -u1 -v1 -b
          -mathType1 -filterFormat2 -dataType3 -n5 -c32 -h16 -w16 -k64 -r5 -s5 -pad_h0 -pad_w0 -u1 -v1 -b
          -mathType1 -filterFormat2 -dataType3 -n5 -c64 -h32 -w32 -k64 -r5 -s5 -pad_h0 -pad_w0 -u1 -v1 -b
          -mathType1 -filterFormat2 -dataType3 -n5 -c128 -h16 -w16 -k64 -r5 -s5 -pad_h0 -pad_w0 -u1 -v1 -b
          -mathType1 -filterFormat2 -dataType3 -n5 -c128 -h16 -w16 -k128 -r5 -s5 -pad_h0 -pad_w0 -u1 -v1 -b

9. Use the following arguments to transform NCHW data to NC/32H32W format. Dimension of input NCHW have been given
using n, c, h, w flags
        -n1 -c3 -h2 -w2 -transformFromNCHW
        -n1 -c18 -h2 -w2 -transformFromNCHW
        -n1 -c30 -h2 -w2 -transformFromNCHW

10. Use the following arguments to transform NC/32H32W data to NCHW format. Dimension of output NCHW have been given
using n, c, h, w flags
        -n1 -c3 -h2 -w2 -transformToNCHW
        -n1 -c18 -h2 -w2 -transformToNCHW
        -n1 -c30 -h2 -w2 -transformToNCHW