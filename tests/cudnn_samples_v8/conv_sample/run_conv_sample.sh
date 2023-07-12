#Use the following arguments to run sample with different convolution parameters:
./conv_sample -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
./conv_sample -c512 -h28 -w28 -k128 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
./conv_sample -c512 -h28 -w28 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
./conv_sample -c512 -h28 -w28 -k256 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
./conv_sample -c256 -h14 -w14 -k256 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
./conv_sample -c256 -h14 -w14 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
./conv_sample -c1024 -h14 -w14 -k256 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
./conv_sample -c1024 -h14 -w14 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
./conv_sample -c1024 -h14 -w14 -k512 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
./conv_sample -c512 -h7 -w7 -k512 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
./conv_sample -c512 -h7 -w7 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
./conv_sample -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1

#Use the following arguments to run sample with int8x4 and int8x32 benchmarks:
./conv_sample -mathType1 -filterFormat2 -dataType2 -n1 -c512 -h100 -w100 -k64 -r8 -s8 -pad_h0 -pad_w0 -u1 -v1 -b
./conv_sample -mathType1 -filterFormat2 -dataType2 -n1 -c4096 -h64 -w64 -k64 -r4 -s4 -pad_h1 -pad_w1 -u1 -v1 -b
./conv_sample -mathType1 -filterFormat2 -dataType2 -n1 -c512 -h100 -w100 -k64 -r8 -s8 -pad_h1 -pad_w1 -u1 -v1 -b
./conv_sample -mathType1 -filterFormat2 -dataType2 -n1 -c512 -h128 -w128 -k64 -r13 -s13 -pad_h1 -pad_w1 -u1 -v1 -b

./conv_sample -mathType1 -filterFormat2 -dataType3 -n1 -c512 -h100 -w100 -k64 -r8 -s8 -pad_h0 -pad_w0 -u1 -v1 -b
./conv_sample -mathType1 -filterFormat2 -dataType3 -n1 -c4096 -h64 -w64 -k64 -r4 -s4 -pad_h1 -pad_w1 -u1 -v1 -b
./conv_sample -mathType1 -filterFormat2 -dataType3 -n1 -c512 -h100 -w100 -k64 -r8 -s8 -pad_h1 -pad_w1 -u1 -v1 -b
./conv_sample -mathType1 -filterFormat2 -dataType3 -n1 -c512 -h128 -w128 -k64 -r13 -s13 -pad_h1 -pad_w1 -u1 -v1 -b

./conv_sample -mathType1 -filterFormat2 -dataType3 -n5 -c32 -h16 -w16 -k32 -r5 -s5 -pad_h0 -pad_w0 -u1 -v1 -b -transformFromNCHW

#Use the following arguments to run sample dgrad with folding:
./conv_sample -dgrad -c1024 -h14 -w14 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2 -fold

