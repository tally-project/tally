import re

# Algorithm:
# 1. Categorize every code block into basic blocks
#   -- Named basic blocks are identified by starting with $BLOCK_NAME:
#   -- Whatever follows branch instruction is the start of a basic block

ptx_body = """
.reg .pred %p<118>;
.reg .b16 %rs<4>;
.reg .f32 %f<1025>;
.reg .b32 %r<686>;
.reg .b64 %rd<154>;


mov.b64 %rd23, _ZN7cutlass6KernelINS_4gemm6kernel18GemmSplitKParallelINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINS_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1ELb0ENSD_9NoPermuteEEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1ELb0ESJ_EENSL_ISQ_fSE_Li0ESS_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSM_fSE_fSE_NSW_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS15_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2ENS8_6thread14UnaryTransform8IdentityEEES1F_bEENS_8epilogue11threadblock8EpilogueIS7_S16_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1M_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfLb0ESJ_Lb0EEENS1H_4warp20FragmentIteratorSimtISY_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSM_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S14_EENS1R_16TileIteratorSimtISY_S1Y_fSE_S14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi4EEENS1H_6thread7ConvertIfLi1EfLS1B_2EEENSB_ILi0ELi17EEELi1ELi1EEENS4_38GemmSplitKHorizontalThreadblockSwizzleEEEEEvNT_6ParamsE_param_0;
mov.u64 %rd24, %rd23;
add.s64 %rd1, %rd24, 12;
ld.param.u32 %r121, [_ZN7cutlass6KernelINS_4gemm6kernel18GemmSplitKParallelINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINS_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1ELb0ENSD_9NoPermuteEEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1ELb0ESJ_EENSL_ISQ_fSE_Li0ESS_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSM_fSE_fSE_NSW_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS15_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2ENS8_6thread14UnaryTransform8IdentityEEES1F_bEENS_8epilogue11threadblock8EpilogueIS7_S16_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1M_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfLb0ESJ_Lb0EEENS1H_4warp20FragmentIteratorSimtISY_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSM_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S14_EENS1R_16TileIteratorSimtISY_S1Y_fSE_S14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi4EEENS1H_6thread7ConvertIfLi1EfLS1B_2EEENSB_ILi0ELi17EEELi1ELi1EEENS4_38GemmSplitKHorizontalThreadblockSwizzleEEEEEvNT_6ParamsE_param_0+12];
mov.u32 %r1, %ctaid.y;
setp.le.s32 %p1, %r121, %r1;
ld.param.u32 %r122, [_ZN7cutlass6KernelINS_4gemm6kernel18GemmSplitKParallelINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINS_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1ELb0ENSD_9NoPermuteEEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1ELb0ESJ_EENSL_ISQ_fSE_Li0ESS_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSM_fSE_fSE_NSW_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS15_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2ENS8_6thread14UnaryTransform8IdentityEEES1F_bEENS_8epilogue11threadblock8EpilogueIS7_S16_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1M_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfLb0ESJ_Lb0EEENS1H_4warp20FragmentIteratorSimtISY_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSM_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S14_EENS1R_16TileIteratorSimtISY_S1Y_fSE_S14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi4EEENS1H_6thread7ConvertIfLi1EfLS1B_2EEENSB_ILi0ELi17EEELi1ELi1EEENS4_38GemmSplitKHorizontalThreadblockSwizzleEEEEEvNT_6ParamsE_param_0+16];
mov.u32 %r2, %ctaid.x;
setp.le.s32 %p2, %r122, %r2;
or.pred %p3, %p1, %p2;
@%p3 bra $L__BB0_8;

mov.u32 %r167, %ctaid.z;
shl.b32 %r168, %r1, 7;
ld.param.u32 %r169, [%rd1+212];
mul.lo.s32 %r170, %r169, %r167;
ld.param.u32 %r171, [%rd1+8];
add.s32 %r172, %r167, 1;
setp.eq.s32 %p4, %r172, %r171;
add.s32 %r173, %r170, %r169;
ld.param.u32 %r174, [%rd1+-4];
selp.b32 %r175, %r174, %r173, %p4;
sub.s32 %r3, %r175, %r170;
ld.param.v2.u32 {%r176, %r177}, [%rd1+-12];
mov.u32 %r146, 0;
ld.param.u64 %rd2, [%rd1+28];
ld.param.u64 %rd3, [%rd1+36];
ld.param.u64 %rd4, [%rd1+44];
shr.s32 %r178, %r3, 31;
shr.u32 %r179, %r178, 29;
add.s32 %r180, %r3, %r179;
and.b32 %r181, %r180, -8;
sub.s32 %r182, %r3, %r181;
setp.eq.s32 %p5, %r182, 0;
selp.b32 %r183, 8, %r182, %p5;
add.s32 %r184, %r170, %r183;
min.s32 %r185, %r175, %r184;
mov.u32 %r186, %tid.x;
shr.s32 %r187, %r186, 31;
shr.u32 %r188, %r187, 29;
add.s32 %r189, %r186, %r188;
and.b32 %r190, %r189, -8;
sub.s32 %r191, %r186, %r190;
shr.s32 %r192, %r189, 3;
add.s32 %r193, %r191, %r170;
add.s32 %r194, %r192, %r168;
setp.lt.s32 %p6, %r194, %r176;
setp.lt.s32 %p7, %r193, %r185;
and.pred %p8, %p7, %p6;
selp.u32 %r124, 1, 0, %p8;
add.s32 %r195, %r194, 32;
setp.lt.s32 %p9, %r195, %r176;
and.pred %p10, %p7, %p9;
selp.u32 %r127, 1, 0, %p10;
add.s32 %r196, %r194, 64;
setp.lt.s32 %p11, %r196, %r176;
and.pred %p12, %p7, %p11;
selp.u32 %r130, 1, 0, %p12;
add.s32 %r197, %r194, 96;
setp.lt.s32 %p13, %r197, %r176;
and.pred %p14, %p7, %p13;
selp.u32 %r133, 1, 0, %p14;
cvt.s64.s32 %rd5, %r193;
cvt.s64.s32 %rd33, %r194;
ld.param.u64 %rd34, [%rd1+20];
mul.lo.s64 %rd6, %rd34, %rd33;
add.s64 %rd35, %rd6, %rd5;
shl.b64 %rd36, %rd35, 2;
ld.param.u64 %rd7, [%rd1+52];
add.s64 %rd25, %rd7, %rd36;
ld.param.u64 %rd8, [%rd1+76];
ld.param.u64 %rd9, [%rd1+84];
ld.param.u64 %rd10, [%rd1+92];
shr.u32 %r198, %r187, 25;
add.s32 %r199, %r186, %r198;
and.b32 %r200, %r199, -128;
sub.s32 %r201, %r186, %r200;
shr.s32 %r202, %r199, 7;
shl.b32 %r203, %r2, 7;
add.s32 %r204, %r201, %r203;
add.s32 %r205, %r202, %r170;
setp.lt.s32 %p15, %r205, %r185;
setp.lt.s32 %p16, %r204, %r177;
and.pred %p17, %p16, %p15;
selp.u32 %r136, 1, 0, %p17;
add.s32 %r206, %r205, 2;
setp.lt.s32 %p18, %r206, %r185;
and.pred %p19, %p16, %p18;
selp.u32 %r139, 1, 0, %p19;
add.s32 %r207, %r205, 4;
setp.lt.s32 %p20, %r207, %r185;
and.pred %p21, %p16, %p20;
selp.u32 %r142, 1, 0, %p21;
add.s32 %r208, %r205, 6;
setp.lt.s32 %p22, %r208, %r185;
and.pred %p23, %p16, %p22;
selp.u32 %r145, 1, 0, %p23;
cvt.s64.s32 %rd11, %r204;
cvt.s64.s32 %rd12, %r205;
ld.param.u64 %rd13, [%rd1+68];
mul.lo.s64 %rd37, %rd13, %rd12;
add.s64 %rd38, %rd37, %rd11;
shl.b64 %rd39, %rd38, 2;
ld.param.u64 %rd14, [%rd1+100];
add.s64 %rd29, %rd14, %rd39;
and.b32 %r209, %r186, 31;
shr.u32 %r210, %r209, 4;
and.b32 %r211, %r186, 1;
bfi.b32 %r212, %r210, %r211, 1, 31;
shl.b32 %r213, %r212, 4;
mov.u32 %r214, _ZN7cutlass17SharedStorageBaseE;
add.s32 %r215, %r214, %r213;
add.s32 %r216, %r214, 8448;
shl.b32 %r217, %r186, 3;
and.b32 %r218, %r217, 112;
add.s32 %r219, %r216, %r218;
mad.lo.s32 %r6, %r191, 132, %r192;
shl.b32 %r220, %r6, 2;
add.s32 %r221, %r214, %r220;
shl.b32 %r222, %r186, 2;
add.s32 %r7, %r216, %r222;
shr.u32 %r223, %r186, 8;
shr.u32 %r224, %r186, 2;
and.b32 %r225, %r224, 24;
shr.u32 %r226, %r186, 3;
and.b32 %r227, %r226, 16;
and.b32 %r228, %r186, 268435200;
or.b32 %r229, %r227, %r228;

	{
.reg .pred p;
setp.ne.b32 p, %r124, 0;
mov.b32 %r123, %r146;
@p ld.global.L2::128B.u32 %r123, [%rd25];
}


	add.s64 %rd26, %rd25, %rd2;

	{
.reg .pred p;
setp.ne.b32 p, %r127, 0;
mov.b32 %r126, %r146;
@p ld.global.L2::128B.u32 %r126, [%rd26];
}


	add.s64 %rd27, %rd26, %rd2;

	{
.reg .pred p;
setp.ne.b32 p, %r130, 0;
mov.b32 %r129, %r146;
@p ld.global.L2::128B.u32 %r129, [%rd27];
}


	add.s64 %rd28, %rd27, %rd2;

	{
.reg .pred p;
setp.ne.b32 p, %r133, 0;
mov.b32 %r132, %r146;
@p ld.global.L2::128B.u32 %r132, [%rd28];
}


	selp.u32 %r230, 1, 0, %p6;
selp.u32 %r231, -1, 0, %p9;
bfi.b32 %r232, %r231, %r230, 1, 1;
selp.u16 %rs1, 1, 0, %p11;
mul.wide.u16 %r233, %rs1, 4;
or.b32 %r234, %r233, %r232;
selp.u16 %rs2, 1, 0, %p13;
mul.wide.u16 %r235, %rs2, 8;
or.b32 %r8, %r235, %r234;
mul.wide.s32 %rd15, %r183, 4;

	{
.reg .pred p;
setp.ne.b32 p, %r136, 0;
mov.b32 %r135, %r146;
@p ld.global.L2::128B.u32 %r135, [%rd29];
}


	add.s64 %rd30, %rd29, %rd8;

	{
.reg .pred p;
setp.ne.b32 p, %r139, 0;
mov.b32 %r138, %r146;
@p ld.global.L2::128B.u32 %r138, [%rd30];
}


	add.s64 %rd31, %rd30, %rd8;

	{
.reg .pred p;
setp.ne.b32 p, %r142, 0;
mov.b32 %r141, %r146;
@p ld.global.L2::128B.u32 %r141, [%rd31];
}


	add.s64 %rd32, %rd31, %rd8;

	{
.reg .pred p;
setp.ne.b32 p, %r145, 0;
mov.b32 %r144, %r146;
@p ld.global.L2::128B.u32 %r144, [%rd32];
}


	selp.u32 %r236, 1, 0, %p16;
selp.u32 %r237, -1, 0, %p16;
bfi.b32 %r238, %r237, %r236, 1, 1;
selp.u16 %rs3, 1, 0, %p16;
mul.wide.u16 %r239, %rs3, 4;
or.b32 %r240, %r239, %r238;
mul.wide.u16 %r241, %rs3, 8;
or.b32 %r9, %r241, %r240;
st.shared.u32 [%r221], %r123;
st.shared.u32 [%r221+128], %r126;
st.shared.u32 [%r221+256], %r129;
st.shared.u32 [%r221+384], %r132;
st.shared.u32 [%r7], %r135;
st.shared.u32 [%r7+1024], %r138;
st.shared.u32 [%r7+2048], %r141;
st.shared.u32 [%r7+3072], %r144;
bar.sync 0;
mad.lo.s32 %r242, %r223, 264, %r225;
shl.b32 %r243, %r242, 4;
add.s32 %r151, %r215, %r243;

	ld.shared.v4.b32 {%r669, %r670, %r671, %r672}, [%r151];

	add.s32 %r156, %r151, 64;

	ld.shared.v4.b32 {%r673, %r674, %r675, %r676}, [%r156];

	add.s32 %r677, %r151, 528;
shl.b32 %r244, %r229, 4;
add.s32 %r161, %r219, %r244;

	ld.shared.v4.b32 {%r661, %r662, %r663, %r664}, [%r161];

	add.s32 %r166, %r161, 128;

	ld.shared.v4.b32 {%r665, %r666, %r667, %r668}, [%r166];

	add.s32 %r678, %r161, 512;
setp.lt.s32 %p24, %r3, 1;
mov.f32 %f961, 0f00000000;
mov.f32 %f962, %f961;
mov.f32 %f963, %f961;
mov.f32 %f964, %f961;
mov.f32 %f965, %f961;
mov.f32 %f966, %f961;
mov.f32 %f967, %f961;
mov.f32 %f968, %f961;
mov.f32 %f969, %f961;
mov.f32 %f970, %f961;
mov.f32 %f971, %f961;
mov.f32 %f972, %f961;
mov.f32 %f973, %f961;
mov.f32 %f974, %f961;
mov.f32 %f975, %f961;
mov.f32 %f976, %f961;
mov.f32 %f977, %f961;
mov.f32 %f978, %f961;
mov.f32 %f979, %f961;
mov.f32 %f980, %f961;
mov.f32 %f981, %f961;
mov.f32 %f982, %f961;
mov.f32 %f983, %f961;
mov.f32 %f984, %f961;
mov.f32 %f985, %f961;
mov.f32 %f986, %f961;
mov.f32 %f987, %f961;
mov.f32 %f988, %f961;
mov.f32 %f989, %f961;
mov.f32 %f990, %f961;
mov.f32 %f991, %f961;
mov.f32 %f992, %f961;
mov.f32 %f993, %f961;
mov.f32 %f994, %f961;
mov.f32 %f995, %f961;
mov.f32 %f996, %f961;
mov.f32 %f997, %f961;
mov.f32 %f998, %f961;
mov.f32 %f999, %f961;
mov.f32 %f1000, %f961;
mov.f32 %f1001, %f961;
mov.f32 %f1002, %f961;
mov.f32 %f1003, %f961;
mov.f32 %f1004, %f961;
mov.f32 %f1005, %f961;
mov.f32 %f1006, %f961;
mov.f32 %f1007, %f961;
mov.f32 %f1008, %f961;
mov.f32 %f1009, %f961;
mov.f32 %f1010, %f961;
mov.f32 %f1011, %f961;
mov.f32 %f1012, %f961;
mov.f32 %f1013, %f961;
mov.f32 %f1014, %f961;
mov.f32 %f1015, %f961;
mov.f32 %f1016, %f961;
mov.f32 %f1017, %f961;
mov.f32 %f1018, %f961;
mov.f32 %f1019, %f961;
mov.f32 %f1020, %f961;
mov.f32 %f1021, %f961;
mov.f32 %f1022, %f961;
mov.f32 %f1023, %f961;
mov.f32 %f1024, %f961;
@%p24 bra $L__BB0_7;

setp.gt.s32 %p25, %r3, 8;
add.s32 %r246, %r3, 7;
shr.s32 %r247, %r246, 31;
shr.u32 %r248, %r247, 29;
add.s32 %r249, %r246, %r248;
shr.s32 %r250, %r249, 3;
add.s32 %r680, %r7, 4096;
add.s32 %r679, %r221, 4224;
shl.b64 %rd40, %rd12, 2;
add.s64 %rd41, %rd15, %rd40;
mul.lo.s64 %rd42, %rd13, %rd41;
shl.b64 %rd43, %rd11, 2;
add.s64 %rd44, %rd9, %rd43;
add.s64 %rd153, %rd14, %rd44;
mul.lo.s64 %rd45, %rd8, 3;
add.s64 %rd46, %rd42, %rd45;
sub.s64 %rd17, %rd46, %rd10;
shl.b64 %rd47, %rd6, 2;
add.s64 %rd48, %rd3, %rd47;
add.s64 %rd49, %rd48, %rd15;
shl.b64 %rd50, %rd5, 2;
add.s64 %rd51, %rd49, %rd50;
sub.s64 %rd52, %rd51, %rd4;
add.s64 %rd152, %rd7, %rd52;
add.s32 %r658, %r250, 1;
mov.u32 %r681, 1;
selp.b32 %r659, %r9, 0, %p25;
selp.b32 %r660, %r8, 0, %p25;

$L__BB0_3:
.pragma "nounroll";
mul.lo.s64 %rd61, %rd2, 3;
add.s64 %rd53, %rd152, %rd61;

	ld.shared.v4.b32 {%r254, %r255, %r256, %r257}, [%r677];

	add.s32 %r263, %r677, 64;

	ld.shared.v4.b32 {%r259, %r260, %r261, %r262}, [%r263];

	
	ld.shared.v4.b32 {%r264, %r265, %r266, %r267}, [%r678];

	add.s32 %r273, %r678, 128;

	ld.shared.v4.b32 {%r269, %r270, %r271, %r272}, [%r273];

	mov.u32 %r297, 0;
and.b32 %r275, %r660, 1;

	{
.reg .pred p;
setp.ne.b32 p, %r275, 0;
mov.b32 %r274, %r297;
@p ld.global.L2::128B.u32 %r274, [%rd53];
}


	and.b32 %r418, %r660, 2;
shr.u32 %r278, %r418, 1;
add.s64 %rd54, %rd53, %rd2;

	{
.reg .pred p;
setp.ne.b32 p, %r278, 0;
mov.b32 %r277, %r297;
@p ld.global.L2::128B.u32 %r277, [%rd54];
}


	and.b32 %r419, %r660, 4;
shr.u32 %r281, %r419, 2;
add.s64 %rd55, %rd54, %rd2;

	{
.reg .pred p;
setp.ne.b32 p, %r281, 0;
mov.b32 %r280, %r297;
@p ld.global.L2::128B.u32 %r280, [%rd55];
}


	and.b32 %r420, %r660, 8;
shr.u32 %r284, %r420, 3;
add.s64 %rd56, %rd55, %rd2;

	{
.reg .pred p;
setp.ne.b32 p, %r284, 0;
mov.b32 %r283, %r297;
@p ld.global.L2::128B.u32 %r283, [%rd56];
}


	and.b32 %r287, %r659, 1;
add.s64 %rd57, %rd153, %rd17;

	{
.reg .pred p;
setp.ne.b32 p, %r287, 0;
mov.b32 %r286, %r297;
@p ld.global.L2::128B.u32 %r286, [%rd57];
}


	and.b32 %r421, %r659, 2;
shr.u32 %r290, %r421, 1;
add.s64 %rd58, %rd57, %rd8;

	{
.reg .pred p;
setp.ne.b32 p, %r290, 0;
mov.b32 %r289, %r297;
@p ld.global.L2::128B.u32 %r289, [%rd58];
}


	and.b32 %r422, %r659, 4;
shr.u32 %r293, %r422, 2;
add.s64 %rd59, %rd58, %rd8;

	{
.reg .pred p;
setp.ne.b32 p, %r293, 0;
mov.b32 %r292, %r297;
@p ld.global.L2::128B.u32 %r292, [%rd59];
}


	and.b32 %r423, %r659, 8;
shr.u32 %r296, %r423, 3;
add.s64 %rd60, %rd59, %rd8;

	{
.reg .pred p;
setp.ne.b32 p, %r296, 0;
mov.b32 %r295, %r297;
@p ld.global.L2::128B.u32 %r295, [%rd60];
}


	mov.b32 %f385, %r661;
mov.b32 %f386, %r669;
fma.rn.f32 %f387, %f386, %f385, %f1024;
mov.b32 %f388, %r670;
fma.rn.f32 %f389, %f388, %f385, %f1016;
mov.b32 %f390, %r662;
fma.rn.f32 %f391, %f388, %f390, %f1015;
fma.rn.f32 %f392, %f386, %f390, %f1023;
mov.b32 %f393, %r671;
fma.rn.f32 %f394, %f393, %f385, %f1008;
mov.b32 %f395, %r672;
fma.rn.f32 %f396, %f395, %f385, %f1000;
fma.rn.f32 %f397, %f395, %f390, %f999;
fma.rn.f32 %f398, %f393, %f390, %f1007;
mov.b32 %f399, %r673;
fma.rn.f32 %f400, %f399, %f385, %f992;
mov.b32 %f401, %r674;
fma.rn.f32 %f402, %f401, %f385, %f984;
fma.rn.f32 %f403, %f401, %f390, %f983;
fma.rn.f32 %f404, %f399, %f390, %f991;
mov.b32 %f405, %r675;
fma.rn.f32 %f406, %f405, %f385, %f976;
mov.b32 %f407, %r676;
fma.rn.f32 %f408, %f407, %f385, %f968;
fma.rn.f32 %f409, %f407, %f390, %f967;
fma.rn.f32 %f410, %f405, %f390, %f975;
mov.b32 %f411, %r663;
fma.rn.f32 %f412, %f405, %f411, %f974;
fma.rn.f32 %f413, %f407, %f411, %f966;
mov.b32 %f414, %r664;
fma.rn.f32 %f415, %f407, %f414, %f965;
fma.rn.f32 %f416, %f405, %f414, %f973;
fma.rn.f32 %f417, %f399, %f411, %f990;
fma.rn.f32 %f418, %f401, %f411, %f982;
fma.rn.f32 %f419, %f401, %f414, %f981;
fma.rn.f32 %f420, %f399, %f414, %f989;
fma.rn.f32 %f421, %f393, %f411, %f1006;
fma.rn.f32 %f422, %f395, %f411, %f998;
fma.rn.f32 %f423, %f395, %f414, %f997;
fma.rn.f32 %f424, %f393, %f414, %f1005;
fma.rn.f32 %f425, %f386, %f411, %f1022;
fma.rn.f32 %f426, %f388, %f411, %f1014;
fma.rn.f32 %f427, %f388, %f414, %f1013;
fma.rn.f32 %f428, %f386, %f414, %f1021;
mov.b32 %f429, %r665;
fma.rn.f32 %f430, %f386, %f429, %f1020;
fma.rn.f32 %f431, %f388, %f429, %f1012;
mov.b32 %f432, %r666;
fma.rn.f32 %f433, %f388, %f432, %f1011;
fma.rn.f32 %f434, %f386, %f432, %f1019;
fma.rn.f32 %f435, %f393, %f429, %f1004;
fma.rn.f32 %f436, %f395, %f429, %f996;
fma.rn.f32 %f437, %f395, %f432, %f995;
fma.rn.f32 %f438, %f393, %f432, %f1003;
fma.rn.f32 %f439, %f399, %f429, %f988;
fma.rn.f32 %f440, %f401, %f429, %f980;
fma.rn.f32 %f441, %f401, %f432, %f979;
fma.rn.f32 %f442, %f399, %f432, %f987;
fma.rn.f32 %f443, %f405, %f429, %f972;
fma.rn.f32 %f444, %f407, %f429, %f964;
fma.rn.f32 %f445, %f407, %f432, %f963;
fma.rn.f32 %f446, %f405, %f432, %f971;
mov.b32 %f447, %r667;
fma.rn.f32 %f448, %f405, %f447, %f970;
fma.rn.f32 %f449, %f407, %f447, %f962;
mov.b32 %f450, %r668;
fma.rn.f32 %f451, %f407, %f450, %f961;
fma.rn.f32 %f452, %f405, %f450, %f969;
fma.rn.f32 %f453, %f399, %f447, %f986;
fma.rn.f32 %f454, %f401, %f447, %f978;
fma.rn.f32 %f455, %f401, %f450, %f977;
fma.rn.f32 %f456, %f399, %f450, %f985;
fma.rn.f32 %f457, %f393, %f447, %f1002;
fma.rn.f32 %f458, %f395, %f447, %f994;
fma.rn.f32 %f459, %f395, %f450, %f993;
fma.rn.f32 %f460, %f393, %f450, %f1001;
fma.rn.f32 %f461, %f386, %f447, %f1018;
fma.rn.f32 %f462, %f388, %f447, %f1010;
fma.rn.f32 %f463, %f388, %f450, %f1009;
fma.rn.f32 %f464, %f386, %f450, %f1017;
add.s32 %r302, %r677, 528;

	ld.shared.v4.b32 {%r298, %r299, %r300, %r301}, [%r302];

	add.s32 %r307, %r677, 592;

	ld.shared.v4.b32 {%r303, %r304, %r305, %r306}, [%r307];

	add.s32 %r312, %r678, 512;

	ld.shared.v4.b32 {%r308, %r309, %r310, %r311}, [%r312];

	add.s32 %r317, %r678, 640;

	ld.shared.v4.b32 {%r313, %r314, %r315, %r316}, [%r317];

	mov.b32 %f465, %r254;
mov.b32 %f466, %r264;
fma.rn.f32 %f467, %f465, %f466, %f387;
mov.b32 %f468, %r255;
fma.rn.f32 %f469, %f468, %f466, %f389;
mov.b32 %f470, %r265;
fma.rn.f32 %f471, %f468, %f470, %f391;
fma.rn.f32 %f472, %f465, %f470, %f392;
mov.b32 %f473, %r256;
fma.rn.f32 %f474, %f473, %f466, %f394;
mov.b32 %f475, %r257;
fma.rn.f32 %f476, %f475, %f466, %f396;
fma.rn.f32 %f477, %f475, %f470, %f397;
fma.rn.f32 %f478, %f473, %f470, %f398;
mov.b32 %f479, %r259;
fma.rn.f32 %f480, %f479, %f466, %f400;
mov.b32 %f481, %r260;
fma.rn.f32 %f482, %f481, %f466, %f402;
fma.rn.f32 %f483, %f481, %f470, %f403;
fma.rn.f32 %f484, %f479, %f470, %f404;
mov.b32 %f485, %r261;
fma.rn.f32 %f486, %f485, %f466, %f406;
mov.b32 %f487, %r262;
fma.rn.f32 %f488, %f487, %f466, %f408;
fma.rn.f32 %f489, %f487, %f470, %f409;
fma.rn.f32 %f490, %f485, %f470, %f410;
mov.b32 %f491, %r266;
fma.rn.f32 %f492, %f485, %f491, %f412;
fma.rn.f32 %f493, %f487, %f491, %f413;
mov.b32 %f494, %r267;
fma.rn.f32 %f495, %f487, %f494, %f415;
fma.rn.f32 %f496, %f485, %f494, %f416;
fma.rn.f32 %f497, %f479, %f491, %f417;
fma.rn.f32 %f498, %f481, %f491, %f418;
fma.rn.f32 %f499, %f481, %f494, %f419;
fma.rn.f32 %f500, %f479, %f494, %f420;
fma.rn.f32 %f501, %f473, %f491, %f421;
fma.rn.f32 %f502, %f475, %f491, %f422;
fma.rn.f32 %f503, %f475, %f494, %f423;
fma.rn.f32 %f504, %f473, %f494, %f424;
fma.rn.f32 %f505, %f465, %f491, %f425;
fma.rn.f32 %f506, %f468, %f491, %f426;
fma.rn.f32 %f507, %f468, %f494, %f427;
fma.rn.f32 %f508, %f465, %f494, %f428;
mov.b32 %f509, %r269;
fma.rn.f32 %f510, %f465, %f509, %f430;
fma.rn.f32 %f511, %f468, %f509, %f431;
mov.b32 %f512, %r270;
fma.rn.f32 %f513, %f468, %f512, %f433;
fma.rn.f32 %f514, %f465, %f512, %f434;
fma.rn.f32 %f515, %f473, %f509, %f435;
fma.rn.f32 %f516, %f475, %f509, %f436;
fma.rn.f32 %f517, %f475, %f512, %f437;
fma.rn.f32 %f518, %f473, %f512, %f438;
fma.rn.f32 %f519, %f479, %f509, %f439;
fma.rn.f32 %f520, %f481, %f509, %f440;
fma.rn.f32 %f521, %f481, %f512, %f441;
fma.rn.f32 %f522, %f479, %f512, %f442;
fma.rn.f32 %f523, %f485, %f509, %f443;
fma.rn.f32 %f524, %f487, %f509, %f444;
fma.rn.f32 %f525, %f487, %f512, %f445;
fma.rn.f32 %f526, %f485, %f512, %f446;
mov.b32 %f527, %r271;
fma.rn.f32 %f528, %f485, %f527, %f448;
fma.rn.f32 %f529, %f487, %f527, %f449;
mov.b32 %f530, %r272;
fma.rn.f32 %f531, %f487, %f530, %f451;
fma.rn.f32 %f532, %f485, %f530, %f452;
fma.rn.f32 %f533, %f479, %f527, %f453;
fma.rn.f32 %f534, %f481, %f527, %f454;
fma.rn.f32 %f535, %f481, %f530, %f455;
fma.rn.f32 %f536, %f479, %f530, %f456;
fma.rn.f32 %f537, %f473, %f527, %f457;
fma.rn.f32 %f538, %f475, %f527, %f458;
fma.rn.f32 %f539, %f475, %f530, %f459;
fma.rn.f32 %f540, %f473, %f530, %f460;
fma.rn.f32 %f541, %f465, %f527, %f461;
fma.rn.f32 %f542, %f468, %f527, %f462;
fma.rn.f32 %f543, %f468, %f530, %f463;
fma.rn.f32 %f544, %f465, %f530, %f464;
add.s32 %r322, %r677, 1056;

	ld.shared.v4.b32 {%r318, %r319, %r320, %r321}, [%r322];

	add.s32 %r327, %r677, 1120;

	ld.shared.v4.b32 {%r323, %r324, %r325, %r326}, [%r327];

	add.s32 %r332, %r678, 1024;

	ld.shared.v4.b32 {%r328, %r329, %r330, %r331}, [%r332];

	add.s32 %r337, %r678, 1152;

	ld.shared.v4.b32 {%r333, %r334, %r335, %r336}, [%r337];

	mov.b32 %f545, %r298;
mov.b32 %f546, %r308;
fma.rn.f32 %f547, %f545, %f546, %f467;
mov.b32 %f548, %r299;
fma.rn.f32 %f549, %f548, %f546, %f469;
mov.b32 %f550, %r309;
fma.rn.f32 %f551, %f548, %f550, %f471;
fma.rn.f32 %f552, %f545, %f550, %f472;
mov.b32 %f553, %r300;
fma.rn.f32 %f554, %f553, %f546, %f474;
mov.b32 %f555, %r301;
fma.rn.f32 %f556, %f555, %f546, %f476;
fma.rn.f32 %f557, %f555, %f550, %f477;
fma.rn.f32 %f558, %f553, %f550, %f478;
mov.b32 %f559, %r303;
fma.rn.f32 %f560, %f559, %f546, %f480;
mov.b32 %f561, %r304;
fma.rn.f32 %f562, %f561, %f546, %f482;
fma.rn.f32 %f563, %f561, %f550, %f483;
fma.rn.f32 %f564, %f559, %f550, %f484;
mov.b32 %f565, %r305;
fma.rn.f32 %f566, %f565, %f546, %f486;
mov.b32 %f567, %r306;
fma.rn.f32 %f568, %f567, %f546, %f488;
fma.rn.f32 %f569, %f567, %f550, %f489;
fma.rn.f32 %f570, %f565, %f550, %f490;
mov.b32 %f571, %r310;
fma.rn.f32 %f572, %f565, %f571, %f492;
fma.rn.f32 %f573, %f567, %f571, %f493;
mov.b32 %f574, %r311;
fma.rn.f32 %f575, %f567, %f574, %f495;
fma.rn.f32 %f576, %f565, %f574, %f496;
fma.rn.f32 %f577, %f559, %f571, %f497;
fma.rn.f32 %f578, %f561, %f571, %f498;
fma.rn.f32 %f579, %f561, %f574, %f499;
fma.rn.f32 %f580, %f559, %f574, %f500;
fma.rn.f32 %f581, %f553, %f571, %f501;
fma.rn.f32 %f582, %f555, %f571, %f502;
fma.rn.f32 %f583, %f555, %f574, %f503;
fma.rn.f32 %f584, %f553, %f574, %f504;
fma.rn.f32 %f585, %f545, %f571, %f505;
fma.rn.f32 %f586, %f548, %f571, %f506;
fma.rn.f32 %f587, %f548, %f574, %f507;
fma.rn.f32 %f588, %f545, %f574, %f508;
mov.b32 %f589, %r313;
fma.rn.f32 %f590, %f545, %f589, %f510;
fma.rn.f32 %f591, %f548, %f589, %f511;
mov.b32 %f592, %r314;
fma.rn.f32 %f593, %f548, %f592, %f513;
fma.rn.f32 %f594, %f545, %f592, %f514;
fma.rn.f32 %f595, %f553, %f589, %f515;
fma.rn.f32 %f596, %f555, %f589, %f516;
fma.rn.f32 %f597, %f555, %f592, %f517;
fma.rn.f32 %f598, %f553, %f592, %f518;
fma.rn.f32 %f599, %f559, %f589, %f519;
fma.rn.f32 %f600, %f561, %f589, %f520;
fma.rn.f32 %f601, %f561, %f592, %f521;
fma.rn.f32 %f602, %f559, %f592, %f522;
fma.rn.f32 %f603, %f565, %f589, %f523;
fma.rn.f32 %f604, %f567, %f589, %f524;
fma.rn.f32 %f605, %f567, %f592, %f525;
fma.rn.f32 %f606, %f565, %f592, %f526;
mov.b32 %f607, %r315;
fma.rn.f32 %f608, %f565, %f607, %f528;
fma.rn.f32 %f609, %f567, %f607, %f529;
mov.b32 %f610, %r316;
fma.rn.f32 %f611, %f567, %f610, %f531;
fma.rn.f32 %f612, %f565, %f610, %f532;
fma.rn.f32 %f613, %f559, %f607, %f533;
fma.rn.f32 %f614, %f561, %f607, %f534;
fma.rn.f32 %f615, %f561, %f610, %f535;
fma.rn.f32 %f616, %f559, %f610, %f536;
fma.rn.f32 %f617, %f553, %f607, %f537;
fma.rn.f32 %f618, %f555, %f607, %f538;
fma.rn.f32 %f619, %f555, %f610, %f539;
fma.rn.f32 %f620, %f553, %f610, %f540;
fma.rn.f32 %f621, %f545, %f607, %f541;
fma.rn.f32 %f622, %f548, %f607, %f542;
fma.rn.f32 %f623, %f548, %f610, %f543;
fma.rn.f32 %f624, %f545, %f610, %f544;
add.s32 %r342, %r677, 1584;

	ld.shared.v4.b32 {%r338, %r339, %r340, %r341}, [%r342];

	add.s32 %r347, %r677, 1648;

	ld.shared.v4.b32 {%r343, %r344, %r345, %r346}, [%r347];

	add.s32 %r352, %r678, 1536;

	ld.shared.v4.b32 {%r348, %r349, %r350, %r351}, [%r352];

	add.s32 %r357, %r678, 1664;

	ld.shared.v4.b32 {%r353, %r354, %r355, %r356}, [%r357];

	mov.b32 %f625, %r318;
mov.b32 %f626, %r328;
fma.rn.f32 %f627, %f625, %f626, %f547;
mov.b32 %f628, %r319;
fma.rn.f32 %f629, %f628, %f626, %f549;
mov.b32 %f630, %r329;
fma.rn.f32 %f631, %f628, %f630, %f551;
fma.rn.f32 %f632, %f625, %f630, %f552;
mov.b32 %f633, %r320;
fma.rn.f32 %f634, %f633, %f626, %f554;
mov.b32 %f635, %r321;
fma.rn.f32 %f636, %f635, %f626, %f556;
fma.rn.f32 %f637, %f635, %f630, %f557;
fma.rn.f32 %f638, %f633, %f630, %f558;
mov.b32 %f639, %r323;
fma.rn.f32 %f640, %f639, %f626, %f560;
mov.b32 %f641, %r324;
fma.rn.f32 %f642, %f641, %f626, %f562;
fma.rn.f32 %f643, %f641, %f630, %f563;
fma.rn.f32 %f644, %f639, %f630, %f564;
mov.b32 %f645, %r325;
fma.rn.f32 %f646, %f645, %f626, %f566;
mov.b32 %f647, %r326;
fma.rn.f32 %f648, %f647, %f626, %f568;
fma.rn.f32 %f649, %f647, %f630, %f569;
fma.rn.f32 %f650, %f645, %f630, %f570;
mov.b32 %f651, %r330;
fma.rn.f32 %f652, %f645, %f651, %f572;
fma.rn.f32 %f653, %f647, %f651, %f573;
mov.b32 %f654, %r331;
fma.rn.f32 %f655, %f647, %f654, %f575;
fma.rn.f32 %f656, %f645, %f654, %f576;
fma.rn.f32 %f657, %f639, %f651, %f577;
fma.rn.f32 %f658, %f641, %f651, %f578;
fma.rn.f32 %f659, %f641, %f654, %f579;
fma.rn.f32 %f660, %f639, %f654, %f580;
fma.rn.f32 %f661, %f633, %f651, %f581;
fma.rn.f32 %f662, %f635, %f651, %f582;
fma.rn.f32 %f663, %f635, %f654, %f583;
fma.rn.f32 %f664, %f633, %f654, %f584;
fma.rn.f32 %f665, %f625, %f651, %f585;
fma.rn.f32 %f666, %f628, %f651, %f586;
fma.rn.f32 %f667, %f628, %f654, %f587;
fma.rn.f32 %f668, %f625, %f654, %f588;
mov.b32 %f669, %r333;
fma.rn.f32 %f670, %f625, %f669, %f590;
fma.rn.f32 %f671, %f628, %f669, %f591;
mov.b32 %f672, %r334;
fma.rn.f32 %f673, %f628, %f672, %f593;
fma.rn.f32 %f674, %f625, %f672, %f594;
fma.rn.f32 %f675, %f633, %f669, %f595;
fma.rn.f32 %f676, %f635, %f669, %f596;
fma.rn.f32 %f677, %f635, %f672, %f597;
fma.rn.f32 %f678, %f633, %f672, %f598;
fma.rn.f32 %f679, %f639, %f669, %f599;
fma.rn.f32 %f680, %f641, %f669, %f600;
fma.rn.f32 %f681, %f641, %f672, %f601;
fma.rn.f32 %f682, %f639, %f672, %f602;
fma.rn.f32 %f683, %f645, %f669, %f603;
fma.rn.f32 %f684, %f647, %f669, %f604;
fma.rn.f32 %f685, %f647, %f672, %f605;
fma.rn.f32 %f686, %f645, %f672, %f606;
mov.b32 %f687, %r335;
fma.rn.f32 %f688, %f645, %f687, %f608;
fma.rn.f32 %f689, %f647, %f687, %f609;
mov.b32 %f690, %r336;
fma.rn.f32 %f691, %f647, %f690, %f611;
fma.rn.f32 %f692, %f645, %f690, %f612;
fma.rn.f32 %f693, %f639, %f687, %f613;
fma.rn.f32 %f694, %f641, %f687, %f614;
fma.rn.f32 %f695, %f641, %f690, %f615;
fma.rn.f32 %f696, %f639, %f690, %f616;
fma.rn.f32 %f697, %f633, %f687, %f617;
fma.rn.f32 %f698, %f635, %f687, %f618;
fma.rn.f32 %f699, %f635, %f690, %f619;
fma.rn.f32 %f700, %f633, %f690, %f620;
fma.rn.f32 %f701, %f625, %f687, %f621;
fma.rn.f32 %f702, %f628, %f687, %f622;
fma.rn.f32 %f703, %f628, %f690, %f623;
fma.rn.f32 %f704, %f625, %f690, %f624;
add.s32 %r362, %r677, 2112;

	ld.shared.v4.b32 {%r358, %r359, %r360, %r361}, [%r362];

	add.s32 %r367, %r677, 2176;

	ld.shared.v4.b32 {%r363, %r364, %r365, %r366}, [%r367];

	add.s32 %r372, %r678, 2048;

	ld.shared.v4.b32 {%r368, %r369, %r370, %r371}, [%r372];

	add.s32 %r377, %r678, 2176;

	ld.shared.v4.b32 {%r373, %r374, %r375, %r376}, [%r377];

	mov.b32 %f705, %r338;
mov.b32 %f706, %r348;
fma.rn.f32 %f707, %f705, %f706, %f627;
mov.b32 %f708, %r339;
fma.rn.f32 %f709, %f708, %f706, %f629;
mov.b32 %f710, %r349;
fma.rn.f32 %f711, %f708, %f710, %f631;
fma.rn.f32 %f712, %f705, %f710, %f632;
mov.b32 %f713, %r340;
fma.rn.f32 %f714, %f713, %f706, %f634;
mov.b32 %f715, %r341;
fma.rn.f32 %f716, %f715, %f706, %f636;
fma.rn.f32 %f717, %f715, %f710, %f637;
fma.rn.f32 %f718, %f713, %f710, %f638;
mov.b32 %f719, %r343;
fma.rn.f32 %f720, %f719, %f706, %f640;
mov.b32 %f721, %r344;
fma.rn.f32 %f722, %f721, %f706, %f642;
fma.rn.f32 %f723, %f721, %f710, %f643;
fma.rn.f32 %f724, %f719, %f710, %f644;
mov.b32 %f725, %r345;
fma.rn.f32 %f726, %f725, %f706, %f646;
mov.b32 %f727, %r346;
fma.rn.f32 %f728, %f727, %f706, %f648;
fma.rn.f32 %f729, %f727, %f710, %f649;
fma.rn.f32 %f730, %f725, %f710, %f650;
mov.b32 %f731, %r350;
fma.rn.f32 %f732, %f725, %f731, %f652;
fma.rn.f32 %f733, %f727, %f731, %f653;
mov.b32 %f734, %r351;
fma.rn.f32 %f735, %f727, %f734, %f655;
fma.rn.f32 %f736, %f725, %f734, %f656;
fma.rn.f32 %f737, %f719, %f731, %f657;
fma.rn.f32 %f738, %f721, %f731, %f658;
fma.rn.f32 %f739, %f721, %f734, %f659;
fma.rn.f32 %f740, %f719, %f734, %f660;
fma.rn.f32 %f741, %f713, %f731, %f661;
fma.rn.f32 %f742, %f715, %f731, %f662;
fma.rn.f32 %f743, %f715, %f734, %f663;
fma.rn.f32 %f744, %f713, %f734, %f664;
fma.rn.f32 %f745, %f705, %f731, %f665;
fma.rn.f32 %f746, %f708, %f731, %f666;
fma.rn.f32 %f747, %f708, %f734, %f667;
fma.rn.f32 %f748, %f705, %f734, %f668;
mov.b32 %f749, %r353;
fma.rn.f32 %f750, %f705, %f749, %f670;
fma.rn.f32 %f751, %f708, %f749, %f671;
mov.b32 %f752, %r354;
fma.rn.f32 %f753, %f708, %f752, %f673;
fma.rn.f32 %f754, %f705, %f752, %f674;
fma.rn.f32 %f755, %f713, %f749, %f675;
fma.rn.f32 %f756, %f715, %f749, %f676;
fma.rn.f32 %f757, %f715, %f752, %f677;
fma.rn.f32 %f758, %f713, %f752, %f678;
fma.rn.f32 %f759, %f719, %f749, %f679;
fma.rn.f32 %f760, %f721, %f749, %f680;
fma.rn.f32 %f761, %f721, %f752, %f681;
fma.rn.f32 %f762, %f719, %f752, %f682;
fma.rn.f32 %f763, %f725, %f749, %f683;
fma.rn.f32 %f764, %f727, %f749, %f684;
fma.rn.f32 %f765, %f727, %f752, %f685;
fma.rn.f32 %f766, %f725, %f752, %f686;
mov.b32 %f767, %r355;
fma.rn.f32 %f768, %f725, %f767, %f688;
fma.rn.f32 %f769, %f727, %f767, %f689;
mov.b32 %f770, %r356;
fma.rn.f32 %f771, %f727, %f770, %f691;
fma.rn.f32 %f772, %f725, %f770, %f692;
fma.rn.f32 %f773, %f719, %f767, %f693;
fma.rn.f32 %f774, %f721, %f767, %f694;
fma.rn.f32 %f775, %f721, %f770, %f695;
fma.rn.f32 %f776, %f719, %f770, %f696;
fma.rn.f32 %f777, %f713, %f767, %f697;
fma.rn.f32 %f778, %f715, %f767, %f698;
fma.rn.f32 %f779, %f715, %f770, %f699;
fma.rn.f32 %f780, %f713, %f770, %f700;
fma.rn.f32 %f781, %f705, %f767, %f701;
fma.rn.f32 %f782, %f708, %f767, %f702;
fma.rn.f32 %f783, %f708, %f770, %f703;
fma.rn.f32 %f784, %f705, %f770, %f704;
add.s32 %r382, %r677, 2640;

	ld.shared.v4.b32 {%r378, %r379, %r380, %r381}, [%r382];

	add.s32 %r387, %r677, 2704;

	ld.shared.v4.b32 {%r383, %r384, %r385, %r386}, [%r387];

	add.s32 %r392, %r678, 2560;

	ld.shared.v4.b32 {%r388, %r389, %r390, %r391}, [%r392];

	add.s32 %r397, %r678, 2688;

	ld.shared.v4.b32 {%r393, %r394, %r395, %r396}, [%r397];

	mov.b32 %f785, %r358;
mov.b32 %f786, %r368;
fma.rn.f32 %f787, %f785, %f786, %f707;
mov.b32 %f788, %r359;
fma.rn.f32 %f789, %f788, %f786, %f709;
mov.b32 %f790, %r369;
fma.rn.f32 %f791, %f788, %f790, %f711;
fma.rn.f32 %f792, %f785, %f790, %f712;
mov.b32 %f793, %r360;
fma.rn.f32 %f794, %f793, %f786, %f714;
mov.b32 %f795, %r361;
fma.rn.f32 %f796, %f795, %f786, %f716;
fma.rn.f32 %f797, %f795, %f790, %f717;
fma.rn.f32 %f798, %f793, %f790, %f718;
mov.b32 %f799, %r363;
fma.rn.f32 %f800, %f799, %f786, %f720;
mov.b32 %f801, %r364;
fma.rn.f32 %f802, %f801, %f786, %f722;
fma.rn.f32 %f803, %f801, %f790, %f723;
fma.rn.f32 %f804, %f799, %f790, %f724;
mov.b32 %f805, %r365;
fma.rn.f32 %f806, %f805, %f786, %f726;
mov.b32 %f807, %r366;
fma.rn.f32 %f808, %f807, %f786, %f728;
fma.rn.f32 %f809, %f807, %f790, %f729;
fma.rn.f32 %f810, %f805, %f790, %f730;
mov.b32 %f811, %r370;
fma.rn.f32 %f812, %f805, %f811, %f732;
fma.rn.f32 %f813, %f807, %f811, %f733;
mov.b32 %f814, %r371;
fma.rn.f32 %f815, %f807, %f814, %f735;
fma.rn.f32 %f816, %f805, %f814, %f736;
fma.rn.f32 %f817, %f799, %f811, %f737;
fma.rn.f32 %f818, %f801, %f811, %f738;
fma.rn.f32 %f819, %f801, %f814, %f739;
fma.rn.f32 %f820, %f799, %f814, %f740;
fma.rn.f32 %f821, %f793, %f811, %f741;
fma.rn.f32 %f822, %f795, %f811, %f742;
fma.rn.f32 %f823, %f795, %f814, %f743;
fma.rn.f32 %f824, %f793, %f814, %f744;
fma.rn.f32 %f825, %f785, %f811, %f745;
fma.rn.f32 %f826, %f788, %f811, %f746;
fma.rn.f32 %f827, %f788, %f814, %f747;
fma.rn.f32 %f828, %f785, %f814, %f748;
mov.b32 %f829, %r373;
fma.rn.f32 %f830, %f785, %f829, %f750;
fma.rn.f32 %f831, %f788, %f829, %f751;
mov.b32 %f832, %r374;
fma.rn.f32 %f833, %f788, %f832, %f753;
fma.rn.f32 %f834, %f785, %f832, %f754;
fma.rn.f32 %f835, %f793, %f829, %f755;
fma.rn.f32 %f836, %f795, %f829, %f756;
fma.rn.f32 %f837, %f795, %f832, %f757;
fma.rn.f32 %f838, %f793, %f832, %f758;
fma.rn.f32 %f839, %f799, %f829, %f759;
fma.rn.f32 %f840, %f801, %f829, %f760;
fma.rn.f32 %f841, %f801, %f832, %f761;
fma.rn.f32 %f842, %f799, %f832, %f762;
fma.rn.f32 %f843, %f805, %f829, %f763;
fma.rn.f32 %f844, %f807, %f829, %f764;
fma.rn.f32 %f845, %f807, %f832, %f765;
fma.rn.f32 %f846, %f805, %f832, %f766;
mov.b32 %f847, %r375;
fma.rn.f32 %f848, %f805, %f847, %f768;
fma.rn.f32 %f849, %f807, %f847, %f769;
mov.b32 %f850, %r376;
fma.rn.f32 %f851, %f807, %f850, %f771;
fma.rn.f32 %f852, %f805, %f850, %f772;
fma.rn.f32 %f853, %f799, %f847, %f773;
fma.rn.f32 %f854, %f801, %f847, %f774;
fma.rn.f32 %f855, %f801, %f850, %f775;
fma.rn.f32 %f856, %f799, %f850, %f776;
fma.rn.f32 %f857, %f793, %f847, %f777;
fma.rn.f32 %f858, %f795, %f847, %f778;
fma.rn.f32 %f859, %f795, %f850, %f779;
fma.rn.f32 %f860, %f793, %f850, %f780;
fma.rn.f32 %f861, %f785, %f847, %f781;
fma.rn.f32 %f862, %f788, %f847, %f782;
fma.rn.f32 %f863, %f788, %f850, %f783;
fma.rn.f32 %f864, %f785, %f850, %f784;
add.s32 %r402, %r677, 3168;

	ld.shared.v4.b32 {%r398, %r399, %r400, %r401}, [%r402];

	add.s32 %r407, %r677, 3232;

	ld.shared.v4.b32 {%r403, %r404, %r405, %r406}, [%r407];

	add.s32 %r412, %r678, 3072;

	ld.shared.v4.b32 {%r408, %r409, %r410, %r411}, [%r412];

	add.s32 %r417, %r678, 3200;

	ld.shared.v4.b32 {%r413, %r414, %r415, %r416}, [%r417];

	mov.b32 %f865, %r378;
mov.b32 %f866, %r388;
fma.rn.f32 %f65, %f865, %f866, %f787;
mov.b32 %f867, %r379;
fma.rn.f32 %f66, %f867, %f866, %f789;
mov.b32 %f868, %r389;
fma.rn.f32 %f67, %f867, %f868, %f791;
fma.rn.f32 %f68, %f865, %f868, %f792;
mov.b32 %f869, %r380;
fma.rn.f32 %f69, %f869, %f866, %f794;
mov.b32 %f870, %r381;
fma.rn.f32 %f70, %f870, %f866, %f796;
fma.rn.f32 %f71, %f870, %f868, %f797;
fma.rn.f32 %f72, %f869, %f868, %f798;
mov.b32 %f871, %r383;
fma.rn.f32 %f73, %f871, %f866, %f800;
mov.b32 %f872, %r384;
fma.rn.f32 %f74, %f872, %f866, %f802;
fma.rn.f32 %f75, %f872, %f868, %f803;
fma.rn.f32 %f76, %f871, %f868, %f804;
mov.b32 %f873, %r385;
fma.rn.f32 %f77, %f873, %f866, %f806;
mov.b32 %f874, %r386;
fma.rn.f32 %f78, %f874, %f866, %f808;
fma.rn.f32 %f79, %f874, %f868, %f809;
fma.rn.f32 %f80, %f873, %f868, %f810;
mov.b32 %f875, %r390;
fma.rn.f32 %f81, %f873, %f875, %f812;
fma.rn.f32 %f82, %f874, %f875, %f813;
mov.b32 %f876, %r391;
fma.rn.f32 %f83, %f874, %f876, %f815;
fma.rn.f32 %f84, %f873, %f876, %f816;
fma.rn.f32 %f85, %f871, %f875, %f817;
fma.rn.f32 %f86, %f872, %f875, %f818;
fma.rn.f32 %f87, %f872, %f876, %f819;
fma.rn.f32 %f88, %f871, %f876, %f820;
fma.rn.f32 %f89, %f869, %f875, %f821;
fma.rn.f32 %f90, %f870, %f875, %f822;
fma.rn.f32 %f91, %f870, %f876, %f823;
fma.rn.f32 %f92, %f869, %f876, %f824;
fma.rn.f32 %f93, %f865, %f875, %f825;
fma.rn.f32 %f94, %f867, %f875, %f826;
fma.rn.f32 %f95, %f867, %f876, %f827;
fma.rn.f32 %f96, %f865, %f876, %f828;
mov.b32 %f877, %r393;
fma.rn.f32 %f97, %f865, %f877, %f830;
fma.rn.f32 %f98, %f867, %f877, %f831;
mov.b32 %f878, %r394;
fma.rn.f32 %f99, %f867, %f878, %f833;
fma.rn.f32 %f100, %f865, %f878, %f834;
fma.rn.f32 %f101, %f869, %f877, %f835;
fma.rn.f32 %f102, %f870, %f877, %f836;
fma.rn.f32 %f103, %f870, %f878, %f837;
fma.rn.f32 %f104, %f869, %f878, %f838;
fma.rn.f32 %f105, %f871, %f877, %f839;
fma.rn.f32 %f106, %f872, %f877, %f840;
fma.rn.f32 %f107, %f872, %f878, %f841;
fma.rn.f32 %f108, %f871, %f878, %f842;
fma.rn.f32 %f109, %f873, %f877, %f843;
fma.rn.f32 %f110, %f874, %f877, %f844;
fma.rn.f32 %f111, %f874, %f878, %f845;
fma.rn.f32 %f112, %f873, %f878, %f846;
mov.b32 %f879, %r395;
fma.rn.f32 %f113, %f873, %f879, %f848;
fma.rn.f32 %f114, %f874, %f879, %f849;
mov.b32 %f880, %r396;
fma.rn.f32 %f115, %f874, %f880, %f851;
fma.rn.f32 %f116, %f873, %f880, %f852;
fma.rn.f32 %f117, %f871, %f879, %f853;
fma.rn.f32 %f118, %f872, %f879, %f854;
fma.rn.f32 %f119, %f872, %f880, %f855;
fma.rn.f32 %f120, %f871, %f880, %f856;
fma.rn.f32 %f121, %f869, %f879, %f857;
fma.rn.f32 %f122, %f870, %f879, %f858;
fma.rn.f32 %f123, %f870, %f880, %f859;
fma.rn.f32 %f124, %f869, %f880, %f860;
fma.rn.f32 %f125, %f865, %f879, %f861;
fma.rn.f32 %f126, %f867, %f879, %f862;
fma.rn.f32 %f127, %f867, %f880, %f863;
fma.rn.f32 %f128, %f865, %f880, %f864;
st.shared.u32 [%r679], %r274;
st.shared.u32 [%r679+128], %r277;
st.shared.u32 [%r679+256], %r280;
st.shared.u32 [%r679+384], %r283;
st.shared.u32 [%r680], %r286;
st.shared.u32 [%r680+1024], %r289;
st.shared.u32 [%r680+2048], %r292;
st.shared.u32 [%r680+3072], %r295;
bar.sync 0;
setp.eq.s32 %p26, %r681, 1;
@%p26 bra $L__BB0_5;
bra.uni $L__BB0_4;

$L__BB0_5:
add.s32 %r684, %r678, 3584;
add.s32 %r685, %r677, 3696;
mov.u32 %r683, -4224;
mov.u32 %r682, -4096;
bra.uni $L__BB0_6;

$L__BB0_4:
add.s32 %r685, %r677, -4752;
add.s32 %r684, %r678, -4608;
mov.u32 %r683, 4224;
mov.u32 %r682, 4096;

$L__BB0_6:
add.s32 %r658, %r658, -1;
mul.lo.s64 %rd149, %rd2, 3;
setp.gt.s32 %p27, %r658, 2;
add.s32 %r680, %r680, %r682;
add.s32 %r679, %r679, %r683;
xor.b32 %r681, %r681, 1;

	ld.shared.v4.b32 {%r669, %r670, %r671, %r672}, [%r685];

	add.s32 %r437, %r685, 64;

	ld.shared.v4.b32 {%r673, %r674, %r675, %r676}, [%r437];

	
	ld.shared.v4.b32 {%r661, %r662, %r663, %r664}, [%r684];

	add.s32 %r447, %r684, 128;

	ld.shared.v4.b32 {%r665, %r666, %r667, %r668}, [%r447];

	add.s32 %r677, %r685, 528;
add.s32 %r678, %r684, 512;
mov.b32 %f881, %r408;
mov.b32 %f882, %r398;
fma.rn.f32 %f1024, %f882, %f881, %f65;
mov.b32 %f883, %r399;
fma.rn.f32 %f1016, %f883, %f881, %f66;
mov.b32 %f884, %r409;
fma.rn.f32 %f1015, %f883, %f884, %f67;
fma.rn.f32 %f1023, %f882, %f884, %f68;
mov.b32 %f885, %r400;
fma.rn.f32 %f1008, %f885, %f881, %f69;
mov.b32 %f886, %r401;
fma.rn.f32 %f1000, %f886, %f881, %f70;
fma.rn.f32 %f999, %f886, %f884, %f71;
fma.rn.f32 %f1007, %f885, %f884, %f72;
mov.b32 %f887, %r403;
fma.rn.f32 %f992, %f887, %f881, %f73;
mov.b32 %f888, %r404;
fma.rn.f32 %f984, %f888, %f881, %f74;
fma.rn.f32 %f983, %f888, %f884, %f75;
fma.rn.f32 %f991, %f887, %f884, %f76;
mov.b32 %f889, %r405;
fma.rn.f32 %f976, %f889, %f881, %f77;
mov.b32 %f890, %r406;
fma.rn.f32 %f968, %f890, %f881, %f78;
fma.rn.f32 %f967, %f890, %f884, %f79;
fma.rn.f32 %f975, %f889, %f884, %f80;
mov.b32 %f891, %r410;
fma.rn.f32 %f974, %f889, %f891, %f81;
fma.rn.f32 %f966, %f890, %f891, %f82;
mov.b32 %f892, %r411;
fma.rn.f32 %f965, %f890, %f892, %f83;
fma.rn.f32 %f973, %f889, %f892, %f84;
fma.rn.f32 %f990, %f887, %f891, %f85;
fma.rn.f32 %f982, %f888, %f891, %f86;
fma.rn.f32 %f981, %f888, %f892, %f87;
fma.rn.f32 %f989, %f887, %f892, %f88;
fma.rn.f32 %f1006, %f885, %f891, %f89;
fma.rn.f32 %f998, %f886, %f891, %f90;
fma.rn.f32 %f997, %f886, %f892, %f91;
fma.rn.f32 %f1005, %f885, %f892, %f92;
fma.rn.f32 %f1022, %f882, %f891, %f93;
fma.rn.f32 %f1014, %f883, %f891, %f94;
fma.rn.f32 %f1013, %f883, %f892, %f95;
fma.rn.f32 %f1021, %f882, %f892, %f96;
mov.b32 %f893, %r413;
fma.rn.f32 %f1020, %f882, %f893, %f97;
fma.rn.f32 %f1012, %f883, %f893, %f98;
mov.b32 %f894, %r414;
fma.rn.f32 %f1011, %f883, %f894, %f99;
fma.rn.f32 %f1019, %f882, %f894, %f100;
fma.rn.f32 %f1004, %f885, %f893, %f101;
fma.rn.f32 %f996, %f886, %f893, %f102;
fma.rn.f32 %f995, %f886, %f894, %f103;
fma.rn.f32 %f1003, %f885, %f894, %f104;
fma.rn.f32 %f988, %f887, %f893, %f105;
fma.rn.f32 %f980, %f888, %f893, %f106;
fma.rn.f32 %f979, %f888, %f894, %f107;
fma.rn.f32 %f987, %f887, %f894, %f108;
fma.rn.f32 %f972, %f889, %f893, %f109;
fma.rn.f32 %f964, %f890, %f893, %f110;
fma.rn.f32 %f963, %f890, %f894, %f111;
fma.rn.f32 %f971, %f889, %f894, %f112;
mov.b32 %f895, %r415;
fma.rn.f32 %f970, %f889, %f895, %f113;
fma.rn.f32 %f962, %f890, %f895, %f114;
mov.b32 %f896, %r416;
fma.rn.f32 %f961, %f890, %f896, %f115;
fma.rn.f32 %f969, %f889, %f896, %f116;
fma.rn.f32 %f986, %f887, %f895, %f117;
fma.rn.f32 %f978, %f888, %f895, %f118;
fma.rn.f32 %f977, %f888, %f896, %f119;
fma.rn.f32 %f985, %f887, %f896, %f120;
fma.rn.f32 %f1002, %f885, %f895, %f121;
fma.rn.f32 %f994, %f886, %f895, %f122;
fma.rn.f32 %f993, %f886, %f896, %f123;
fma.rn.f32 %f1001, %f885, %f896, %f124;
fma.rn.f32 %f1018, %f882, %f895, %f125;
fma.rn.f32 %f1010, %f883, %f895, %f126;
fma.rn.f32 %f1009, %f883, %f896, %f127;
fma.rn.f32 %f1017, %f882, %f896, %f128;
add.s64 %rd63, %rd9, %rd45;
add.s64 %rd153, %rd153, %rd63;
add.s64 %rd65, %rd3, %rd149;
add.s64 %rd152, %rd152, %rd65;
setp.gt.s32 %p28, %r658, 1;
selp.b32 %r659, %r659, 0, %p27;
selp.b32 %r660, %r660, 0, %p27;
@%p28 bra $L__BB0_3;

$L__BB0_7:
mov.u32 %r657, %tid.x;
and.b32 %r656, %r657, 31;
and.b32 %r655, %r657, 1;
shr.u32 %r654, %r656, 4;
bfi.b32 %r653, %r654, %r655, 1, 31;
mov.u32 %r652, %tid.x;
mov.u32 %r651, %tid.x;
mov.u32 %r650, _ZN7cutlass17SharedStorageBaseE;
shr.u32 %r649, %r651, 8;
mov.u32 %r648, %ctaid.z;
mov.u32 %r647, %ctaid.x;
shl.b32 %r646, %r647, 7;
mov.u32 %r645, %ctaid.y;
shl.b32 %r644, %r645, 7;
shr.s32 %r643, %r651, 31;
mov.b64 %rd151, _ZN7cutlass6KernelINS_4gemm6kernel18GemmSplitKParallelINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINS_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1ELb0ENSD_9NoPermuteEEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1ELb0ESJ_EENSL_ISQ_fSE_Li0ESS_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSM_fSE_fSE_NSW_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS15_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2ENS8_6thread14UnaryTransform8IdentityEEES1F_bEENS_8epilogue11threadblock8EpilogueIS7_S16_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1M_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfLb0ESJ_Lb0EEENS1H_4warp20FragmentIteratorSimtISY_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSM_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S14_EENS1R_16TileIteratorSimtISY_S1Y_fSE_S14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi4EEENS1H_6thread7ConvertIfLi1EfLS1B_2EEENSB_ILi0ELi17EEELi1ELi1EEENS4_38GemmSplitKHorizontalThreadblockSwizzleEEEEEvNT_6ParamsE_param_0;
mov.u64 %rd150, %rd151;
ld.param.u64 %rd131, [%rd150+144];
ld.param.u64 %rd132, [%rd150+160];
ld.param.u64 %rd133, [%rd150+168];
shr.u32 %r578, %r643, 27;
add.s32 %r579, %r651, %r578;
and.b32 %r580, %r579, -32;
sub.s32 %r581, %r651, %r580;
shr.u32 %r582, %r643, 26;
add.s32 %r583, %r651, %r582;
shr.s32 %r584, %r583, 6;
shr.u32 %r585, %r579, 31;
shr.s32 %r586, %r579, 5;
add.s32 %r587, %r586, %r585;
and.b32 %r588, %r587, -2;
sub.s32 %r589, %r586, %r588;
shl.b32 %r590, %r584, 5;
shl.b32 %r591, %r589, 2;
add.s32 %r592, %r591, %r590;
add.s32 %r595, %r644, %r592;
add.s32 %r598, %r646, %r581;
setp.lt.s32 %p29, %r598, %r177;
add.s32 %r599, %r598, 32;
setp.lt.s32 %p30, %r599, %r177;
add.s32 %r600, %r598, 64;
setp.lt.s32 %p31, %r600, %r177;
add.s32 %r601, %r598, 96;
setp.lt.s32 %p32, %r601, %r177;
ld.param.u64 %rd134, [%rd150+192];
setp.ne.s64 %p33, %rd134, 0;
and.pred %p34, %p32, %p33;
and.pred %p35, %p31, %p33;
and.pred %p36, %p30, %p33;
and.pred %p37, %p29, %p33;
cvt.s64.s32 %rd135, %r595;
ld.param.u64 %rd136, [%rd150+128];
mul.lo.s64 %rd137, %rd136, %rd135;
mul.wide.s32 %rd138, %r598, 4;
add.s64 %rd139, %rd137, %rd138;
cvt.s64.s32 %rd140, %r648;
ld.param.u64 %rd141, [%rd150+216];
mul.lo.s64 %rd142, %rd140, %rd141;
shl.b64 %rd143, %rd142, 2;
add.s64 %rd144, %rd139, %rd143;
add.s64 %rd66, %rd134, %rd144;
shr.u32 %r604, %r651, 5;
and.b32 %r605, %r604, 3;
bfi.b32 %r606, %r649, %r605, 2, 30;
shl.b32 %r607, %r606, 2;
shr.u32 %r608, %r651, 1;
and.b32 %r609, %r608, 64;
or.b32 %r614, %r607, %r653;
shl.b32 %r615, %r651, 1;
and.b32 %r616, %r615, 28;
or.b32 %r617, %r609, %r616;
mad.lo.s32 %r618, %r614, 145, %r617;
shl.b32 %r619, %r584, 2;
add.s32 %r620, %r619, %r589;
shl.b32 %r621, %r581, 2;
mad.lo.s32 %r622, %r620, 580, %r621;
bar.sync 0;
shl.b32 %r623, %r618, 2;
add.s32 %r625, %r650, %r623;
st.shared.f32 [%r625], %f1024;
st.shared.f32 [%r625+4], %f1023;
st.shared.f32 [%r625+8], %f1022;
st.shared.f32 [%r625+12], %f1021;
st.shared.f32 [%r625+128], %f1020;
st.shared.f32 [%r625+132], %f1019;
st.shared.f32 [%r625+136], %f1018;
st.shared.f32 [%r625+140], %f1017;
bar.sync 0;
add.s32 %r626, %r650, %r622;
ld.shared.u32 %r448, [%r626];
ld.shared.u32 %r450, [%r626+128];
ld.shared.u32 %r452, [%r626+256];
ld.shared.u32 %r454, [%r626+384];
ld.shared.u32 %r456, [%r626+1160];
ld.shared.u32 %r458, [%r626+1288];
ld.shared.u32 %r460, [%r626+1416];
ld.shared.u32 %r462, [%r626+1544];
setp.lt.s32 %p38, %r595, %r176;
and.pred %p39, %p38, %p37;
selp.u32 %r449, 1, 0, %p39;

	{
.reg .pred p;
setp.ne.b32 p, %r449, 0;
@p st.global.u32 [%rd66], %r448;
}


	add.s64 %rd67, %rd66, 128;
and.pred %p40, %p38, %p36;
selp.u32 %r451, 1, 0, %p40;

	{
.reg .pred p;
setp.ne.b32 p, %r451, 0;
@p st.global.u32 [%rd67], %r450;
}


	add.s64 %rd68, %rd66, 256;
and.pred %p41, %p38, %p35;
selp.u32 %r453, 1, 0, %p41;

	{
.reg .pred p;
setp.ne.b32 p, %r453, 0;
@p st.global.u32 [%rd68], %r452;
}


	add.s64 %rd69, %rd66, 384;
and.pred %p42, %p38, %p34;
selp.u32 %r455, 1, 0, %p42;

	{
.reg .pred p;
setp.ne.b32 p, %r455, 0;
@p st.global.u32 [%rd69], %r454;
}


	add.s64 %rd70, %rd66, %rd131;
add.s32 %r627, %r595, 8;
setp.lt.s32 %p43, %r627, %r176;
and.pred %p44, %p43, %p37;
selp.u32 %r457, 1, 0, %p44;

	{
.reg .pred p;
setp.ne.b32 p, %r457, 0;
@p st.global.u32 [%rd70], %r456;
}


	add.s64 %rd71, %rd70, 128;
and.pred %p45, %p43, %p36;
selp.u32 %r459, 1, 0, %p45;

	{
.reg .pred p;
setp.ne.b32 p, %r459, 0;
@p st.global.u32 [%rd71], %r458;
}


	add.s64 %rd72, %rd70, 256;
and.pred %p46, %p43, %p35;
selp.u32 %r461, 1, 0, %p46;

	{
.reg .pred p;
setp.ne.b32 p, %r461, 0;
@p st.global.u32 [%rd72], %r460;
}


	add.s64 %rd73, %rd70, 384;
and.pred %p47, %p43, %p34;
selp.u32 %r463, 1, 0, %p47;

	{
.reg .pred p;
setp.ne.b32 p, %r463, 0;
@p st.global.u32 [%rd73], %r462;
}


	add.s64 %rd74, %rd66, %rd132;
or.b32 %r628, %r595, 1;
bar.sync 0;
st.shared.f32 [%r625], %f1016;
st.shared.f32 [%r625+4], %f1015;
st.shared.f32 [%r625+8], %f1014;
st.shared.f32 [%r625+12], %f1013;
st.shared.f32 [%r625+128], %f1012;
st.shared.f32 [%r625+132], %f1011;
st.shared.f32 [%r625+136], %f1010;
st.shared.f32 [%r625+140], %f1009;
bar.sync 0;
ld.shared.u32 %r464, [%r626];
ld.shared.u32 %r466, [%r626+128];
ld.shared.u32 %r468, [%r626+256];
ld.shared.u32 %r470, [%r626+384];
ld.shared.u32 %r472, [%r626+1160];
ld.shared.u32 %r474, [%r626+1288];
ld.shared.u32 %r476, [%r626+1416];
ld.shared.u32 %r478, [%r626+1544];
setp.lt.s32 %p48, %r628, %r176;
and.pred %p49, %p48, %p37;
selp.u32 %r465, 1, 0, %p49;

	{
.reg .pred p;
setp.ne.b32 p, %r465, 0;
@p st.global.u32 [%rd74], %r464;
}


	and.pred %p50, %p48, %p36;
selp.u32 %r467, 1, 0, %p50;
add.s64 %rd145, %rd132, 128;
add.s64 %rd75, %rd66, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r467, 0;
@p st.global.u32 [%rd75], %r466;
}


	and.pred %p51, %p48, %p35;
selp.u32 %r469, 1, 0, %p51;
add.s64 %rd146, %rd132, 256;
add.s64 %rd76, %rd66, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r469, 0;
@p st.global.u32 [%rd76], %r468;
}


	and.pred %p52, %p48, %p34;
selp.u32 %r471, 1, 0, %p52;
add.s64 %rd147, %rd132, 384;
add.s64 %rd77, %rd66, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r471, 0;
@p st.global.u32 [%rd77], %r470;
}


	add.s32 %r629, %r595, 9;
setp.lt.s32 %p53, %r629, %r176;
and.pred %p54, %p53, %p37;
selp.u32 %r473, 1, 0, %p54;
add.s64 %rd78, %rd70, %rd132;

	{
.reg .pred p;
setp.ne.b32 p, %r473, 0;
@p st.global.u32 [%rd78], %r472;
}


	and.pred %p55, %p53, %p36;
selp.u32 %r475, 1, 0, %p55;
add.s64 %rd79, %rd70, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r475, 0;
@p st.global.u32 [%rd79], %r474;
}


	and.pred %p56, %p53, %p35;
selp.u32 %r477, 1, 0, %p56;
add.s64 %rd80, %rd70, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r477, 0;
@p st.global.u32 [%rd80], %r476;
}


	and.pred %p57, %p53, %p34;
selp.u32 %r479, 1, 0, %p57;
add.s64 %rd81, %rd70, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r479, 0;
@p st.global.u32 [%rd81], %r478;
}


	add.s64 %rd82, %rd74, %rd132;
or.b32 %r630, %r595, 2;
bar.sync 0;
st.shared.f32 [%r625], %f1008;
st.shared.f32 [%r625+4], %f1007;
st.shared.f32 [%r625+8], %f1006;
st.shared.f32 [%r625+12], %f1005;
st.shared.f32 [%r625+128], %f1004;
st.shared.f32 [%r625+132], %f1003;
st.shared.f32 [%r625+136], %f1002;
st.shared.f32 [%r625+140], %f1001;
bar.sync 0;
ld.shared.u32 %r480, [%r626];
ld.shared.u32 %r482, [%r626+128];
ld.shared.u32 %r484, [%r626+256];
ld.shared.u32 %r486, [%r626+384];
ld.shared.u32 %r488, [%r626+1160];
ld.shared.u32 %r490, [%r626+1288];
ld.shared.u32 %r492, [%r626+1416];
ld.shared.u32 %r494, [%r626+1544];
setp.lt.s32 %p58, %r630, %r176;
and.pred %p59, %p58, %p37;
selp.u32 %r481, 1, 0, %p59;

	{
.reg .pred p;
setp.ne.b32 p, %r481, 0;
@p st.global.u32 [%rd82], %r480;
}


	and.pred %p60, %p58, %p36;
selp.u32 %r483, 1, 0, %p60;
add.s64 %rd83, %rd74, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r483, 0;
@p st.global.u32 [%rd83], %r482;
}


	and.pred %p61, %p58, %p35;
selp.u32 %r485, 1, 0, %p61;
add.s64 %rd84, %rd74, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r485, 0;
@p st.global.u32 [%rd84], %r484;
}


	and.pred %p62, %p58, %p34;
selp.u32 %r487, 1, 0, %p62;
add.s64 %rd85, %rd74, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r487, 0;
@p st.global.u32 [%rd85], %r486;
}


	add.s32 %r631, %r595, 10;
setp.lt.s32 %p63, %r631, %r176;
and.pred %p64, %p63, %p37;
selp.u32 %r489, 1, 0, %p64;
add.s64 %rd86, %rd78, %rd132;

	{
.reg .pred p;
setp.ne.b32 p, %r489, 0;
@p st.global.u32 [%rd86], %r488;
}


	and.pred %p65, %p63, %p36;
selp.u32 %r491, 1, 0, %p65;
add.s64 %rd87, %rd78, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r491, 0;
@p st.global.u32 [%rd87], %r490;
}


	and.pred %p66, %p63, %p35;
selp.u32 %r493, 1, 0, %p66;
add.s64 %rd88, %rd78, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r493, 0;
@p st.global.u32 [%rd88], %r492;
}


	and.pred %p67, %p63, %p34;
selp.u32 %r495, 1, 0, %p67;
add.s64 %rd89, %rd78, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r495, 0;
@p st.global.u32 [%rd89], %r494;
}


	add.s64 %rd90, %rd82, %rd132;
or.b32 %r632, %r595, 3;
bar.sync 0;
st.shared.f32 [%r625], %f1000;
st.shared.f32 [%r625+4], %f999;
st.shared.f32 [%r625+8], %f998;
st.shared.f32 [%r625+12], %f997;
st.shared.f32 [%r625+128], %f996;
st.shared.f32 [%r625+132], %f995;
st.shared.f32 [%r625+136], %f994;
st.shared.f32 [%r625+140], %f993;
bar.sync 0;
ld.shared.u32 %r496, [%r626];
ld.shared.u32 %r498, [%r626+128];
ld.shared.u32 %r500, [%r626+256];
ld.shared.u32 %r502, [%r626+384];
ld.shared.u32 %r504, [%r626+1160];
ld.shared.u32 %r506, [%r626+1288];
ld.shared.u32 %r508, [%r626+1416];
ld.shared.u32 %r510, [%r626+1544];
setp.lt.s32 %p68, %r632, %r176;
and.pred %p69, %p68, %p37;
selp.u32 %r497, 1, 0, %p69;

	{
.reg .pred p;
setp.ne.b32 p, %r497, 0;
@p st.global.u32 [%rd90], %r496;
}


	and.pred %p70, %p68, %p36;
selp.u32 %r499, 1, 0, %p70;
add.s64 %rd91, %rd82, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r499, 0;
@p st.global.u32 [%rd91], %r498;
}


	and.pred %p71, %p68, %p35;
selp.u32 %r501, 1, 0, %p71;
add.s64 %rd92, %rd82, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r501, 0;
@p st.global.u32 [%rd92], %r500;
}


	and.pred %p72, %p68, %p34;
selp.u32 %r503, 1, 0, %p72;
add.s64 %rd93, %rd82, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r503, 0;
@p st.global.u32 [%rd93], %r502;
}


	add.s32 %r633, %r595, 11;
setp.lt.s32 %p73, %r633, %r176;
and.pred %p74, %p73, %p37;
selp.u32 %r505, 1, 0, %p74;
add.s64 %rd94, %rd86, %rd132;

	{
.reg .pred p;
setp.ne.b32 p, %r505, 0;
@p st.global.u32 [%rd94], %r504;
}


	and.pred %p75, %p73, %p36;
selp.u32 %r507, 1, 0, %p75;
add.s64 %rd95, %rd86, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r507, 0;
@p st.global.u32 [%rd95], %r506;
}


	and.pred %p76, %p73, %p35;
selp.u32 %r509, 1, 0, %p76;
add.s64 %rd96, %rd86, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r509, 0;
@p st.global.u32 [%rd96], %r508;
}


	and.pred %p77, %p73, %p34;
selp.u32 %r511, 1, 0, %p77;
add.s64 %rd97, %rd86, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r511, 0;
@p st.global.u32 [%rd97], %r510;
}


	add.s64 %rd148, %rd132, %rd133;
add.s64 %rd98, %rd90, %rd148;
add.s32 %r634, %r595, 16;
bar.sync 0;
st.shared.f32 [%r625], %f992;
st.shared.f32 [%r625+4], %f991;
st.shared.f32 [%r625+8], %f990;
st.shared.f32 [%r625+12], %f989;
st.shared.f32 [%r625+128], %f988;
st.shared.f32 [%r625+132], %f987;
st.shared.f32 [%r625+136], %f986;
st.shared.f32 [%r625+140], %f985;
bar.sync 0;
ld.shared.u32 %r512, [%r626];
ld.shared.u32 %r514, [%r626+128];
ld.shared.u32 %r516, [%r626+256];
ld.shared.u32 %r518, [%r626+384];
ld.shared.u32 %r520, [%r626+1160];
ld.shared.u32 %r522, [%r626+1288];
ld.shared.u32 %r524, [%r626+1416];
ld.shared.u32 %r526, [%r626+1544];
setp.lt.s32 %p78, %r634, %r176;
and.pred %p79, %p78, %p37;
selp.u32 %r513, 1, 0, %p79;

	{
.reg .pred p;
setp.ne.b32 p, %r513, 0;
@p st.global.u32 [%rd98], %r512;
}


	add.s64 %rd99, %rd98, 128;
and.pred %p80, %p78, %p36;
selp.u32 %r515, 1, 0, %p80;

	{
.reg .pred p;
setp.ne.b32 p, %r515, 0;
@p st.global.u32 [%rd99], %r514;
}


	add.s64 %rd100, %rd98, 256;
and.pred %p81, %p78, %p35;
selp.u32 %r517, 1, 0, %p81;

	{
.reg .pred p;
setp.ne.b32 p, %r517, 0;
@p st.global.u32 [%rd100], %r516;
}


	add.s64 %rd101, %rd98, 384;
and.pred %p82, %p78, %p34;
selp.u32 %r519, 1, 0, %p82;

	{
.reg .pred p;
setp.ne.b32 p, %r519, 0;
@p st.global.u32 [%rd101], %r518;
}


	add.s64 %rd102, %rd98, %rd131;
add.s32 %r635, %r595, 24;
setp.lt.s32 %p83, %r635, %r176;
and.pred %p84, %p83, %p37;
selp.u32 %r521, 1, 0, %p84;

	{
.reg .pred p;
setp.ne.b32 p, %r521, 0;
@p st.global.u32 [%rd102], %r520;
}


	add.s64 %rd103, %rd102, 128;
and.pred %p85, %p83, %p36;
selp.u32 %r523, 1, 0, %p85;

	{
.reg .pred p;
setp.ne.b32 p, %r523, 0;
@p st.global.u32 [%rd103], %r522;
}


	add.s64 %rd104, %rd102, 256;
and.pred %p86, %p83, %p35;
selp.u32 %r525, 1, 0, %p86;

	{
.reg .pred p;
setp.ne.b32 p, %r525, 0;
@p st.global.u32 [%rd104], %r524;
}


	add.s64 %rd105, %rd102, 384;
and.pred %p87, %p83, %p34;
selp.u32 %r527, 1, 0, %p87;

	{
.reg .pred p;
setp.ne.b32 p, %r527, 0;
@p st.global.u32 [%rd105], %r526;
}


	add.s32 %r636, %r595, 17;
bar.sync 0;
st.shared.f32 [%r625], %f984;
st.shared.f32 [%r625+4], %f983;
st.shared.f32 [%r625+8], %f982;
st.shared.f32 [%r625+12], %f981;
st.shared.f32 [%r625+128], %f980;
st.shared.f32 [%r625+132], %f979;
st.shared.f32 [%r625+136], %f978;
st.shared.f32 [%r625+140], %f977;
bar.sync 0;
ld.shared.u32 %r528, [%r626];
ld.shared.u32 %r530, [%r626+128];
ld.shared.u32 %r532, [%r626+256];
ld.shared.u32 %r534, [%r626+384];
ld.shared.u32 %r536, [%r626+1160];
ld.shared.u32 %r538, [%r626+1288];
ld.shared.u32 %r540, [%r626+1416];
ld.shared.u32 %r542, [%r626+1544];
setp.lt.s32 %p88, %r636, %r176;
and.pred %p89, %p88, %p37;
selp.u32 %r529, 1, 0, %p89;
add.s64 %rd106, %rd98, %rd132;

	{
.reg .pred p;
setp.ne.b32 p, %r529, 0;
@p st.global.u32 [%rd106], %r528;
}


	and.pred %p90, %p88, %p36;
selp.u32 %r531, 1, 0, %p90;
add.s64 %rd107, %rd98, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r531, 0;
@p st.global.u32 [%rd107], %r530;
}


	and.pred %p91, %p88, %p35;
selp.u32 %r533, 1, 0, %p91;
add.s64 %rd108, %rd98, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r533, 0;
@p st.global.u32 [%rd108], %r532;
}


	and.pred %p92, %p88, %p34;
selp.u32 %r535, 1, 0, %p92;
add.s64 %rd109, %rd98, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r535, 0;
@p st.global.u32 [%rd109], %r534;
}


	add.s32 %r637, %r595, 25;
setp.lt.s32 %p93, %r637, %r176;
and.pred %p94, %p93, %p37;
selp.u32 %r537, 1, 0, %p94;
add.s64 %rd110, %rd102, %rd132;

	{
.reg .pred p;
setp.ne.b32 p, %r537, 0;
@p st.global.u32 [%rd110], %r536;
}


	and.pred %p95, %p93, %p36;
selp.u32 %r539, 1, 0, %p95;
add.s64 %rd111, %rd102, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r539, 0;
@p st.global.u32 [%rd111], %r538;
}


	and.pred %p96, %p93, %p35;
selp.u32 %r541, 1, 0, %p96;
add.s64 %rd112, %rd102, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r541, 0;
@p st.global.u32 [%rd112], %r540;
}


	and.pred %p97, %p93, %p34;
selp.u32 %r543, 1, 0, %p97;
add.s64 %rd113, %rd102, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r543, 0;
@p st.global.u32 [%rd113], %r542;
}


	add.s32 %r638, %r595, 18;
bar.sync 0;
st.shared.f32 [%r625], %f976;
st.shared.f32 [%r625+4], %f975;
st.shared.f32 [%r625+8], %f974;
st.shared.f32 [%r625+12], %f973;
st.shared.f32 [%r625+128], %f972;
st.shared.f32 [%r625+132], %f971;
st.shared.f32 [%r625+136], %f970;
st.shared.f32 [%r625+140], %f969;
bar.sync 0;
ld.shared.u32 %r544, [%r626];
ld.shared.u32 %r546, [%r626+128];
ld.shared.u32 %r548, [%r626+256];
ld.shared.u32 %r550, [%r626+384];
ld.shared.u32 %r552, [%r626+1160];
ld.shared.u32 %r554, [%r626+1288];
ld.shared.u32 %r556, [%r626+1416];
ld.shared.u32 %r558, [%r626+1544];
setp.lt.s32 %p98, %r638, %r176;
and.pred %p99, %p98, %p37;
selp.u32 %r545, 1, 0, %p99;
add.s64 %rd114, %rd106, %rd132;

	{
.reg .pred p;
setp.ne.b32 p, %r545, 0;
@p st.global.u32 [%rd114], %r544;
}


	and.pred %p100, %p98, %p36;
selp.u32 %r547, 1, 0, %p100;
add.s64 %rd115, %rd106, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r547, 0;
@p st.global.u32 [%rd115], %r546;
}


	and.pred %p101, %p98, %p35;
selp.u32 %r549, 1, 0, %p101;
add.s64 %rd116, %rd106, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r549, 0;
@p st.global.u32 [%rd116], %r548;
}


	and.pred %p102, %p98, %p34;
selp.u32 %r551, 1, 0, %p102;
add.s64 %rd117, %rd106, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r551, 0;
@p st.global.u32 [%rd117], %r550;
}


	add.s32 %r639, %r595, 26;
setp.lt.s32 %p103, %r639, %r176;
and.pred %p104, %p103, %p37;
selp.u32 %r553, 1, 0, %p104;
add.s64 %rd118, %rd110, %rd132;

	{
.reg .pred p;
setp.ne.b32 p, %r553, 0;
@p st.global.u32 [%rd118], %r552;
}


	and.pred %p105, %p103, %p36;
selp.u32 %r555, 1, 0, %p105;
add.s64 %rd119, %rd110, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r555, 0;
@p st.global.u32 [%rd119], %r554;
}


	and.pred %p106, %p103, %p35;
selp.u32 %r557, 1, 0, %p106;
add.s64 %rd120, %rd110, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r557, 0;
@p st.global.u32 [%rd120], %r556;
}


	and.pred %p107, %p103, %p34;
selp.u32 %r559, 1, 0, %p107;
add.s64 %rd121, %rd110, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r559, 0;
@p st.global.u32 [%rd121], %r558;
}


	add.s32 %r640, %r595, 19;
bar.sync 0;
st.shared.f32 [%r625], %f968;
st.shared.f32 [%r625+4], %f967;
st.shared.f32 [%r625+8], %f966;
st.shared.f32 [%r625+12], %f965;
st.shared.f32 [%r625+128], %f964;
st.shared.f32 [%r625+132], %f963;
st.shared.f32 [%r625+136], %f962;
st.shared.f32 [%r625+140], %f961;
bar.sync 0;
ld.shared.u32 %r560, [%r626];
ld.shared.u32 %r562, [%r626+128];
ld.shared.u32 %r564, [%r626+256];
ld.shared.u32 %r566, [%r626+384];
ld.shared.u32 %r568, [%r626+1160];
ld.shared.u32 %r570, [%r626+1288];
ld.shared.u32 %r572, [%r626+1416];
ld.shared.u32 %r574, [%r626+1544];
setp.lt.s32 %p108, %r640, %r176;
and.pred %p109, %p108, %p37;
selp.u32 %r561, 1, 0, %p109;
add.s64 %rd122, %rd114, %rd132;

	{
.reg .pred p;
setp.ne.b32 p, %r561, 0;
@p st.global.u32 [%rd122], %r560;
}


	and.pred %p110, %p108, %p36;
selp.u32 %r563, 1, 0, %p110;
add.s64 %rd123, %rd114, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r563, 0;
@p st.global.u32 [%rd123], %r562;
}


	and.pred %p111, %p108, %p35;
selp.u32 %r565, 1, 0, %p111;
add.s64 %rd124, %rd114, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r565, 0;
@p st.global.u32 [%rd124], %r564;
}


	and.pred %p112, %p108, %p34;
selp.u32 %r567, 1, 0, %p112;
add.s64 %rd125, %rd114, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r567, 0;
@p st.global.u32 [%rd125], %r566;
}


	add.s32 %r641, %r595, 27;
setp.lt.s32 %p113, %r641, %r176;
and.pred %p114, %p113, %p37;
selp.u32 %r569, 1, 0, %p114;
add.s64 %rd126, %rd118, %rd132;

	{
.reg .pred p;
setp.ne.b32 p, %r569, 0;
@p st.global.u32 [%rd126], %r568;
}


	and.pred %p115, %p113, %p36;
selp.u32 %r571, 1, 0, %p115;
add.s64 %rd127, %rd118, %rd145;

	{
.reg .pred p;
setp.ne.b32 p, %r571, 0;
@p st.global.u32 [%rd127], %r570;
}


	and.pred %p116, %p113, %p35;
selp.u32 %r573, 1, 0, %p116;
add.s64 %rd128, %rd118, %rd146;

	{
.reg .pred p;
setp.ne.b32 p, %r573, 0;
@p st.global.u32 [%rd128], %r572;
}


	and.pred %p117, %p113, %p34;
selp.u32 %r575, 1, 0, %p117;
add.s64 %rd129, %rd118, %rd147;

	{
.reg .pred p;
setp.ne.b32 p, %r575, 0;
@p st.global.u32 [%rd129], %r574;
}



$L__BB0_8:
ret;
"""

branch_counter = 0

class BasicBlock:

    def __init__(self, name) -> None:
        self.name = name
        self.insts = []
        self.branch_targets = None
        self._has_sync = False

    def add_inst(self, line):
        self.insts.append(line)
        if "bar.sync" in line:
            self._has_sync = True

    def is_end_block(self):
        return True if self.branch_targets else False

    def has_sync(self):
        return self._has_sync

    def __str__(self):
        _str = f"BasicBlock {self.name}:\n"
        _str += f"is_sync_block: {self.has_sync()}\n"
        _str += "Code:\n"

        code = f"\t${self.name}:\n"
        for inst in self.insts:
            code += f"\t{inst}\n"

        _str += code

        if self.branch_targets:
            branch_targets = "Branch Targets: \n"
            for branch_target in self.branch_targets:
                branch_targets += f"\t{branch_target.name}\n"

            _str += branch_targets
        
        return _str.strip()


def contains_path_btw_nodes(start_node, end_node):

    visited = set()

    def dfs(start, end) -> bool:

        if start == end:
            return True
        
        visited.add(start)

        if start.branch_targets:
            for branch in start.branch_targets:
                if branch not in visited:
                    if dfs(branch, end):
                        return True
    
        return False

    return dfs(start_node, end_node)


def contains_path_to_dest(start_node, exclude_nodes):
    
    visited = set()

    def dfs(start) -> bool:

        if not start.branch_targets:
            return True
        
        visited.add(start)

        for branch in start.branch_targets:
            if branch not in visited and branch not in exclude_nodes:
                if dfs(branch):
                    return True
        return False

    return dfs(start_node)


def check_cfg_safe(basic_blocks):

    sync_blocks = [block for block in basic_blocks if block.has_sync()]

    if not sync_blocks:
        return True

    root_block = basic_blocks[0]
    is_last = [True for _ in sync_blocks]

    # check whether each sync block is guaranteed to be the last sync block if it is in the execution path
    for idx, sync_block in enumerate(sync_blocks):

        for other_sync_block in sync_blocks:

            if other_sync_block == sync_block:
                continue
        
            if contains_path_btw_nodes(other_sync_block, sync_block):
                is_last[idx] = False
                break
    
    if contains_path_to_dest(root_block, sync_blocks):
        return False

    for idx in range(len(sync_blocks)):

        if is_last[idx]:
            continue

        if contains_path_to_dest(sync_blocks[idx], sync_blocks):
            return False
    
    return True


def get_block_idx(basic_blocks, target_name):
    for idx, basic_block in enumerate(basic_blocks):
        if basic_block.name == target_name:
            return idx
    
    return -1


def resolve_blocks_dependencies(basic_blocks):

    for idx, basic_block in enumerate(basic_blocks):

        last_line = basic_block.insts[-1]
        if is_cond_branch_stmt(last_line):
            _, target_bb = parse_cond_branch_stmt(last_line)
            basic_block.branch_targets = [
                basic_blocks[idx + 1],
                basic_blocks[get_block_idx(basic_blocks, target_bb)],
            ]
        elif is_uncond_branch_stmt(last_line):
            target_bb = parse_uncond_branch_stmt(last_line)
            basic_block.branch_targets = [
                basic_blocks[get_block_idx(basic_blocks, target_bb)],
            ]
        elif last_line == "ret;":
            pass
        else:
            basic_block.branch_targets = [
                basic_blocks[idx + 1],
            ]


def is_named_basic_block(line):
    pattern = r"\$([A-Za-z0-9_]*):"
    match = re.search(pattern, line)
    return True if match else False


def is_cond_branch_stmt(line):
    pattern = pattern = r"@%p(\d+) bra \$([A-Za-z0-9_]+);"
    match = re.search(pattern, line)
    return True if match else False


def is_uncond_branch_stmt(line):
    pattern = pattern = r"bra.uni \$([A-Za-z0-9_]+);"
    match = re.search(pattern, line)
    return True if match else False


def parse_named_basic_block(line):
    pattern = r"\$([A-Za-z0-9_]*):"
    match = re.search(pattern, line)
    bb_name = match.group(1)
    return bb_name
    

def parse_cond_branch_stmt(line):
    pattern = pattern = r"@%p(\d+) bra \$([A-Za-z0-9_]+);"
    match = re.search(pattern, line)
    pred_reg, target_bb = match.group(1), match.group(2)
    return pred_reg, target_bb


def parse_uncond_branch_stmt(line):
    pattern = pattern = r"bra.uni \$([A-Za-z0-9_]+);"
    match = re.search(pattern, line)
    target_bb = match.group(1)
    return target_bb


def construct_basic_blocks(ptx_body):
    basic_blocks = []
    curr_block = None

    for line in ptx_body.split("\n"):

        line = line.strip()

        if not line:
            continue

        # if the line is a named basic block
        line_is_named_bb = is_named_basic_block(line)

        # handle the first basic block if not named
        if not curr_block and not line_is_named_bb:
            curr_block = BasicBlock("L__ENTRY_BLOCK")

        if not line_is_named_bb:
            curr_block.add_inst(line)

        # test if the line is a branch
        line_is_branch = is_cond_branch_stmt(line) or is_uncond_branch_stmt(line)

        if line_is_named_bb or line_is_branch:

            global branch_counter

            if line_is_named_bb:
                bb_name = parse_named_basic_block(line)
            else:
                bb_name = f"L__BB_BRANCH_{branch_counter}"
                branch_counter += 1

            if curr_block.insts:
                basic_blocks.append(curr_block)
            
            curr_block = BasicBlock(bb_name)

    # last block
    basic_blocks.append(curr_block)
    resolve_blocks_dependencies(basic_blocks)

    for basic_block in basic_blocks:
        print(basic_block)
        print()

    print(f"check_cfg_safe(basic_blocks): {check_cfg_safe(basic_blocks)}")

construct_basic_blocks(ptx_body)