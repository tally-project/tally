import sys

sys.path.append("../python")

from preload.gen_preload import *

generate_preload(
    header_files=["def.h"],
    profile_kernel=False,
    output_file="preload.cpp",
    print_trace=False
)