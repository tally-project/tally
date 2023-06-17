import sys

sys.path.append("../python")

from tally.preload.gen_preload import *

generate_preload(
    profile_kernel=True,
    output_file="preload_gpu.cpp",
    print_trace=True
)