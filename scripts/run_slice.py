import sys
import argparse

sys.path.append("../python")

from tally.slice.slice import slice_kernel, get_kernel_names

parser = argparse.ArgumentParser(prog="kernel slicer", description="Generate sliced version of kernel")

parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--output-file", type=str, required=False)
parser.add_argument("--get-names", action="store_true", default=False)

args = parser.parse_args()

if not args.get_names:
    slice_kernel(
        args.input_file,
        args.output_file,
    )
else:
    get_kernel_names(args.input_file)
