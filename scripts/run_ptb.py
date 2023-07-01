import sys
import argparse

sys.path.append("../python")

from tally.transform.ptb import ptb_kernel

parser = argparse.ArgumentParser(prog="kernel ptb", description="Generate ptb version of kernel")

parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--output-file", type=str, required=False)

args = parser.parse_args()

ptb_kernel(
    args.input_file,
    args.output_file,
)
