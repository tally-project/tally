import sys

sys.path.append("../python")

from tally.preload.client_preload import gen_client_code
from tally.preload.profile_preload import gen_profile_preload


gen_client_code()
gen_profile_preload(profile_kernel=False, output_file="tally_profile_cpu.cpp")
gen_profile_preload(profile_kernel=True, output_file="tally_profile_gpu.cpp")
