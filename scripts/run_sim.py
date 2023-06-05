
import sys

sys.path.append("../python")

from preload.simulate.sim_api import get_nsys_trace, get_preload_trace, run_single_job_simulation, run_two_job_timesharing_simulation

def main():
    bert_preload_trace = get_preload_trace("bert", "preload-trace/bert_8_with_mp_cpu.txt", "preload-trace/bert_8_with_mp_gpu.txt")
    run_single_job_simulation(mobilenet_preload_trace)

if __name__ == "__main__":
    main()
    