
import sys

sys.path.append("./python")

from preload.simulate.sim_api import (
    get_nsys_trace,
    get_preload_trace,
    run_single_job_simulation,
    run_two_job_timesharing_simulation,
    run_two_job_timesharing_fair_simulation
)

def main():
    bert_trace = get_preload_trace("bert_8_with_mp", "preload-trace/bert_8_with_mp_cpu.txt", "preload-trace/bert_8_with_mp_gpu.txt")
    # run_single_job_simulation(bert_trace)

    neumf_trace = get_preload_trace("NeuMF-pre_128_no_mp", "preload-trace/NeuMF-pre_128_no_mp_cpu.txt", "preload-trace/NeuMF-pre_128_no_mp_gpu.txt")
    # run_single_job_simulation(mobilenet_trace)

    run_two_job_timesharing_fair_simulation(bert_trace, neumf_trace, sim_time=90)

    # pointnet_trace = get_preload_trace("pointnet_128_no_mp", "preload-trace/pointnet_128_no_mp_cpu.txt", "preload-trace/pointnet_128_no_mp_cpu.txt")
    # run_single_job_simulation(pointnet_trace)

    # efficientnet_trace = get_preload_trace("EfficientNetB0_64_with_mp", "preload-trace/EfficientNetB0_64_with_mp_cpu.txt", "preload-trace/EfficientNetB0_64_with_mp_cpu.txt")
    # run_single_job_simulation(efficientnet_trace)

    # run_two_job_timesharing_fair_simulation(pointnet_trace, efficientnet_trace, sim_time=90)

if __name__ == "__main__":
    main()
    