
import sys

sys.path.append("preload")

from preload.simulate.sim_api import get_nsys_trace, get_preload_trace, run_single_job_simulation, run_two_job_timesharing_simulation

def main():
    mobilenet_nsys_trace = get_nsys_trace("mobilenet_nsys", "trace/mobilenet-64-2000.json", 104156, 24203333)
    run_single_job_simulation(mobilenet_nsys_trace, sim_time=10)

    vgg_nsys_trace = get_nsys_trace("vgg_nsys", "trace/vgg-32-8000.json", 3970, 21399607)
    run_single_job_simulation(vgg_nsys_trace, sim_time=10)

    run_two_job_timesharing_simulation(mobilenet_nsys_trace, vgg_nsys_trace)
    
    mobilenet_preload_trace = get_preload_trace("mobilenet_preload", "preload-trace/mobilenet-trace-cpu.txt", "preload-trace/mobilenet-trace-gpu.txt")
    run_single_job_simulation(mobilenet_preload_trace, sim_time=10)

    vgg_preload_trace = get_preload_trace("vgg_preload", "preload-trace/vgg-trace-cpu.txt", "preload-trace/vgg-trace-gpu.txt")
    run_single_job_simulation(vgg_preload_trace, sim_time=10)

    run_two_job_timesharing_simulation(mobilenet_preload_trace, vgg_preload_trace, sim_time=10)


if __name__ == "__main__":
    main()
    