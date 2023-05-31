import pickle
import os

from trace_util import NsysTrace, PreloadTrace
from sim import SingleJobSimulator, TwoJobTimeSharingSimulator

def get_nsys_trace(trace_name, trace_file_path, start_id, end_id):
    cache_file_name = f"{trace_name}.pickle"
    if os.path.exists(cache_file_name):
        with open(cache_file_name, 'rb') as handle:
            trace = pickle.load(handle)
    else:
        trace = NsysTrace(trace_name, trace_file_path, start_id, end_id)
        with open(cache_file_name, 'wb') as handle:
            pickle.dump(trace, handle)
    return trace

def get_preload_trace(trace_name, cpu_trace_file_path, gpu_trace_file_path):
    cache_file_name = f"{trace_name}.pickle"
    if os.path.exists(cache_file_name):
        with open(cache_file_name, 'rb') as handle:
            trace = pickle.load(handle)
    else:
        trace = PreloadTrace(trace_name, cpu_trace_file_path, gpu_trace_file_path)
        with open(cache_file_name, 'wb') as handle:
            pickle.dump(trace, handle)
    return trace

def run_single_job_simulation(trace):
    simulator = SingleJobSimulator()
    time = simulator.simulate(trace)
    print(f"{trace.model_name}: single job simulation time: {time / (10 ** 9)}s")

def run_two_job_timesharing_simulation(trace1, trace2):
    simulator = TwoJobTimeSharingSimulator()
    t1_time, t2_time = simulator.simulate(trace1, trace2)
    print(f"{trace1.model_name} and {trace2.model_name}: Time sharing simulation time: {t1_time / (10 ** 9)}s and {t2_time / (10 ** 9)}s")

if __name__ == "__main__":

    mobilenet_nsys_trace = get_nsys_trace("mobilenet_nsys", "trace/mobilenet-64-2000.json", 104156, 24203333)
    run_single_job_simulation(mobilenet_nsys_trace)

    vgg_nsys_trace = get_nsys_trace("vgg_nsys", "trace/vgg-32-8000.json", 3970, 21399607)
    run_single_job_simulation(vgg_nsys_trace)

    run_two_job_timesharing_simulation(mobilenet_nsys_trace, vgg_nsys_trace)
    
    mobilenet_preload_trace = get_preload_trace("mobilenet_preload", "preload-trace/mobilenet-trace-cpu.txt", "preload-trace/mobilenet-trace-gpu.txt")
    run_single_job_simulation(mobilenet_preload_trace)

    vgg_preload_trace = get_preload_trace("vgg_preload", "preload-trace/vgg-trace-cpu.txt", "preload-trace/vgg-trace-gpu.txt")
    run_single_job_simulation(vgg_preload_trace)

    run_two_job_timesharing_simulation(mobilenet_preload_trace, vgg_preload_trace)