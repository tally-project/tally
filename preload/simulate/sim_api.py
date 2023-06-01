import pickle
import os

from preload.simulate.trace_util import NsysTrace, PreloadTrace
from preload.simulate.sim import SingleJobSimulator, TwoJobTimeSharingSimulator

def get_nsys_trace(trace_name, trace_file_path, start_id, end_id):
    os.makedirs("trace-cache", exist_ok=True)
    cache_file_name = f"trace-cache/{trace_name}.pickle"
    if os.path.exists(cache_file_name):
        with open(cache_file_name, 'rb') as handle:
            trace = pickle.load(handle)
    else:
        trace = NsysTrace(trace_name, trace_file_path, start_id, end_id)
        with open(cache_file_name, 'wb') as handle:
            pickle.dump(trace, handle)
    return trace

def get_preload_trace(trace_name, cpu_trace_file_path, gpu_trace_file_path):
    os.makedirs("trace-cache", exist_ok=True)
    cache_file_name = f"trace-cache/{trace_name}.pickle"
    if os.path.exists(cache_file_name):
        with open(cache_file_name, 'rb') as handle:
            trace = pickle.load(handle)
    else:
        trace = PreloadTrace(trace_name, cpu_trace_file_path, gpu_trace_file_path)
        with open(cache_file_name, 'wb') as handle:
            pickle.dump(trace, handle)
    return trace

def run_single_job_simulation(trace, sim_time=None):
    simulator = SingleJobSimulator()
    time, iters, gr_active = simulator.simulate(trace, sim_time)
    print(f"Single job simulation: model: {trace.model_name} time: {time / (10 ** 9)}s iters: {iters} ")
    return time, iters, gr_active

def run_two_job_timesharing_simulation(trace1, trace2):
    simulator = TwoJobTimeSharingSimulator()
    t1_time, t1_iters, t2_time, t2_iters, gr_active = simulator.simulate(trace1, trace2)
    print(f"Time sharing simulation: model1 {trace1.model_name} time: {t1_time / (10 ** 9)}s iters: {t1_iters} model2 {trace2.model_name} time: {t2_time / (10 ** 9)}s iters: {t2_iters}")
    return t1_time, t1_iters, t2_time, t2_iters, gr_active