from trace_util import NsysTrace, PreloadTrace
from sim import SingleJobSimulator, TwoJobTimeSharingSimulator

if __name__ == "__main__":

    # mobilenet_trace_file = "trace/mobilenet-64-2000.json"
    # mobilenet_trace = Trace("mobilenet", mobilenet_trace_file, 104156, 24203333)

    # vgg_trace_file = "trace/vgg-32-8000.json"
    # vgg_trace = Trace("vgg", vgg_trace_file, 3970, 21399607)

    # # simulator = SingleJobSimulator()
    # # time = simulator.simulate(vgg_trace)
    # # print(f"Single job simulation time: {time / (10 ** 9)}s")

    # simulator = TwoJobTimeSharingSimulator()
    # t1_time, t2_time = simulator.simulate(mobilenet_trace, vgg_trace)
    # print(f"Time sharing simulation time: {t1_time / (10 ** 9)}s and {t2_time / (10 ** 9)}s")

    mobilenet_trace_file_cpu = "preload-trace/mobilenet-trace-cpu.txt"
    mobilenet_trace_file_gpu = "preload-trace/mobilenet-trace-gpu.txt"
    mobilenet_trace = PreloadTrace("mobilenet", mobilenet_trace_file_cpu, mobilenet_trace_file_gpu)

    simulator = SingleJobSimulator()
    time = simulator.simulate(mobilenet_trace)
    print(f"Single job simulation time: {time / (10 ** 9)}s")