from preload.simulate.trace_util import Trace, check_time_share

class Simulator:

    # kernel switch overhead (switch from one kernel to next)
    kernel_swtch_overhead = 600

    # minimum time to wait for kernel to start after CUDA API call
    # this is hidden is there are unfinished kernels
    kernel_launch_overhead = 1000

    # minimum time for a synchronization call to return
    min_sync_call_t = 4000
    
    def __init__(self, logging=False):
        self.logging = logging
        self.trace_throughput_monitor = {}
        self.sample_frequency = 5 * 10 ** 9
        self.last_ckpt_time = 0

        if logging:
            print(f"logging frequency is every {self.sample_frequency / (10 ** 9)} seconds")

    def compute_gr_actice(self, _kernels):
        gr_active_time = 0

        for _idx, kernel in enumerate(_kernels):
            gr_active_time += (kernel.end_t - kernel.start_t)
            
        last_kernel_end_t = _kernels[-1]

        gr_active_rate = gr_active_time / last_kernel_end_t.end_t
        return gr_active_rate

    def dequeue_finished_kernels(self, kernel_queue, curr_t):

        dequeued_list = []

        # Dequeue kernels that have finished
        while True:
            if not kernel_queue:
                break
            first_kernel = kernel_queue[0]
            if first_kernel.end_t <= curr_t:
                kernel_queue.pop(0)
                dequeued_list.append(first_kernel)

                assert(first_kernel.trace)

                if first_kernel.trace not in self.trace_throughput_monitor:
                    self.trace_throughput_monitor[first_kernel.trace] = {}
                
                if not self.trace_throughput_monitor[first_kernel.trace]:
                    self.trace_throughput_monitor[first_kernel.trace]["total_iters"] = 0
                    self.trace_throughput_monitor[first_kernel.trace]["interval_iters"] = 0
                
                if first_kernel.iter_head:
                    self.trace_throughput_monitor[first_kernel.trace]["total_iters"] += 1
                    self.trace_throughput_monitor[first_kernel.trace]["interval_iters"] += 1
            
            else:
                break
        
        return dequeued_list

        if self.logging:
            if (curr_t // self.sample_frequency) > (self.last_ckpt_time // self.sample_frequency):
                
                time_since_ckpt = (curr_t - self.last_ckpt_time) / (10 ** 9)

                for trace in self.trace_throughput_monitor:
                    interval_iters = self.trace_throughput_monitor[trace]["interval_iters"]
                    print(f"{trace.model_name}: {interval_iters / time_since_ckpt} iters/s")
                    self.trace_throughput_monitor[trace]["interval_iters"] = 0
                
                self.last_ckpt_time = curr_t

    
    def simulate(*arg):
        raise NotImplementedError()


class SingleJobSimulator(Simulator):

    def __init__(self):
        super().__init__()

    def simulate(self, trace: Trace, sim_time=None):
        
        curr_t = 0
        kernel_queue = []
        finish_time = -1
        iters_finished = 0
        finished = False
        finished_kernels = []

        if sim_time:
            # convert to nanoseconds
            sim_time_ns = sim_time * (10 ** 9)

        for call in trace:

            if call.dur_from_last < 0:
                print(call)

            assert(call.dur_from_last >= 0)

            # Advance from last call
            curr_t += call.dur_from_last
            call.start_t = curr_t

            # Check finished kernels
            dequeued_kernels = self.dequeue_finished_kernels(kernel_queue, curr_t)

            curr_kernel = call.kernel

            # Synchronization call
            if "Synchronize" in curr_kernel.name:

                # Wait until all kernels to finish
                if kernel_queue:
                    last_kernel = kernel_queue[-1]
                    # kernel_queue.clear()

                    curr_t = max(last_kernel.end_t, curr_t + Simulator.min_sync_call_t)
                else:
                    curr_t += Simulator.min_sync_call_t

            # launch a kernel
            else:

                # The soonest kernel can start is after this call finishes + some overhead
                min_kernel_start_t = curr_t + call.dur + Simulator.kernel_launch_overhead
                
                if kernel_queue:
                    last_kernel = kernel_queue[-1]
                    curr_kernel.start_t = max(last_kernel.end_t + Simulator.kernel_swtch_overhead, min_kernel_start_t)
                else:
                    curr_kernel.start_t = min_kernel_start_t
                
                curr_kernel.end_t = curr_kernel.start_t + curr_kernel.dur
                
                kernel_queue.append(curr_kernel)

                curr_t += call.dur
            
            call.end_t = curr_t

            if sim_time is not None and not finished and curr_t > sim_time_ns:
                iters_finished = self.trace_throughput_monitor[trace]["total_iters"]
                finished = True
                finish_time = curr_t
                finished_kernels.extend(dequeued_kernels)
        
        max_end_t = trace.run_check()

        if sim_time is None:
            for call in trace:
                if "Synchronize" not in call.kernel.name:
                    finished_kernels.append(call.kernel)

        gr_active = self.compute_gr_actice(finished_kernels)
        print(f"GR Activce: {gr_active}")

        if sim_time is None:
            if kernel_queue:
                finish_time = kernel_queue[-1].end_t
            else:
                finish_time = curr_t
            iters_finished = trace.get_iterations()

        return finish_time, iters_finished, gr_active
    

class TwoJobTimeSharingSimulator(Simulator):
    
    def __init__(self):
        super().__init__()

    def get_last_kernel_of_trace(self, kernel_queue, trace):

        for kernel in reversed(kernel_queue):
            if kernel.trace == trace:
                return kernel
        
        return None

    def simulate(self, t1: Trace, t2: Trace, sim_time=None):

        traces = [t1, t2]
        finish_time = [-1, -1]
        iters_finished = [0, 0]
        finished = [False, False]
        finished_kernels = []
        if sim_time:
            # convert to nanoseconds
            sim_time_ns = sim_time * (10 ** 9)
        
        # Shared kernel queue by both traces
        kernel_queue = []

        # Current index of each trace
        trace_indices = [0, 0]

        # Record the last call end time for each trace
        trace_last_call_end_t = [0, 0]

        # Loop until both trace are finished
        while trace_indices[0] < len(t1) or trace_indices[1] < len(t2):

            # Candidate calls from t1 and t2
            trace_calls = [
                t1[trace_indices[0]] if trace_indices[0] < len(t1) else None,
                t2[trace_indices[1]] if trace_indices[1] < len(t2) else None
            ]

            # Choose the API call that starts earlier
            trace_call_start_t = [
                trace_last_call_end_t[i] + trace_calls[i].dur_from_last if trace_calls[i] else float('inf')
                for i in [0, 1]
            ]
            chosen = trace_call_start_t.index(min(trace_call_start_t))

            # Advance from last call
            trace_last_call_end_t[chosen] += trace_calls[chosen].dur_from_last
            trace_calls[chosen].start_t = trace_last_call_end_t[chosen]

            # Check finished kernels
            dequeued_kernels = self.dequeue_finished_kernels(kernel_queue, trace_call_start_t[chosen])

            curr_kernel = trace_calls[chosen].kernel
            trace = [t1, t2][chosen]

            # Synchronization call
            if "Synchronize" in curr_kernel.name:

                # Wait until all kernels to finish
                last_trace_kernel = self.get_last_kernel_of_trace(kernel_queue, trace)

                if last_trace_kernel:
                    trace_last_call_end_t[chosen] = max(last_trace_kernel.end_t, trace_last_call_end_t[chosen] + Simulator.min_sync_call_t)
                else:
                    trace_last_call_end_t[chosen] += Simulator.min_sync_call_t

            # launch a kernel
            else:

                # The soonest kernel can start is after this call finishes + some overhead
                min_kernel_start_t = trace_last_call_end_t[chosen] + trace_calls[chosen].dur + Simulator.kernel_launch_overhead
                
                if kernel_queue:
                    last_kernel = kernel_queue[-1]
                    curr_kernel.start_t = max(last_kernel.end_t + Simulator.kernel_swtch_overhead, min_kernel_start_t)

                    assert(curr_kernel.start_t > last_kernel.end_t)
                else:
                    curr_kernel.start_t = min_kernel_start_t
                
                curr_kernel.end_t = curr_kernel.start_t + curr_kernel.dur
                kernel_queue.append(curr_kernel)

                trace_last_call_end_t[chosen] += trace_calls[chosen].dur
            
            trace_indices[chosen] += 1
            if trace_indices[chosen] == len([t1, t2][chosen]):
                if self.logging:
                    print(f"{[t1, t2][chosen].model_name} has finished.")

            trace_calls[chosen].end_t = trace_last_call_end_t[chosen]

            if sim_time is not None:
                if not finished[chosen]:

                    finished_kernels.extend(dequeued_kernels)

                    if traces[chosen] in self.trace_throughput_monitor:
                        iters_finished[chosen] = self.trace_throughput_monitor[traces[chosen]]["total_iters"]
                        finish_time[chosen] = trace_calls[chosen].end_t

                    if finish_time[chosen] > sim_time_ns:
                        finished[chosen] = True
                
        t1_max_end_t = t1.run_check()
        t2_max_end_t = t2.run_check()

        check_time_share(t1, t2)

        if sim_time is None:
            for call in t1 + t2:
                if "Synchronize" not in call.kernel.name:
                    finished_kernels.append(call.kernel)
            
            finished_kernels.sort(key=lambda kernel: kernel.start_t)        

        gr_active = self.compute_gr_actice(finished_kernels)
        print(f"GR Activce: {gr_active}")

        if kernel_queue:
            assert((kernel_queue[-1].end_t) == max(t1_max_end_t, t2_max_end_t))
        else:
            assert(max(trace_last_call_end_t) == max(t1_max_end_t, t2_max_end_t))

        if sim_time is None:
            iters_finished[0] = t1.get_iterations()
            iters_finished[1] = t2.get_iterations()
            finish_time[0] = t1_max_end_t
            finish_time[1] = t2_max_end_t
        
        return finish_time[0], iters_finished[0], finish_time[1], iters_finished[1], gr_active