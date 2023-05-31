from trace_util import Trace, check_time_share

class Simulator:

    # kernel switch overhead (switch from one kernel to next)
    kernel_swtch_overhead = 600

    # minimum time to wait for kernel to start after CUDA API call
    # this is hidden is there are unfinished kernels
    kernel_launch_overhead = 1000

    # minimum time for a synchronization call to return
    min_sync_call_t = 4000
    
    def __init__(self, monitor_throughput=True):
        self.monitor_throughput = monitor_throughput
        if monitor_throughput:
            self.trace_throughput_monitor = {}
            self.sample_frequency = 5 * 10 ** 9
            self.last_ckpt_time = 0
            print(f"logging frequency is every {self.sample_frequency / (10 ** 9)} seconds")

    def compute_gr_actice(self, _kernels):
        gr_active_time = 0

        for _idx, kernel in enumerate(_kernels):
            gr_active_time += (kernel.end_t - kernel.start_t)
            
        last_kernel_end_t = _kernels[-1]

        gr_active_rate = gr_active_time / last_kernel_end_t.end_t
        return gr_active_rate

    def dequeue_finished_kernels(self, kernel_queue, curr_t):

        # Dequeue kernels that have finished
        while True:
            if not kernel_queue:
                break
            first_kernel = kernel_queue[0]
            if first_kernel.end_t <= curr_t:
                kernel_queue.pop(0)

                assert(first_kernel.trace)
                if self.monitor_throughput:
                    if first_kernel.trace not in self.trace_throughput_monitor:
                        self.trace_throughput_monitor[first_kernel.trace] = {}
                    
                    if not self.trace_throughput_monitor[first_kernel.trace]:
                        self.trace_throughput_monitor[first_kernel.trace]["num_iters"] = 0
                    
                    if first_kernel.iter_head:
                        self.trace_throughput_monitor[first_kernel.trace]["num_iters"] += 1
            
            else:
                break

        if self.monitor_throughput:
            if (curr_t // self.sample_frequency) > (self.last_ckpt_time // self.sample_frequency):
                
                time_since_ckpt = (curr_t - self.last_ckpt_time) / (10 ** 9)

                for trace in self.trace_throughput_monitor:
                    num_iters = self.trace_throughput_monitor[trace]["num_iters"]
                    print(f"{trace.model_name}: {num_iters / time_since_ckpt} iters/s")
                    self.trace_throughput_monitor[trace]["num_iters"] = 0
                
                self.last_ckpt_time = curr_t

    
    def simulate(*arg):
        raise NotImplementedError()


class SingleJobSimulator(Simulator):

    def __init__(self):
        super().__init__()

    def simulate(self, trace: Trace):
        
        curr_t = 0
        kernel_queue = []

        for call in trace:

            if isinstance(call, str):
                print(call)

            # Advance from last call
            curr_t += call.dur_from_last
            call.start_t = curr_t

            # Check finished kernels
            self.dequeue_finished_kernels(kernel_queue, curr_t)

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
        
        max_end_t = trace.run_check()

        trace_kernels = []
        for call in trace:
            if "Synchronize" not in call.kernel.name:
                trace_kernels.append(call.kernel)

        gr_active = self.compute_gr_actice(trace_kernels)
        print(f"GR Activce: {gr_active}")

        if kernel_queue:
            assert(max_end_t == kernel_queue[-1].end_t)
            return kernel_queue[-1].end_t

        assert(max_end_t == curr_t)
        return curr_t
    

class TwoJobTimeSharingSimulator(Simulator):
    
    def __init__(self):
        super().__init__()

    def get_last_kernel_of_trace(self, kernel_queue, trace):

        for kernel in reversed(kernel_queue):
            if kernel.trace == trace:
                return kernel
        
        return None

    def simulate(self, t1: Trace, t2: Trace):
        
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
            self.dequeue_finished_kernels(kernel_queue, trace_call_start_t[chosen])

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
                if self.monitor_throughput:
                    print(f"{[t1, t2][chosen].model_name} has finished.")
            trace_calls[chosen].end_t = trace_last_call_end_t[chosen]
    
        t1_max_end_t = t1.run_check()
        t2_max_end_t = t2.run_check()

        check_time_share(t1, t2)

        t1_kernels = []
        for call in t1:
            if "Synchronize" not in call.kernel.name:
                t1_kernels.append(call.kernel)
        
        t2_kernels = []
        for call in t2:
            if "Synchronize" not in call.kernel.name:
                t2_kernels.append(call.kernel)
        
        all_kernels = t1_kernels + t2_kernels
        all_kernels.sort(key=lambda kernel: kernel.start_t)

        gr_active = self.compute_gr_actice(all_kernels)
        print(f"GR Activce: {gr_active}")

        if kernel_queue:
            assert((kernel_queue[-1].end_t) == max(t1_max_end_t, t2_max_end_t))
        else:
            assert(max(trace_last_call_end_t) == max(t1_max_end_t, t2_max_end_t))
        
        return t1_max_end_t, t2_max_end_t