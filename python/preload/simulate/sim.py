from preload.simulate.trace_util import Trace, check_time_share
from copy import copy

class Simulator:

    # kernel switch overhead (switch from one kernel to next)
    kernel_switch_overhead = 600

    # minimum time to wait for kernel to start after CUDA API call
    # this is hidden is there are unfinished kernels
    kernel_launch_overhead = 1000

    # minimum time for a synchronization call to return
    min_sync_call_t = 4000
    
    def __init__(self, logging=False):
        self.logging = logging
        self.trace_throughput_monitor = {}
        self.sample_frequency = 5 * 10 ** 9
        self.last_next_start_time = 0

        if logging:
            print(f"logging frequency is every {self.sample_frequency / (10 ** 9)} seconds")

    def compute_gr_actice(self, _kernels):
        gr_active_time = 0

        for _idx, kernel in enumerate(_kernels):
            gr_active_time += (kernel.end_t - kernel.start_t)
            
        last_kernel_end_t = _kernels[-1]

        gr_active_rate = gr_active_time / last_kernel_end_t.end_t
        return gr_active_rate

    def log_finished_kernel(self, kernel):
        if kernel.trace not in self.trace_throughput_monitor:
            self.trace_throughput_monitor[kernel.trace] = {}
        
        if not self.trace_throughput_monitor[kernel.trace]:
            self.trace_throughput_monitor[kernel.trace]["total_iters"] = 0
            self.trace_throughput_monitor[kernel.trace]["interval_iters"] = 0
        
        if kernel.iter_head:
            self.trace_throughput_monitor[kernel.trace]["total_iters"] += 1
            self.trace_throughput_monitor[kernel.trace]["interval_iters"] += 1


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
                self.log_finished_kernel(first_kernel)
            else:
                break

        if self.logging:
            if (curr_t // self.sample_frequency) > (self.last_next_start_time // self.sample_frequency):
                
                time_since_ckpt = (curr_t - self.last_next_start_time) / (10 ** 9)

                for trace in self.trace_throughput_monitor:
                    interval_iters = self.trace_throughput_monitor[trace]["interval_iters"]
                    print(f"{trace.model_name}: {interval_iters / time_since_ckpt} iters/s")
                    self.trace_throughput_monitor[trace]["interval_iters"] = 0
                
                self.last_next_start_time = curr_t
        
        return dequeued_list

    
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
                    curr_kernel.start_t = max(last_kernel.end_t + Simulator.kernel_switch_overhead, min_kernel_start_t)
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
            trace_call_end_t = [
                trace_last_call_end_t[i] + trace_calls[i].dur_from_last if trace_calls[i] else float('inf')
                for i in [0, 1]
            ]
            chosen = trace_call_end_t.index(min(trace_call_end_t))

            # Advance from last call
            trace_last_call_end_t[chosen] += trace_calls[chosen].dur_from_last
            trace_calls[chosen].start_t = trace_last_call_end_t[chosen]

            # Check finished kernels
            dequeued_kernels = self.dequeue_finished_kernels(kernel_queue, trace_call_end_t[chosen])

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
                    curr_kernel.start_t = max(last_kernel.end_t + Simulator.kernel_switch_overhead, min_kernel_start_t)

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


class TwoJobTimeSharingFairSimulator(Simulator):
    
    def __init__(self):
        super().__init__()
        self.traces = None
        self.curr_turn = 0
        self.kernel_queue = []
        self.kernel_queue_t1 = []
        self.kernel_queue_t2 = []
        self.next_start_time = -1

    def append_to_kernel_queue(self, kernel_event):
        self.kernel_queue.append(kernel_event)
        if kernel_event.trace == self.traces[0]:
            self.kernel_queue_t1.append(kernel_event)
        else:
            self.kernel_queue_t2.append(kernel_event)
    
    def remove_from_kernel_queue(self, kernel_event):

        for i, kernel in enumerate(self.kernel_queue):
            if kernel is kernel_event:
                self.kernel_queue.pop(i)
                break

        if kernel_event.trace == self.traces[0]:
            assert(self.kernel_queue_t1[0] is kernel_event)
            self.kernel_queue_t1.pop(0)
        else:
            assert(self.kernel_queue_t2[0] is kernel_event)
            self.kernel_queue_t2.pop(0)

    def compute_kernel_finish_time(self, trace):

        kernel_queue_copy = []
        for kernel in self.kernel_queue:
            kernel_queue_copy.append(copy(kernel))

        latest_time = -1
        next_start_t = self.next_start_time
        trace_idx = self.traces.index(trace)
        turn = self.curr_turn

        t1_kernels = [kernel for kernel in kernel_queue_copy if kernel.trace == self.traces[0]]
        t2_kernels = [kernel for kernel in kernel_queue_copy if kernel.trace == self.traces[1]]
        t1_t2_list = [t1_kernels, t2_kernels]
    
        while len(t1_t2_list[trace_idx]) > 0:
            if len(t1_t2_list[1 - trace_idx]) == 0:
                idx = trace_idx
            else:
                # if already started, then finish it first
                if t1_kernels[0].start_t >= 0:
                    idx = 0
                    assert(t2_kernels[0].start_t < 0)
                elif t2_kernels[0].start_t >= 0:
                    idx = 1
                    assert(t1_kernels[0].start_t < 0)
                else:
                    idx = turn
                    # switch turn
                    turn = 1 - turn

            queue_to_run = t1_t2_list[idx]
            first_kernel = queue_to_run[0]

            if first_kernel.start_t < 0:
                first_kernel.start_t = max(next_start_t, first_kernel.enqueue_time)

            # remove the first kernel
            for i, kernel in enumerate(kernel_queue_copy):
                if kernel is first_kernel:
                    kernel_queue_copy.pop(i)
                    break
            t1_t2_list[idx].pop(0)
            first_kernel.end_t = first_kernel.start_t + first_kernel.dur

            if idx == trace_idx:
                latest_time = first_kernel.end_t
            next_start_t = first_kernel.end_t + Simulator.kernel_switch_overhead
        
        return latest_time + 1
        
    def dequeue_finished_kernels(self, curr_t):

        dequeued_list = []

        t1_t2_list = [self.kernel_queue_t1, self.kernel_queue_t2]

        while True:

            no_t1_kernels = len(self.kernel_queue_t1) == 0
            no_t2_kernels = len(self.kernel_queue_t2) == 0
            
            if no_t1_kernels and no_t2_kernels:
                break
            elif no_t1_kernels:
                idx = 1
            elif no_t2_kernels:
                idx = 0
            else:
                # both have something to launch

                # if already started, then finish it first
                if self.kernel_queue_t1[0].start_t >= 0:
                    idx = 0
                    assert(self.kernel_queue_t2[0].start_t < 0)
                elif self.kernel_queue_t2[0].start_t >= 0:
                    idx = 1
                    assert(self.kernel_queue_t1[0].start_t < 0)
                else:
                    # round robin
                    idx = self.curr_turn
                    # switch turn
                    self.curr_turn = 1 - self.curr_turn

            queue_to_run = t1_t2_list[idx]
            first_kernel = queue_to_run[0]

            # has not started yet
            if first_kernel.start_t < 0:
                if curr_t >= self.next_start_time:
                    assert(curr_t >= first_kernel.enqueue_time)
                    first_kernel.start_t = max(first_kernel.enqueue_time, self.next_start_time)
                else:
                    break

            # try running
            if curr_t - first_kernel.start_t >= first_kernel.dur:

                # remove from kernel queue
                self.remove_from_kernel_queue(first_kernel)
                self.log_finished_kernel(first_kernel)

                dequeued_list.append(first_kernel)
                first_kernel.end_t = first_kernel.start_t + first_kernel.dur
                self.next_start_time = first_kernel.end_t + Simulator.kernel_switch_overhead
            else:
                break
    
        return dequeued_list

    def simulate(self, t1: Trace, t2: Trace, sim_time=None):

        traces = [t1, t2]
        finish_time = [-1, -1]
        iters_finished = [0, 0]
        finished = [False, False]
        finished_kernels = []
        self.traces = [t1, t2]
        if sim_time:
            # convert to nanoseconds
            sim_time_ns = sim_time * (10 ** 9)

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
            trace_call_end_t = []
            for i in [0, 1]:
                end_t_i = 0
                if not trace_calls[i]:
                    end_t_i = float('inf')
                else:
                    call_kernel = trace_calls[i].kernel

                    if "Synchronize" in call_kernel.name:
                        end_t_i = max(trace_last_call_end_t[i] + trace_calls[i].dur_from_last + Simulator.min_sync_call_t, self.compute_kernel_finish_time(traces[i]))
                    else:
                        end_t_i = trace_last_call_end_t[i] + trace_calls[i].dur_from_last + trace_calls[i].dur
               
                trace_call_end_t.append(end_t_i)

            chosen = trace_call_end_t.index(min(trace_call_end_t))

            trace_calls[chosen].start_t = trace_last_call_end_t[chosen] + trace_calls[chosen].dur_from_last
            trace_last_call_end_t[chosen] = trace_call_end_t[chosen]

            # Check finished kernels
            dequeued_kernels = self.dequeue_finished_kernels(trace_call_end_t[chosen])

            curr_kernel = trace_calls[chosen].kernel

            # Synchronization call
            if "Synchronize" in curr_kernel.name:

                if chosen == 0:
                    assert(len(self.kernel_queue_t1) == 0)
                else: 
                    assert(len(self.kernel_queue_t2) == 0)
                
                assert(len(self.kernel_queue_t1) + len(self.kernel_queue_t2) == len(self.kernel_queue))
            # launch a kernel
            else:
                curr_kernel.enqueue_time = trace_last_call_end_t[chosen]
                self.append_to_kernel_queue(curr_kernel)

            trace_calls[chosen].end_t = trace_last_call_end_t[chosen]

            trace_indices[chosen] += 1
            if trace_indices[chosen] == len([t1, t2][chosen]):
                if self.logging:
                    print(f"{[t1, t2][chosen].model_name} has finished.")

            if sim_time is not None:
                if not finished[chosen]:

                    finished_kernels.extend(dequeued_kernels)

                    if traces[chosen] in self.trace_throughput_monitor:
                        iters_finished[chosen] = self.trace_throughput_monitor[traces[chosen]]["total_iters"]
                        finish_time[chosen] = trace_calls[chosen].end_t

                    if finish_time[chosen] > sim_time_ns:
                        finished[chosen] = True

        t1_finish_time = self.compute_kernel_finish_time(t1)
        t2_finish_time = self.compute_kernel_finish_time(t2)

        self.dequeue_finished_kernels(max(t1_finish_time, t2_finish_time))

        assert(len(self.kernel_queue) == 0)

        t1_max_end_t = t1.run_check()
        t2_max_end_t = t2.run_check()

        check_time_share(t1, t2)

        if sim_time is None:
            for call in t1.trace_events + t2.trace_events:
                if "Synchronize" not in call.kernel.name:
                    finished_kernels.append(call.kernel)
            
            finished_kernels.sort(key=lambda kernel: kernel.start_t)        

        gr_active = self.compute_gr_actice(finished_kernels)
        print(f"GR Activce: {gr_active}")

        if sim_time is None:
            iters_finished[0] = t1.get_iterations()
            iters_finished[1] = t2.get_iterations()
            finish_time[0] = t1_max_end_t
            finish_time[1] = t2_max_end_t
        
        return finish_time[0], iters_finished[0], finish_time[1], iters_finished[1], gr_active