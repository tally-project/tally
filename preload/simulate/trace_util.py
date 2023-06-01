import json

from preload.simulate.util import split_and_strip, longest_common_sequence

class Event:

    def __init__(self, trace):
        self.trace = trace

        # Will be set by simulator
        self.start_t = -1
        self.end_t = -1

class CudaEvent(Event):

    def __init__(self, name, dur, dur_from_last=0, trace=None):
        super().__init__(trace)
        self.name = name
        self.dur = dur
        self.dur_from_last = dur_from_last
        self.kernel = None
    
    def __str__(self):
        if self.kernel:
            _str = f"CUDA API call:\n" + \
                   f"\tBegin : {self.start_t}ns\n" + \
                   f"\tEnd : {self.end_t}ns\n" + \
                   f"\tLatency : {self.dur}ns\n" + \
                   f"\tTime elapsed from last call: {self.dur_from_last}ns\n" + \
                   f"\tKernel name: {self.kernel.name}\n" + \
                   f"\tKernel latency: {self.kernel.dur}ns"
                   
            return _str
        else:
            return "Unknown kernel"

    # For matching cpu/gpu events
    def __eq__(self, other):
        if not isinstance(other, KernelEvent):
            raise NotImplementedError
        return self.name == other.name


class KernelEvent(Event):

    def __init__(self, name, dur, trace=None):
        super().__init__(trace)
        self.name = name
        self.dur = dur

        # will be set by simulator
        self.iter_head = False
    
    def __str__(self):
        _str = f"CUDA kernel:\n" + \
                f"\tBegin : {self.start_t}ns\n" + \
                f"\tEnd : {self.end_t}ns\n" + \
                f"\tKernel name: {self.name}\n" + \
                f"\tLatency : {self.dur}ns"
                
        return _str

    # For matching cpu/gpu events
    def __eq__(self, other):
        if not isinstance(other, CudaEvent):
            raise NotImplementedError
        return self.name == other.name


class Trace:
    def __init__(self, name):
        self.model_name = name
        self.trace_events = []
    
    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx < len(self.trace_events):
            call = self.trace_events[self._idx]
            self._idx += 1
            return call
        else:
            raise StopIteration
    
    def __getitem__(self, _idx):
        return self.trace_events[_idx]

    def __len__(self):
        return len(self.trace_events)

    def run_check(self):

        last_kernel_end_time = -1
        max_end_t = -1

        for _idx, call in enumerate(self.trace_events):
            
            # Assert that timestamps are set
            assert(call.start_t >= 0)
            assert(call.end_t >= 0)

            # end time > start time
            assert(call.end_t > call.start_t)

            # start time of event[i] > end time of event[i - 1]
            if _idx > 0:
                assert(self.trace_events[_idx].start_t > self.trace_events[_idx - 1].end_t)
    
            kernel = call.kernel

            if "Synchronize" not in kernel.name:
                assert(kernel.start_t >= 0)
                assert(kernel.end_t >= 0)
                if kernel.end_t <= kernel.start_t:
                    print(f"{kernel.name} kernel.end_t: {kernel.end_t}, kernel.start_t: {kernel.start_t}")
                assert(kernel.end_t > kernel.start_t)
 
                assert(kernel.start_t > last_kernel_end_time)
                last_kernel_end_time = kernel.end_t
            
            max_end_t = max(max_end_t, call.end_t, kernel.end_t)
        
        return max_end_t

    def load_trace(self):
        raise NotImplementedError
    
    def get_iterations(self):

        id = 1
        kernel_name_to_id = {}
        kernel_seq = []

        for call in self.trace_events:
            kernel_name = call.kernel.name
            if kernel_name not in kernel_name_to_id:
                kernel_name_to_id[kernel_name] = id
                id += 1
            kernel_seq.append(kernel_name_to_id[kernel_name])
        
        # Use the first 10 as a sequence matcher
        guess_seq = kernel_seq[:10]

        count_iter = 0

        for i in range(len(kernel_seq) - len(guess_seq) + 1):
            if kernel_seq[i:i+len(guess_seq)] == guess_seq:
                count_iter += 1

                self.trace_events[i].kernel.iter_head = True

        return count_iter


class PreloadTrace(Trace):
    def __init__(self, name, cpu_trace=None, gpu_trace=None):
        super().__init__(name)

        if cpu_trace and gpu_trace:
            self.load_trace(cpu_trace, gpu_trace)
        
        # Set Iteration heads
        self.get_iterations()

    def find_most_common_func(self, cpu_calls, gpu_kernels):
        cpu_func_count = {}
        gpu_func_count = {}

        for call in cpu_calls:
            func_name = call.name

            if func_name not in cpu_func_count:
                cpu_func_count[func_name] = 0

            cpu_func_count[func_name] += 1
        
        for kernel in gpu_kernels:

            func_name = kernel.name

            if func_name not in gpu_func_count:
                gpu_func_count[func_name] = 0

            gpu_func_count[func_name] += 1
        
        func_names_with_count = []
        for func_name in cpu_func_count:
            if func_name not in gpu_func_count:
                continue
            if cpu_func_count[func_name] == gpu_func_count[func_name]:
                func_names_with_count.append((func_name, cpu_func_count[func_name]))
        
        func_names_with_count.sort(key=lambda pair : pair[1])

        most_common_func = func_names_with_count[-1][0]
        print(f"most_common_func is {most_common_func}")
        return most_common_func

    def partition_calls_by_func(self, calls, most_common_func_name):
        calls_partitions = []

        ckpt_idx = 0

        for idx, call in enumerate(calls):
            if call.name == most_common_func_name:
                partition = calls[ckpt_idx : idx + 1]
                calls_partitions.append(partition)

                ckpt_idx = idx + 1
        
        last_partition = calls[ckpt_idx + 1:]
        if len(last_partition) > 0:
            calls_partitions.append(last_partition)
        
        return calls_partitions
    
    def filter_by_common_seq(self, orig_seq, common_seq):
        common_seq_cpy = common_seq.copy()
        filtered_seq = []

        for call in orig_seq:
            if "Synchronize" in call.name:
                filtered_seq.append(call)
                continue
            if call.name == common_seq_cpy[0].name:
                filtered_seq.append(call)
                common_seq_cpy.pop(0)
        
        return filtered_seq
            
    def load_trace(self, cpu_trace, gpu_trace):
        
        last_end_ns = -1

        kernels = []
        cpu_calls = []

        with open(gpu_trace) as f:
            for line in f:

                if "Kernel Time:" in line:
                    line = line.strip()
                    func_name, time_str = split_and_strip(line, "Kernel Time:")
                    duration = float(time_str) * 1000000

                    kernel_event = KernelEvent(func_name, duration, trace=self)
                    kernels.append(kernel_event)
            
        with open(cpu_trace) as f:
            for line in f:
                
                if "Start:" in line and "End:" in line:
                    line = line.strip()

                    func_name, time_str = split_and_strip(line, "Start:")
                    start_t, end_t = split_and_strip(time_str, "End:")
                    start_t, end_t = int(start_t), int(end_t)
                    duration = end_t - start_t

                    # CPU time in between CUDA API calls
                    dur_from_last = 0
                    if last_end_ns > 0:
                        dur_from_last = start_t - last_end_ns

                    last_end_ns = end_t

                    cuda_api_call = CudaEvent(func_name, duration, dur_from_last, trace=self)

                    cpu_calls.append(cuda_api_call)

        most_common_func_name = self.find_most_common_func(cpu_calls, kernels)
        
        cpu_calls_partitions = self.partition_calls_by_func(cpu_calls, most_common_func_name)
        kernel_calls_partitions = self.partition_calls_by_func(kernels, most_common_func_name)

        # Not necessarily, but assume this will hold most likely
        assert(len(cpu_calls_partitions) == len(kernel_calls_partitions))

        common_cpu_calls = []
        common_gpu_calls = []

        for part_idx in range(len(cpu_calls_partitions)):
            cpu_calls_partition = cpu_calls_partitions[part_idx]
            kernel_calls_partition = kernel_calls_partitions[part_idx]
            lcs = longest_common_sequence(cpu_calls_partition, kernel_calls_partition)

            common_cpu_part = self.filter_by_common_seq(cpu_calls_partition, lcs)
            common_gpu_part = self.filter_by_common_seq(kernel_calls_partition, lcs)

            common_cpu_calls.extend(common_cpu_part)
            common_gpu_calls.extend(common_gpu_part)

        for cpu_call in common_cpu_calls:
            if "Synchronize" in cpu_call.name:
                cpu_call.kernel = KernelEvent(cpu_call.name, dur=0, trace=self)
                continue
        
            assert(cpu_call.name == common_gpu_calls[0].name)
            cpu_call.kernel = common_gpu_calls.pop(0)
        
        print(f"len common_cpu_calls: {len(common_cpu_calls)}")
        print(f"len cpu_calls: {len(cpu_calls)}")


        self.trace_events = common_cpu_calls
                    

class NsysTrace(Trace):

    def __init__(self, name, file_path=None, start_id=None, end_id=None):
        super().__init__(name)

        if file_path and start_id and end_id:
            self.load_trace(file_path, start_id, end_id)

    def load_trace(self, file_name, start_id, end_id):

        kernel_types = ["kernel", "memcpy", "memset", "sync"]
        unprocessed_calls = []
        cuda_api_calls = []
        id_to_kernel_map = {}
        profile_overhead = 0

        last_end_ns = -1

        with open(file_name) as f:
            first_line = f.readline()
            meta_data = json.loads(first_line)["data"]

            # First pass, collect Kernel info
            for line in f:
                event = json.loads(line)

                if "CudaEvent" in event:
                    type = "CudaEvent"
                    if not any([kernel_type in event["CudaEvent"] for kernel_type in kernel_types]):
                        continue
                elif "TraceProcessEvent" in event:
                    type = "TraceProcessEvent"
                else:
                    continue

                corr_id = event[type]["correlationId"]
                if corr_id < start_id:
                    continue
                if corr_id > end_id:
                    break
                
                start_ns = int(event[type]["startNs"])
                end_ns = int(event[type]["endNs"])

                if start_ns <= 0:
                    continue

                duration = end_ns - start_ns

                # CUDA API call
                if "TraceProcessEvent" in event:
                    unprocessed_calls.append(event)

                # CUDA Kernel
                elif "CudaEvent" in event:
                    stream_id = event["CudaEvent"]["streamId"]
                    if "kernel" in event["CudaEvent"]:
                        kernel_name = meta_data[int(event["CudaEvent"]["kernel"]["demangledName"])]
                    elif "memcpy" in event["CudaEvent"]:
                        kernel_name = "cudaMemcpyAsync"
                    elif "sync" in event["CudaEvent"]:
                        kernel_name = "cudaStreamSynchronize"
                    elif "memset" in event["CudaEvent"]:
                        kernel_name = "cudaMemsetAsync"
                    else:
                        print("Unknown kernel.")

                    kernel_event = KernelEvent(kernel_name, duration, trace=self)
                    id_to_kernel_map[corr_id] = kernel_event
            
            end_ns = float('inf')

            # With kernel info, start processing the CUDA API calls
            for event in unprocessed_calls:

                corr_id = event["TraceProcessEvent"]["correlationId"]
                assert(start_ns < end_ns)
                start_ns = int(event["TraceProcessEvent"]["startNs"])
                end_ns = int(event["TraceProcessEvent"]["endNs"])
                duration = end_ns - start_ns

                if corr_id not in id_to_kernel_map:
                    non_kernel_call = meta_data[int(event["TraceProcessEvent"]["name"])]
                    profile_overhead += duration
                    continue

                # CPU time in between CUDA API calls
                dur_from_last = 0
                if last_end_ns > 0:
                    dur_from_last = start_ns - last_end_ns

                last_end_ns = end_ns

                _kernel = id_to_kernel_map[corr_id]
                cuda_api_call = CudaEvent(_kernel.name, duration, dur_from_last, trace=self)
                cuda_api_call.kernel = _kernel

                cuda_api_calls.append(cuda_api_call)
            
            # Pop the the last kernel from last iteration
            # cuda_api_calls.pop(0)
        
        self.trace_events = cuda_api_calls

        # Set Iteration heads
        self.get_iterations()

# Check the kernels of trace1 and trace2 never overlaps
def check_time_share(t1: Trace, t2: Trace):
    
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

    for _idx, kernel in enumerate(all_kernels):
        assert(kernel.start_t >= 0)
        assert(kernel.end_t >= 0)
        assert(kernel.end_t > kernel.start_t)

        if _idx > 0:
            assert(kernel.start_t > all_kernels[_idx - 1].end_t)