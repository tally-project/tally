# Tally: Non-Intrusive Performance Isolation for GPUs

Tally is a non-intrusive GPU sharing mechanism that provides robust performance isolation and seamless workload compatibility. It employs block-level GPU kernel scheduling to mitigate interference from workload co-execution, ensuring high-priority tasks (e.g., real-time inference) can effectively maintain their performance when sharing GPUs with best-effort workloads (e.g., training).

For more details, please refer to our paper: [Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads](https://arxiv.org/abs/2410.07381)

### Reproducing Results

Benchmark scripts for reproducing the results from our paper are available in the [tally-bench](https://github.com/tally-project/tally-bench) repository.

### Citation

```
@inproceedings{10.1145/3669940.3707282,
    author = {Zhao, Wei and Jayarajan, Anand and Pekhimenko, Gennady},
    title = {Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads},
    year = {2025},
    isbn = {9798400706981},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3669940.3707282},
    doi = {10.1145/3669940.3707282},
    booktitle = {Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1},
    pages = {1052–1068},
    location = {Rotterdam, Netherlands},
    series = {ASPLOS '25}
}
```
