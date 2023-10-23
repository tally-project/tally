#ifndef TALLY_SERIALIZATION_H
#define TALLY_SERIALIZATION_H

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <tally/cache_struct.h>

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive & ar, dim3 & g, const unsigned int version)
{
    ar & g.x;
    ar & g.y;
    ar & g.z;
}

template<class Archive>
void serialize(Archive & ar, CudaLaunchKey & g, const unsigned int version)
{
    ar & g.kernel_name;
    ar & g.gridDim;
    ar & g.blockDim;
    ar & g.cubin_uid;
}

template<class Archive>
void serialize(Archive & ar, CudaLaunchKeyPair & g, const unsigned int version)
{
    ar & g.launch_key_1;
    ar & g.launch_key_2;
}

template<class Archive>
void serialize(Archive & ar, CudaLaunchConfig & g, const unsigned int version)
{
    ar & g.use_original;
    ar & g.use_ptb;
    ar & g.use_dynamic_ptb;
    ar & g.use_preemptive_ptb;

    ar & g.num_blocks_per_sm;
}

template<class Archive>
void serialize(Archive & ar, CudaLaunchKeyConfig & g, const unsigned int version)
{
    ar & g.key;
    ar & g.config;
}

template<class Archive>
void serialize(Archive & ar, CudaLaunchKeyConfigPair & g, const unsigned int version)
{
    ar & g.key_config_1;
    ar & g.key_config_2;
}

template<class Archive>
void serialize(Archive & ar, KernelProfileMetrics & g, const unsigned int version)
{
    ar & g.latency_ms;
    ar & g.norm_speed;
    ar & g.iters;
}

template<class Archive>
void serialize(Archive & ar, CudaLaunchKeyConfigResult & g, const unsigned int version)
{
    ar & g.key;
    ar & g.config;
    ar & g.metrics;
}

template<class Archive>
void serialize(Archive & ar, WorkloadPerformance & g, const unsigned int version)
{
    ar & g.latency_ms;
    ar & g.speedup;
}

template<class Archive>
void serialize(Archive & ar, CudaLaunchKeyConfigPairResult & g, const unsigned int version)
{
    ar & g.config_key_norm_speed_1;
    ar & g.config_key_norm_speed_2;
    ar & g.fixed_workload_perf;
    ar & g.unfair_workload_perf;
}

template<class Archive>
void serialize(Archive & ar, CubinData & g, const unsigned int version)
{
    ar & g.cubin_uid;
    ar & g.cubin_data;
    ar & g.kernel_args;
    ar & g.ptx_str;
    ar & g.fatbin_str;
}

template<class Archive>
void serialize(Archive & ar, CubinCache & g, const unsigned int version)
{
    ar & g.uid_counter;
    ar & g.cubin_map;
}

template<class Archive>
void serialize(Archive & ar, PerformanceCache & g, const unsigned int version)
{
    ar & g.single_kernel_perf_map;
    ar & g.single_kernel_best_config_map;
    ar & g.kernel_pair_perf_map;
    ar & g.kernel_pair_best_config_map;
}

} // namespace serialization
} // namespace boost

#endif // TALLY_SERIALIZATION_H