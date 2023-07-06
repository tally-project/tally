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
}

template<class Archive>
void serialize(Archive & ar, CudaLaunchConfig & g, const unsigned int version)
{
    ar & g.use_original;
    ar & g.use_sliced;
    ar & g.use_ptb;

    ar & g.use_cuda_graph;
    ar & g.threads_per_slice;
    ar & g.num_blocks_per_sm;
}

template<class Archive>
void serialize(Archive & ar, CudaLaunchKeyConfig & g, const unsigned int version)
{
    ar & g.key;
    ar & g.config;
}

template<class Archive>
void serialize(Archive & ar, TransformData & g, const unsigned int version)
{
    ar & g.cubin_data;
    ar & g.sliced_data;
    ar & g.ptb_data;
}

template<class Archive>
void serialize(Archive & ar, TransformCache & g, const unsigned int version)
{
    ar & g.cubin_map;
}

template<class Archive>
void serialize(Archive & ar, PerformanceCache & g, const unsigned int version)
{
    ar & g.kernel_config_map;
    ar & g.config_latency_map;
}

} // namespace serialization
} // namespace boost

#endif // TALLY_SERIALIZATION_H