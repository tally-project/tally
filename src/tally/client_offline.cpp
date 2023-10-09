#include <tally/client_offline.h>
#include <tally/cuda_launch.h>

TallyClientOffline *TallyClientOffline::client_offline;

__attribute__((__constructor__)) void init_client()
{
    TallyClientOffline::client_offline = new TallyClientOffline();
}

TallyClientOffline::TallyClientOffline()
{}

const CudaLaunchConfig CudaLaunchConfig::default_config = CudaLaunchConfig();
