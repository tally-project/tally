#include <cstring>
#include <memory>

#include <tally/client.h>
#include <tally/generated/cuda_api.h>

std::unique_ptr<TallyClient> TallyClient::client = std::make_unique<TallyClient>();