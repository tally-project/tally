#ifndef TALLY_IPC_UTIL_H
#define TALLY_IPC_UTIL_H

#define LOG_ERR_AND_EXIT(STR, ERR)          \
    std::cout << STR << ERR << std::endl;   \
    exit(1);

#define IOX_RECV_RETURN_STATUS(RET_TYPE)                                                \
    while(!TallyClient::client->iox_client->take()                                      \
        .and_then([&](const auto& responsePayload) {                                    \
            auto response = static_cast<const RET_TYPE*>(responsePayload);              \
            err = *response;                                                            \
            TallyClient::client->iox_client->releaseResponse(responsePayload);          \
        }))                                                                             \
    {}

#if defined(RUN_LOCALLY)
    #define IOX_CLIENT_ACQUIRE_LOCK 
#else
    #define IOX_CLIENT_ACQUIRE_LOCK                                                             \
        if (!TallyClient::client->has_connected) TallyClient::client->connect_to_server();      \
        std::lock_guard<std::recursive_mutex> guard(TallyClient::client->iox_mtx);
#endif

#endif // TALLY_IPC_UTIL_H