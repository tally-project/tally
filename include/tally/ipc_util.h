#ifndef TALLY_IPC_UTIL_H
#define TALLY_IPC_UTIL_H

#define CLIENT_SEND_MSG_AND_FREE \
    while (!TallyClient::client->send_ipc->send(msg, msg_len, 60000)) { \
        TallyClient::client->send_ipc->wait_for_recv(1); \
    } \
    if (msg != TallyClient::client->msg) { \
        std::free(msg); \
    }

#define CLIENT_RECV_MSG \
    ipc::buff_t ipc_buf; \
    while (ipc_buf.empty()) { \
        ipc_buf = TallyClient::client->recv_ipc->recv(10000); \
    } \
    const char *dat = ipc_buf.get<const char *>();

#define LOG_ERR_AND_EXIT(STR, ERR) \
    std::cout << STR << ERR << std::endl; \
    exit(1);

#define IOX_RECV_RETURN_STATUS(RET_TYPE) \
    while(!TallyClient::client->iox_client->take() \
        .and_then([&](const auto& responsePayload) { \
            auto response = static_cast<const RET_TYPE*>(responsePayload); \
            err = *response; \
            TallyClient::client->iox_client->releaseResponse(responsePayload); \
        })) \
    {}

#endif // TALLY_IPC_UTIL_H