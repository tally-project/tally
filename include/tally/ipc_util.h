#ifndef TALLY_IPC_UTIL_H
#define TALLY_IPC_UTIL_H

#define CLIENT_SEND_MSG_AND_FREE \
    while (!TallyClient::client->send_ipc->send(msg, msg_len, 60000)) { \
        TallyClient::client->send_ipc->wait_for_recv(1); \
    } \
    std::free(msg);

#define CLIENT_RECV_MSG \
    ipc::buff_t buf; \
    while (buf.empty()) { \
        buf = TallyClient::client->recv_ipc->recv(1000); \
    } \
    const char *dat = buf.get<const char *>();

#endif // TALLY_IPC_UTIL_H