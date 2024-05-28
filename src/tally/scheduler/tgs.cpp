#include <cassert>
#include <cfloat>
#include <random>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_util.h>
#include <tally/util.h>
#include <tally/env.h>

// Use this to indicate how many clients
int num_clients = 0;

#define GPU_MAX_NUM 8
#define MILLISEC (1000UL * 1000UL)
#define TIME_TICK (10)
#define FACTOR (32)
#define MAX_UTILIZATION (100)
#define CHANGE_LIMIT_INTERVAL (30)
#define USAGE_THRESHOLD (5)

const size_t g_spare_memory = 1ull << 30;

static double global_max_rate = 0.;
static double global_rate_counter = 0.;

inline double shift_window(double rate_window[], const int WINDOW_SIZE, double recv_rate) {
  double max_window_rate = 0;

  for (int i = WINDOW_SIZE-1; i > 0; --i) {
    double mean_rate = (rate_window[i] + rate_window[i-1]) / 2;
    max_window_rate = max_window_rate > mean_rate ? max_window_rate : mean_rate;
    rate_window[i] = rate_window[i-1];
  }
  rate_window[0] = recv_rate;

  return max_window_rate;
}

static int tgs_set_cpu_affinity(pthread_t thread_id, int core_id) {
    int ret;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    ret = pthread_setaffinity_np(thread_id, sizeof(cpuset), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "failed to set cpu affinity, core_id=%d", core_id);
        return -1;
    } else {
        ret = pthread_getaffinity_np(thread_id, sizeof(cpuset), &cpuset);
        if (ret != 0) {
            fprintf(stderr, "failed to get cpu affinity");
            return -1;
        } else {
            fprintf(stderr, "set returned by pthread_getaffinity_np() contained:");
            int cnt = 0, cpu_in_set = 0;
            for (int i = 0; i < CPU_SETSIZE; i++) {
                if (CPU_ISSET(i, &cpuset)) {
                    cnt++;
                    cpu_in_set = i;
                    fprintf(stderr, "  cpu=%d", i);
                }
            }
            // this should not happen though
            if (cnt != 1 || cpu_in_set != core_id) {
                fprintf(stderr, "failed to set cpu affinity with cpu=%d", core_id);
                return -1;
            }
        }
    }
    return 0;
}

namespace HighPriorityTGS {

static long long g_current_rate[GPU_MAX_NUM] = {};
static long long g_rate_counter[GPU_MAX_NUM] = {};
static int g_active_gpu[GPU_MAX_NUM] = {};
static CUuuid g_uuid[GPU_MAX_NUM];
static int g_gpu_id[GPU_MAX_NUM];

static void activate_rate_watcher();
static void *rate_watcher(void *);
static void rate_estimator(const long long);

static void initialization(CUdevice);

static inline void rate_estimator(const long long kernel_size) {
    CUdevice device = 0;
    cuCtxGetDevice(&device);

    if (!g_active_gpu[device])
        initialization(device);

    __sync_add_and_fetch_8(&g_rate_counter[device], kernel_size);
}

static void *rate_monitor(void *v_device) {
    const CUdevice device = (uintptr_t)v_device;
    const unsigned long duration = 5000;
    const struct timespec unit_time = {
        .tv_sec = duration / 1000,
        .tv_nsec = duration % 1000 * MILLISEC,
    };
    struct timespec req = unit_time, rem;

    fprintf(stderr, "[%d] rate_monitor start\n", device);

    g_rate_counter[device] = 0;
    while (g_active_gpu[device] > 0) {
        int ret = nanosleep(&req, &rem);
        if (ret < 0) {
        if (errno == EINTR) {
            req = rem;
            continue;
        }
        else fprintf(stderr, "nanosleep error: %s\n", strerror(errno));
        }
        else
        req = unit_time;
        
        g_current_rate[device] = g_rate_counter[device];

        g_rate_counter[device] = 0;
    }
    return NULL;
}

static void activate_rate_monitor(CUdevice device) {
    pthread_t tid;

    pthread_create(&tid, NULL, rate_monitor, (void *)(uintptr_t)device);
    tgs_set_cpu_affinity(tid, g_gpu_id[device]);

    #ifdef __APPLE__
    pthread_setname_np("rate_monitor");
    #else
    pthread_setname_np(tid, "rate_monitor");
    #endif
}

static void activate_rate_watcher(CUdevice device) {
    pthread_t tid;

    pthread_create(&tid, NULL, rate_watcher, (void *)(uintptr_t)device);
    tgs_set_cpu_affinity(tid, g_gpu_id[device]);

    #ifdef __APPLE__
    pthread_setname_np("rate_watcher");
    #else
    pthread_setname_np(tid, "rate_watcher");
    #endif
}

static void *rate_watcher(void *v_device) {
    const CUdevice device = (uintptr_t)v_device;
    const unsigned long duration = 5000;

    const struct timespec unit_time = {
        .tv_sec = duration / 1000,
        .tv_nsec = duration % 1000 * MILLISEC,
    };
    const struct timespec listen_time = {
        .tv_nsec = 100 * MILLISEC,
    };
    struct timespec req = listen_time, rem;
    const int WINDOW_SIZE = 5;
    double rate_window[WINDOW_SIZE];
    double max_rate = 0;

    while (g_active_gpu[device] > 0) {

        int clientfd;
        int loop_cnt = 0;

        // before the low-priority joins
        while (num_clients < 2) {
            if (g_active_gpu[device] <= 0) {
                return NULL;
            }

            nanosleep(&listen_time, &rem);
            
            if (loop_cnt < 20) {
                ++loop_cnt;
                continue;
            }
            else
                loop_cnt = 0;

            double rate_counter =  g_current_rate[device];
            double max_window_rate = shift_window(rate_window, WINDOW_SIZE, (double)rate_counter);

            double max_delta = (max_window_rate - max_rate) / max_rate;
            
            if (max_delta >= -0.2 && max_delta <= 0.2)
                max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
            else
                max_rate = max_window_rate;

            fprintf(stderr, "high prioirty max_rate: %lf\n", max_rate);
        }
        
        // if (rio_writen(clientfd, (void *)&max_rate, sizeof(double)) != sizeof(double)) {
        //   LOGGER(4, "rio_writen error\n");
        //   continue;
        // }

        global_max_rate = max_rate;
        fprintf(stderr, "high prioirty set global_max_rate: %lf\n", global_max_rate);

        int ret = 0;
        req = unit_time;
        while (g_active_gpu[device] > 0) {

        ret = nanosleep(&req, &rem);
        if (ret < 0) {
            if (errno == EINTR) {
            req = rem;
            continue;
            }
            else fprintf(stderr, "nanosleep error: %s\n", strerror(errno));
        }
        else
            req = unit_time;
        double rate_counter = g_current_rate[device];
        double max_window_rate = shift_window(rate_window, WINDOW_SIZE, (double)rate_counter);
        double max_delta = (max_window_rate - max_rate) / max_rate;
        
        if (max_delta >= -0.2 && max_delta <= 0.2)
            max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
        else
            max_rate = max_window_rate;

        //   if (rio_writen(clientfd, (void *)&rate_counter, sizeof(double)) != sizeof(double)) {
        //     LOGGER(4, "rio_writen error\n");
        //     break;
        //   }

            global_rate_counter = rate_counter;
            fprintf(stderr, "high prioirty set global_rate_counter: %lf\n", global_rate_counter);
        }

        // close(clientfd);

        max_rate *= 0.8;
    }
    return NULL;
}

static inline void initialization(CUdevice device) {
    g_active_gpu[device] = 1;

    fprintf(stderr, "initialize device %d\n", device);

    cuDeviceGetUuid(&g_uuid[device], device);

    int gpu_id = 0;
    for (int i = 0; i < 16; ++i) {
        gpu_id += (int)g_uuid[device].bytes[i];
    }
    gpu_id = (gpu_id % 8 + 8) % 8;
    g_gpu_id[device] = gpu_id;

    activate_rate_watcher(device);
    activate_rate_monitor(device);
}
}

namespace LowPriorityTGS {

static size_t g_used_memory = 0;

static long long g_rate_counter[GPU_MAX_NUM] = {};
static long long g_rate_limit[GPU_MAX_NUM] = {};
static long long g_rate_control_flag[GPU_MAX_NUM] = {};
static long long g_current_rate[GPU_MAX_NUM] = {};
static int g_active_gpu[GPU_MAX_NUM] = {};
static CUuuid g_uuid[GPU_MAX_NUM];
static int g_gpu_id[GPU_MAX_NUM];

const long long LIMIT_INITIALIZER = 20000;
const long long RATE_MIN = 1000;

#define TGS_SLOW_START 0
#define TGS_CONGESTION_AVOIDANCE 1

static const struct timespec g_cycle = {
    .tv_sec = 0,
    .tv_nsec = TIME_TICK * MILLISEC,
};


struct MemRange {
  CUdeviceptr devPtr;
  size_t count;
  CUdevice device;
  struct MemRange *successor, *precursor;
};

static struct MemRange *list_head = NULL;
static size_t list_size = 0;
static pthread_mutex_t g_map_mutex = PTHREAD_MUTEX_INITIALIZER;

static void *rate_watcher(void *);
static bool rate_limiter(const long long);
static void *limit_manager(void *);
static void init_rate_limit(long long, volatile long long *, int *);
static void *memory_transfer_routine(CUdevice device);

/*
 * memory transfer
 */

void init_list() {
  pthread_mutex_lock(&g_map_mutex);
  list_head = (struct MemRange*)malloc(sizeof(struct MemRange));
  list_head->count = 0;
  list_head->devPtr = -1;
  list_head->device = 0;
  list_head->precursor = NULL;
  list_head->successor = NULL;
  pthread_mutex_unlock(&g_map_mutex);
}


void list_insert(struct MemRange *pos, struct MemRange *item) {
  item->successor = pos->successor;
  item->precursor = pos;
  if (pos->successor)
    pos->successor->precursor = item;
  pos->successor = item;
  ++list_size;
}


void list_delete(struct MemRange *item) {
  if (item->precursor)
    item->precursor->successor = item->successor;
  if (item->successor)
    item->successor->precursor = item->precursor;
  item->precursor = NULL;
  item->successor = NULL;
  --list_size;
}


void allocate_mem(CUdeviceptr devPtr, size_t count, CUdevice device) {
  struct MemRange *item = (struct MemRange *)malloc(sizeof(struct MemRange));
  item->devPtr = devPtr;
  item->count = count;
  item->device = device;
  item->precursor = item->successor = NULL;
  
  if (list_head == NULL)
    init_list();

  pthread_mutex_lock(&g_map_mutex);
  g_used_memory += count;
  list_insert(list_head, item);
  pthread_mutex_unlock(&g_map_mutex);
}


void delete_mem(CUdeviceptr devPtr) {
  if (list_head == NULL)
    init_list();

  int ptr_find = 0;
  
  pthread_mutex_lock(&g_map_mutex);

  for (struct MemRange *it = list_head->successor; it; it = it->successor) {
    if (it->devPtr == devPtr) {
      ptr_find = 1;
      g_used_memory -= it->count;
      list_delete(it);
      break;
    }
  }
  pthread_mutex_unlock(&g_map_mutex);
  
  if (ptr_find == 0) {
  }
}

static void activate_rate_watcher(CUdevice device) {
  pthread_t tid;

  pthread_create(&tid, NULL, rate_watcher, (void *)(uintptr_t)device);
  tgs_set_cpu_affinity(tid, g_gpu_id[device]);

#ifdef __APPLE__
  pthread_setname_np("rate_watcher");
#else
  pthread_setname_np(tid, "rate_watcher");
#endif
}

static void activate_limit_manager(CUdevice device) {
  pthread_t tid;

  pthread_create(&tid, NULL, limit_manager, (void *)(uintptr_t)device);
  tgs_set_cpu_affinity(tid, g_gpu_id[device]);

#ifdef __APPLE__
  pthread_setname_np("limit_manager");
#else
  pthread_setname_np(tid, "limit_manager");
#endif
}

static inline void low_priority_initialization(const CUdevice device) {
  
  g_active_gpu[device] = 1;
  cuDeviceGetUuid(&g_uuid[device], device);

//   CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetUuid, &g_uuid[device], device);
//   if (ret != CUDA_SUCCESS) {
//     // LOGGER(FATAL, "cuDeviceGetUuid error\n");
//   }

  int gpu_id = 0;
  for (int i = 0; i < 16; ++i) {
    gpu_id += (int)g_uuid[device].bytes[i];
  }
  gpu_id = (gpu_id % 8 + 8) % 8;
  g_gpu_id[device] = gpu_id;

//   ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxResetPersistingL2Cache);
//   if (ret != CUDA_SUCCESS) {
//     fprintf(stderr, "cuCtxResetPersistingL2Cache error\n");
//   }
//   ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxSetLimit, CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 0);
//   if (ret != CUDA_SUCCESS) {
//     fprintf(stderr, "cuCtxSetLimit error, ret=%d\n", (int)ret);
//   }

  activate_rate_watcher(device);
  activate_limit_manager(device);
}

static void init_rate_limit(long long initial_value, volatile long long *p_rate_limit, int *p_state) {
  *p_rate_limit = initial_value;
  *p_state = TGS_SLOW_START;
}

static inline long long min(long long a, long long b) {
  return a < b ? a : b;
}

static inline long long max(long long a, long long b) {
  return a > b ? a : b;
}

static const long long update_rate_limit(int *p_state, CUdevice device, double recv_rate, double max_rate, double *p_max_rate) {
  const static long long UPPER_LIMIT = 100000000000000LL;
  const double threshold = 0.03;
  static int sign = 0;
  double delta = (recv_rate - max_rate) / max_rate;
  delta = delta > 0 ? delta : -delta;

  long long rate_limit = g_rate_limit[device];

  switch (*p_state)
  {
  case TGS_SLOW_START:
    if (delta <= threshold) {
      rate_limit = min(rate_limit * 1.5 + 1, min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)(1ll << 40))));
    }
    else {
      rate_limit = rate_limit / 1.5;
      sign = -1;
      *p_state = TGS_CONGESTION_AVOIDANCE;
    }
    break;

  case TGS_CONGESTION_AVOIDANCE:
    if ((sign == -1 && delta <= threshold) || (sign == 1 && delta < threshold)) {
      rate_limit += max(max_rate * 0.00025, RATE_MIN);
      rate_limit = min(rate_limit, min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)65536LL * 65536LL)));
      sign = 1;
    }
    else {
      rate_limit -= max(rate_limit * 0.08, RATE_MIN);
      rate_limit = min(rate_limit, min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)65536LL * 65536LL)));
      sign = -1;
    }

    if (delta >= 3. * threshold) {
      *p_state = TGS_SLOW_START;
      rate_limit /= 10;
    }
    break;
  }

  static int max_diff_counter = 0;
  if (delta >= 0.12) {
    ++max_diff_counter;
    if (max_diff_counter >= 20) {
      *p_max_rate *= 0.8;
      max_diff_counter = 0;
      *p_state = TGS_SLOW_START;
      rate_limit /= 2;
    }
  }
  else {
    max_diff_counter = 0;
  }

  rate_limit = (rate_limit <= 0) ? 0 : rate_limit;

  g_rate_limit[device] = rate_limit;
  return rate_limit;
}

static void *limit_manager(void *v_device) {
  const CUdevice device = (uintptr_t)v_device;
  const int MAXLINE = 4096;
  const static long long UPPER_LIMIT = 100000000000000LL;
  const double alpha = 0.0;
//   int listenfd, connfd, ret;
//   socklen_t clientlen;
//   struct sockaddr_storage clientaddr;
//   char client_hostname[MAXLINE], client_port[MAXLINE];

//   listenfd = open_listenfd(device);
//   clientlen = sizeof(struct sockaddr_storage);

  
  while (1) {
    // if ((connfd = accept(listenfd, (struct sockaddr *)&clientaddr, &clientlen)) < 0)
    //   LOGGER(FATAL, "accept error\n");
    // if ((ret = getnameinfo((const struct sockaddr *)&clientaddr, clientlen, client_hostname, MAXLINE,
    //                        client_port, MAXLINE, 0)) != 0)
    //   LOGGER(FATAL, "getnameinfo error: %s\n", gai_strerror(ret));
    
    double max_rate = -1;
    max_rate = global_max_rate;
    // if (rio_readn(connfd, (void *)&max_rate, sizeof(double)) != sizeof(double)) {
    //   continue;
    // }

    fprintf(stderr, "low priority recv max_rate: %lf\n", max_rate);

    g_rate_limit[device] = 0;
    g_rate_control_flag[device] = 1;

    double recv_rate = 1.;
    long long cnt = 0;
    int state = -1;

    const int WINDOW_SIZE = 5;
    const int PREWARM_TIME = 0;
    const int PROFILE_TIME = 5;
    double rate_window[WINDOW_SIZE];
    int continue_flag = 0;

profile:
    for (int t = 1; t <= PREWARM_TIME + PROFILE_TIME || continue_flag; ++t) {
      double recv_counter = -1;
    //   ssize_t n = rio_readn(connfd, (void *)&recv_counter, sizeof(double));
    //   if (n != sizeof(double)) {
    //     break;
    //   }

        recv_counter = global_rate_counter;
        // fprintf(stderr, "low priority recv recv_counter: %lf\n", recv_counter);

      if (t <= PREWARM_TIME) {
        continue;
      }

      recv_counter = recv_counter >= 1. ? recv_counter : 1.;
      recv_rate = alpha * recv_rate + (1 - alpha) * recv_counter;
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, recv_rate);
      double max_delta = (max_window_rate - max_rate) / max_rate;

      if (max_delta >= -0.05 && max_delta <= 0.05) {
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
        // continue_flag = (abs(max_rate - max_window_rate) < 1e-5);
        continue_flag = false;
      }
      else {
        if (max_delta > 0.05) {
          double new_max_rate = max_window_rate * 0.975;
          max_rate = max_rate > new_max_rate ? max_rate : new_max_rate;
        }
        else {
          double new_max_rate = max_window_rate * 1.025;
          max_rate = max_rate < new_max_rate ? max_rate : new_max_rate;
        }
        continue_flag = 1;
      }
    }
    fprintf(stderr, "profile max rate: %lf\n", max_rate);

    while (1) {
      double recv_counter = -1;

        recv_counter = global_rate_counter;
        // fprintf(stderr, "low priority recv recv_counter: %lf\n", recv_counter);

    //   ssize_t n = rio_readn(connfd, (void *)&recv_counter, sizeof(double));
    //   if (n != sizeof(double)) {
    //     if (n)
    //     //   LOGGER(4, "readn error: receive %d byte\n", (int)n);
    //     break;
    //   }

      recv_counter = recv_counter >= 1. ? recv_counter : 1.;
      recv_rate = alpha * recv_rate + (1 - alpha) * recv_counter;
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, recv_rate);
      double max_delta = (max_window_rate - max_rate) / max_rate;
      
      if (max_delta >= -0.1 && max_delta <= 0.1)
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
      else if (max_delta > 0.2 || max_delta < -0.2) {
        if (max_delta > 0.2) {
          double new_max_rate = max_window_rate * 0.975;
          max_rate = max_rate > new_max_rate ? max_rate : new_max_rate;
        }
        else {
          double new_max_rate = max_window_rate * 1.025;
          max_rate = max_rate < new_max_rate ? max_rate : new_max_rate;
        }
        fprintf(stderr, "change max rate: %lf\n", max_rate);
        goto profile;
      }

      ++cnt;
      if (cnt == 1) {
        init_rate_limit(LIMIT_INITIALIZER, &g_rate_limit[device], &state);
        continue;
      }

      int num_zero = 0;
      for(int i = 0; i < WINDOW_SIZE; ++i){
        if(rate_window[i] < 1000)
          ++num_zero;
      }
      
      long long rate_limit;
      if(num_zero <= WINDOW_SIZE / 5 * 2 || cnt < 15){
        if(num_zero == 2 && cnt > 15){
          rate_limit = LIMIT_INITIALIZER;
          init_rate_limit(rate_limit, &g_rate_limit[device], &state);
        }
        else
          rate_limit = update_rate_limit(&state, device, recv_rate, (double)max_rate, &max_rate);
      }
      else{
        rate_limit = min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)65536LL * 65536LL));
        init_rate_limit(rate_limit, &g_rate_limit[device], &state);
      }

    }
    // if ((ret = close(connfd)) < 0)
    //   LOGGER(FATAL, "close error\n");

    g_rate_limit[device] = 0;
    // activate_memory_transfer_routine(device);
    g_rate_control_flag[device] = 0;
  }
}

static inline int launch_test(const long long kernel_size, const CUdevice device) {
  return g_rate_control_flag[device] == 1 && g_rate_counter[device] > g_rate_limit[device];
}


static inline bool rate_limiter(const long long kernel_size) {
  CUdevice device = 0;
  cuCtxGetDevice(&device);
//   const CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
//   if (ret != CUDA_SUCCESS) {
//     fprintf(stderr, "cuCtxGetDevice error\n");
//   }

  if (!g_active_gpu[device])
    low_priority_initialization(device);

  // if not ready, return false
  if (launch_test(kernel_size, device))
    return false;

  // if ready to launch, return true
  __sync_add_and_fetch_8(&g_rate_counter[device], kernel_size);
  return true;
}


static void *rate_watcher(void *v_device) {

  const CUdevice device = (uintptr_t)v_device;
  const unsigned long duration = 50;
  const struct timespec unit_time = {
    .tv_sec = duration / 1000,
    .tv_nsec = duration % 1000 * MILLISEC,
  };
  g_rate_counter[device] = 0;

  int log_count = 0;

  while (1) {
    nanosleep(&unit_time, NULL);
    
    long long current_rate = g_rate_counter[device];
    g_rate_counter[device] = 0;
    g_current_rate[device] = current_rate;

    log_count++;
    if (log_count == 20) {
      log_count = 0;
      fprintf(stderr, "low priority current_rate: %lld\n", current_rate);
    }

  }
  return NULL;
}

}

void TallyServer::run_tgs_scheduler()
{
    TALLY_SPD_LOG_ALWAYS("Running TGS scheduler ...");

    CudaLaunchConfig config = CudaLaunchConfig::default_config;
    KernelLaunchWrapper kernel_wrapper;

    while (!iox::posix::hasTerminationRequested()) {

        bool is_highest_priority = true;
        num_clients = client_data_all.size();

        for (auto &pair : client_data_all) {

            auto &client_data = pair.second;

            if (client_data.has_exit) {
                auto client_id = pair.first;
                client_data_all.erase(client_id);
                break;
            }

            if (!is_highest_priority) {

                auto kernel_wrapper_ptr = client_data.kernel_dispatch_queue.peek();
                if (!kernel_wrapper_ptr) {
                    continue;
                }

                auto launch_call = kernel_wrapper_ptr->launch_call;
                bool can_launch = LowPriorityTGS::rate_limiter(launch_call.num_blocks);
                if (!can_launch) {
                    continue;
                }
            }

            bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

            if (succeeded) {

                auto launch_call = kernel_wrapper.launch_call;
        
                if (is_highest_priority) {
                    HighPriorityTGS::rate_estimator(launch_call.num_blocks);
                }

                kernel_wrapper.kernel_to_dispatch(CudaLaunchConfig::default_config, nullptr, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);
                kernel_wrapper.free_args();
                client_data.queue_size--;
            }

            is_highest_priority = false;
        }
    }
}