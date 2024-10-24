#include "cpu_info.h"

#if defined(__APPLE__)

#include <sys/sysctl.h>
size_t cache_size(int level) {
    size_t cache_size = 0;
    size_t sizeof_cache_size = sizeof(cache_size);
    if (level == 1) {
        sysctlbyname("hw.l1dcachesize", &cache_size, &sizeof_cache_size, 0, 0);
    } else if (level == 2) {
        sysctlbyname("hw.l2cachesize", &cache_size, &sizeof_cache_size, 0, 0);
    } else if (level == 3) {
        sysctlbyname("hw.l3cachesize", &cache_size, &sizeof_cache_size, 0, 0);
    }
    return line_size;
}

#elif defined(_WIN64)

#include <cstdlib>
#include <windows.h>

size_t cache_size(int level) {
    size_t line_size = 0;
    DWORD buffer_size = 0;
    DWORD i = 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buffer = 0;

    GetLogicalProcessorInformation(0, &buffer_size);
    buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *) malloc(buffer_size);
    GetLogicalProcessorInformation(&buffer[0], &buffer_size);

    for (i = 0; i != buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i) {
        if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == level) {
            line_size = buffer[i].Cache.Size;
            break;
        }
    }

    free(buffer);
    return line_size;
}

#else

#include <unistd.h>
size_t cache_size(int level) {
    if (level == 1) {
        return sysconf(_SC_LEVEL1_DCACHE_SIZE);
    }
    if (level == 2) {
        return sysconf(_SC_LEVEL2_CACHE_SIZE);
    }
    if (level == 3) {
        return sysconf(_SC_LEVEL3_CACHE_SIZE);
    }
}

#endif



