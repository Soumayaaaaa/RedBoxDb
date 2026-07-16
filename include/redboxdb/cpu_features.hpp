#pragma once

#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <cpuid.h>
#endif

namespace Platform {
    inline bool has_avx2() {
#ifdef _MSC_VER
        int info[4] = { 0, 0, 0, 0 };
        __cpuid(info, 0);
        if (info[0] < 7) return false;
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0;
#else
        unsigned int info[4] = { 0, 0, 0, 0 };
        if (!__get_cpuid(0, &info[0], &info[1], &info[2], &info[3])) return false;
        if (info[0] < 7) return false;
        __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
        return (info[1] & (1 << 5)) != 0;
#endif
    }
}
