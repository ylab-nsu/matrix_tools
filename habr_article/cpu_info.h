#ifndef HABR_ARTICLE_PROJECT_CPU_INFO_H
#define HABR_ARTICLE_PROJECT_CPU_INFO_H

#include <immintrin.h>
#include <vector>
#include <bitset>
#include <array>
#include <intrin.h>

size_t cache_size(int level);

class InstructionSet {
    // forward declarations
    class InstructionSet_Internal;

public:
    // getters
    static std::string Vendor() { return CPU_Rep.vendor_; }

    static std::string Brand() { return CPU_Rep.brand_; }

    static bool SSE3() { return CPU_Rep.f_1_ECX_[0]; }

    static bool PCLMULQDQ() { return CPU_Rep.f_1_ECX_[1]; }

    static bool MONITOR() { return CPU_Rep.f_1_ECX_[3]; }

    static bool SSSE3() { return CPU_Rep.f_1_ECX_[9]; }

    static bool FMA() { return CPU_Rep.f_1_ECX_[12]; }

    static bool CMPXCHG16B() { return CPU_Rep.f_1_ECX_[13]; }

    static bool SSE41() { return CPU_Rep.f_1_ECX_[19]; }

    static bool SSE42() { return CPU_Rep.f_1_ECX_[20]; }

    static bool MOVBE() { return CPU_Rep.f_1_ECX_[22]; }

    static bool POPCNT() { return CPU_Rep.f_1_ECX_[23]; }

    static bool AES() { return CPU_Rep.f_1_ECX_[25]; }

    static bool XSAVE() { return CPU_Rep.f_1_ECX_[26]; }

    static bool OSXSAVE() { return CPU_Rep.f_1_ECX_[27]; }

    static bool AVX() { return CPU_Rep.f_1_ECX_[28]; }

    static bool F16C() { return CPU_Rep.f_1_ECX_[29]; }

    static bool RDRAND() { return CPU_Rep.f_1_ECX_[30]; }

    static bool MSR() { return CPU_Rep.f_1_EDX_[5]; }

    static bool CX8() { return CPU_Rep.f_1_EDX_[8]; }

    static bool SEP() { return CPU_Rep.f_1_EDX_[11]; }

    static bool CMOV() { return CPU_Rep.f_1_EDX_[15]; }

    static bool CLFSH() { return CPU_Rep.f_1_EDX_[19]; }

    static bool MMX() { return CPU_Rep.f_1_EDX_[23]; }

    static bool FXSR() { return CPU_Rep.f_1_EDX_[24]; }

    static bool SSE() { return CPU_Rep.f_1_EDX_[25]; }

    static bool SSE2() { return CPU_Rep.f_1_EDX_[26]; }

    static bool FSGSBASE() { return CPU_Rep.f_7_EBX_[0]; }

    static bool BMI1() { return CPU_Rep.f_7_EBX_[3]; }

    static bool HLE() { return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[4]; }

    static bool AVX2() { return CPU_Rep.f_7_EBX_[5]; }

    static bool BMI2() { return CPU_Rep.f_7_EBX_[8]; }

    static bool ERMS() { return CPU_Rep.f_7_EBX_[9]; }

    static bool INVPCID() { return CPU_Rep.f_7_EBX_[10]; }

    static bool RTM() { return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[11]; }

    static bool AVX512F() { return CPU_Rep.f_7_EBX_[16]; }

    static bool RDSEED() { return CPU_Rep.f_7_EBX_[18]; }

    static bool ADX() { return CPU_Rep.f_7_EBX_[19]; }

    static bool AVX512PF() { return CPU_Rep.f_7_EBX_[26]; }

    static bool AVX512ER() { return CPU_Rep.f_7_EBX_[27]; }

    static bool AVX512CD() { return CPU_Rep.f_7_EBX_[28]; }

    static bool SHA() { return CPU_Rep.f_7_EBX_[29]; }

    static bool PREFETCHWT1() { return CPU_Rep.f_7_ECX_[0]; }

    static bool LAHF() { return CPU_Rep.f_81_ECX_[0]; }

    static bool LZCNT() { return CPU_Rep.isIntel_ && CPU_Rep.f_81_ECX_[5]; }

    static bool ABM() { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[5]; }

    static bool SSE4a() { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[6]; }

    static bool XOP() { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[11]; }

    static bool TBM() { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[21]; }

    static bool SYSCALL() { return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[11]; }

    static bool MMXEXT() { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[22]; }

    static bool RDTSCP() { return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[27]; }

    static bool _3DNOWEXT() { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[30]; }

    static bool _3DNOW() { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[31]; }

private:
    static const InstructionSet_Internal CPU_Rep;

    class InstructionSet_Internal {
    public:
        InstructionSet_Internal()
                : nIds_{0},
                  nExIds_{0},
                  isIntel_{false},
                  isAMD_{false},
                  f_1_ECX_{0},
                  f_1_EDX_{0},
                  f_7_EBX_{0},
                  f_7_ECX_{0},
                  f_81_ECX_{0},
                  f_81_EDX_{0},
                  data_{},
                  extdata_{} {
            //int cpuInfo[4] = {-1};
            std::array<int, 4> cpui{};

            // Calling __cpuid with 0x0 as the function_id argument
            // gets the number of the highest valid function ID.
            __cpuid(cpui.data(), 0);
            nIds_ = cpui[0];

            for (int i = 0; i <= nIds_; ++i) {
                __cpuidex(cpui.data(), i, 0);
                data_.push_back(cpui);
            }

            // Capture vendor string
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int *>(vendor) = data_[0][1];
            *reinterpret_cast<int *>(vendor + 4) = data_[0][3];
            *reinterpret_cast<int *>(vendor + 8) = data_[0][2];
            vendor_ = vendor;
            if (vendor_ == "GenuineIntel") {
                isIntel_ = true;
            } else if (vendor_ == "AuthenticAMD") {
                isAMD_ = true;
            }

            // load bitset with flags for function 0x00000001
            if (nIds_ >= 1) {
                f_1_ECX_ = data_[1][2];
                f_1_EDX_ = data_[1][3];
            }

            // load bitset with flags for function 0x00000007
            if (nIds_ >= 7) {
                f_7_EBX_ = data_[7][1];
                f_7_ECX_ = data_[7][2];
            }

            // Calling __cpuid with 0x80000000 as the function_id argument
            // gets the number of the highest valid extended ID.
            __cpuid(cpui.data(), 0x80000000);
            nExIds_ = cpui[0];

            char brand[0x40];
            memset(brand, 0, sizeof(brand));

            for (int i = 0x80000000; i <= nExIds_; ++i) {
                __cpuidex(cpui.data(), i, 0);
                extdata_.push_back(cpui);
            }

            // load bitset with flags for function 0x80000001
            if (nExIds_ >= 0x80000001) {
                f_81_ECX_ = extdata_[1][2];
                f_81_EDX_ = extdata_[1][3];
            }

            // Interpret CPU brand string if reported
            if (nExIds_ >= 0x80000004) {
                memcpy(brand, extdata_[2].data(), sizeof(cpui));
                memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
                memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
                brand_ = brand;
            }
        };

        int nIds_;
        int nExIds_;
        std::string vendor_;
        std::string brand_;
        bool isIntel_;
        bool isAMD_;
        std::bitset<32> f_1_ECX_;
        std::bitset<32> f_1_EDX_;
        std::bitset<32> f_7_EBX_;
        std::bitset<32> f_7_ECX_;
        std::bitset<32> f_81_ECX_;
        std::bitset<32> f_81_EDX_;
        std::vector<std::array<int, 4>> data_;
        std::vector<std::array<int, 4>> extdata_;
    };
};

#endif //HABR_ARTICLE_PROJECT_CPU_INFO_H
