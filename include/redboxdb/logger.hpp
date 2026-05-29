#pragma once
#include <spdlog/spdlog.h>

namespace Log {
    inline void info (const std::string& msg) { spdlog::info (msg); }
    inline void warn (const std::string& msg) { spdlog::warn (msg); }
    inline void error(const std::string& msg) { spdlog::error(msg); }

}