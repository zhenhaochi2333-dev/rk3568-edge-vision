#pragma once

#include "edgevision/app_options.hpp"

namespace edgevision {

// 把命令行字符串转成已校验的 AppOptions；帮助文本与解析规则在此配对。
// 解析本身不打开模型、摄像头或网络监听。
class CliParser {
public:
    static CliParseResult parse(int argc, char** argv);
    static const char* usage();
};

}  // namespace edgevision
