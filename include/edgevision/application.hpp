#pragma once

#include "edgevision/core_types.hpp"

namespace edgevision {

// 应用入口：校验选项和模型/标签，选择文件或实时采集路径，建立检测器、
// 可选 TCP 事件服务与画面输出，并在退出时统一释放相关资源。
int run_application(const AppOptions& options);

}  // namespace edgevision
