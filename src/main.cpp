// main 是命令行与应用层的边界：解析参数后将 AppOptions 交给 run_application。
// 模型加载、输入模式选择和资源生命周期均在应用层完成；这里统一把异常
// 转成错误日志和非零退出码，避免各输入路径各自处理顶层退出语义。
#include "edgevision/application.hpp"
#include "edgevision/cli_parser.hpp"
#include "edgevision/logger.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv)
{
    try {
        const edgevision::CliParseResult parsed = edgevision::CliParser::parse(argc, argv);
        if (parsed.show_help) {
            std::cout << edgevision::CliParser::usage() << '\n';
            return 0;
        }
        return edgevision::run_application(parsed.options);
    } catch (const std::exception& error) {
        edgevision::log_error(error.what());
        return 1;
    }
}
