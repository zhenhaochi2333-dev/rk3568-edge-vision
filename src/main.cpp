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
