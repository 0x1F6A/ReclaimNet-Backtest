#include "rnbt/job.hpp"

int main(int argc, char **argv) { return rnbt::job::run({argv, static_cast<std::size_t>(argc)}); }
