#include <cstdio>

#include <nlopt11.hpp>

int main(int argc, char *argv[]) {

  std::printf("nlopt v%d.%d.%d\n", nlopt::version_major(), nlopt::version_minor(),
              nlopt::version_bugfix());
  for (std::size_t i = 0; i < nlopt::NUM_ALGORITHMS; ++i) {
    std::printf("%4zd: %s\n", i, nlopt::algorithm_name(static_cast<nlopt::algorithm>(i)));
  }

  return 0;
}