#include <ctime>
#include <iostream>

inline
void print_header(const char* app_name, unsigned n1, unsigned n2) {

  std::time_t now = std::time(nullptr);
  std::cout << "; "<< app_name
            << " " << n1
            << " " << n2
            << ", " << strtok(ctime(&now), "\n")
            << "\n";

  std::cout << "index"
            << ",dev_id"
            << ",dev_name"
            << ",dev_CC"
            << ",dev_memClock"
            << ",dev_clock"
            << ",n"
            << ",numSMs"
            << ",blocks_i"
            << ",blocks_i/numSMs"
            << ",blocks_n"
            << ",TBlocksize"
            << ",TRuns"
            << ",min_time"
            << ",max_throughput"
            << "\n"
    ;

}
