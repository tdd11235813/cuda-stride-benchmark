#ifndef APP_HELPER_HPP
#define APP_HELPER_HPP

#include <ctime>
#include <cstring>
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

inline
void print_header_matrix(const char* app_name, unsigned n1, unsigned n2) {

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
            << ",dev_cores"
            << ",dev_GFLOPs_FMA"
            << ",nx"
            << ",n"
            << ",numSMs"
            << ",blocks_i"
            << ",blocks_i/numSMs"
            << ",blocks_n"
            << ",blocks_i.x"
            << ",blocks_n.x"
            << ",TBlocksizeX"
            << ",TBlocksize"
            << ",TTilewidthX"
            << ",TRuns"
            << ",min_time"
            << ",max_throughput"
            << ",max_flops"
            << "\n"
    ;

}

#endif /* APP_HELPER_HPP */
