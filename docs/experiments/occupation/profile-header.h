#pragma once
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
namespace occupation_profile {
struct Totals {
 std::map<std::string, double> seconds;
 ~Totals() { if (auto path=std::getenv("OCCUPATION_PROFILE")) { std::ofstream out(path); out << "{"; bool first=true; for(auto &[name,value]:seconds) { if(!first) out << ","; first=false; out << "\"" << name << "\":" << value; } out << "}"; } }
};
inline Totals totals;
struct Region {
 std::string name;
 std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
 Region(const char *label):name(label){}
 ~Region() { totals.seconds[name] += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); }
};
}
