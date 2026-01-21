#include "global_metrics.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace scv {

GlobalMetrics&
GlobalMetrics::instance()
{
  static GlobalMetrics instance;
  return instance;
}

void
GlobalMetrics::recordTiming(const std::string& tName, uint64_t tDurationNs,
                            std::chrono::steady_clock::time_point tEndTime)
{
  std::lock_guard<std::mutex> lock(_timersMutex);

  auto& stats = _timers[tName];
  stats.totalTimeNs.fetch_add(tDurationNs);
  stats.callCount.fetch_add(1);

  // Store the end time (thread-safe since we have the mutex)
  if (tEndTime != std::chrono::steady_clock::time_point{})
  {
    stats.lastEndTime = tEndTime;
  }

  // Update min/max atomically
  uint64_t currentMin = stats.minTimeNs.load();
  while (tDurationNs < currentMin && !stats.minTimeNs.compare_exchange_weak(currentMin, tDurationNs))
  {
    // Keep trying until we successfully update or find a smaller value
  }

  uint64_t currentMax = stats.maxTimeNs.load();
  while (tDurationNs > currentMax && !stats.maxTimeNs.compare_exchange_weak(currentMax, tDurationNs))
  {
    // Keep trying until we successfully update or find a larger value
  }
}

double
GlobalMetrics::getAvgTime(const std::string& tName) const
{
  std::lock_guard<std::mutex> lock(_timersMutex);
  auto it = _timers.find(tName);
  return it != _timers.end() ? it->second.getAvgMs() : 0.0;
}

double
GlobalMetrics::getMinTime(const std::string& tName) const
{
  std::lock_guard<std::mutex> lock(_timersMutex);
  auto it = _timers.find(tName);
  return it != _timers.end() ? it->second.getMinMs() : 0.0;
}

double
GlobalMetrics::getMaxTime(const std::string& tName) const
{
  std::lock_guard<std::mutex> lock(_timersMutex);
  auto it = _timers.find(tName);
  return it != _timers.end() ? it->second.getMaxMs() : 0.0;
}

uint64_t
GlobalMetrics::getTimerCount(const std::string& tName) const
{
  std::lock_guard<std::mutex> lock(_timersMutex);
  auto it = _timers.find(tName);
  return it != _timers.end() ? it->second.getCount() : 0;
}

std::chrono::steady_clock::time_point
GlobalMetrics::getLastEndTime(const std::string& tName) const
{
  std::lock_guard<std::mutex> lock(_timersMutex);
  auto it = _timers.find(tName);
  return it != _timers.end() ? it->second.lastEndTime : std::chrono::steady_clock::time_point{};
}

std::chrono::steady_clock::time_point
GlobalMetrics::getStartTime(const std::string& tName) const
{
  std::lock_guard<std::mutex> lock(_timersMutex);
  auto it = _timers.find(tName);
  return it != _timers.end() ? it->second.startTime : std::chrono::steady_clock::time_point{};
}

void
GlobalMetrics::startTimer(const std::string& tName, std::chrono::steady_clock::time_point tStartTime)
{
  std::lock_guard<std::mutex> lock(_timersMutex);

  auto& stats = _timers[tName];
  stats.startTime =
    (tStartTime != std::chrono::steady_clock::time_point{}) ? tStartTime : std::chrono::steady_clock::now();
  stats.isRunning.store(true);
}

double
GlobalMetrics::stopTimer(const std::string& tName, std::chrono::steady_clock::time_point tEndTime)
{
  std::lock_guard<std::mutex> lock(_timersMutex);

  auto it = _timers.find(tName);
  if (it == _timers.end() || !it->second.isRunning.load())
  {
    return 0.0; // Timer doesn't exist or isn't running
  }

  auto& stats = it->second;
  auto endTime = (tEndTime != std::chrono::steady_clock::time_point{}) ? tEndTime : std::chrono::steady_clock::now();
  auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - stats.startTime).count();

  // Record timing using existing infrastructure
  stats.totalTimeNs.fetch_add(static_cast<uint64_t>(durationNs));
  stats.callCount.fetch_add(1);
  stats.lastEndTime = endTime;
  stats.isRunning.store(false);

  // Update min/max atomically
  uint64_t currentMin = stats.minTimeNs.load();
  while (static_cast<uint64_t>(durationNs) < currentMin && // NOLINT
         !stats.minTimeNs.compare_exchange_weak(currentMin, static_cast<uint64_t>(durationNs)))
  {
    // Keep trying until we successfully update or find a smaller value
  }

  uint64_t currentMax = stats.maxTimeNs.load();
  while (static_cast<uint64_t>(durationNs) > currentMax && // NOLINT
         !stats.maxTimeNs.compare_exchange_weak(currentMax, static_cast<uint64_t>(durationNs)))
  {
    // Keep trying until we successfully update or find a larger value
  }

  return static_cast<double>(durationNs) / 1e6; // Convert to milliseconds
}

double
GlobalMetrics::getElapsedTime(const std::string& tName, std::chrono::steady_clock::time_point tCurrentTime) const
{
  std::lock_guard<std::mutex> lock(_timersMutex);

  auto it = _timers.find(tName);
  if (it == _timers.end() || !it->second.isRunning.load())
  {
    return 0.0; // Timer doesn't exist or isn't running
  }

  auto currentTime =
    (tCurrentTime != std::chrono::steady_clock::time_point{}) ? tCurrentTime : std::chrono::steady_clock::now();
  auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - it->second.startTime).count();

  return static_cast<double>(durationNs) / 1e6; // Convert to milliseconds
}

bool
GlobalMetrics::isTimerRunning(const std::string& tName) const
{
  std::lock_guard<std::mutex> lock(_timersMutex);

  auto it = _timers.find(tName);
  return it != _timers.end() && it->second.isRunning.load();
}

void
GlobalMetrics::restartTimer(const std::string& tName, std::chrono::steady_clock::time_point tStartTime)
{
  std::lock_guard<std::mutex> lock(_timersMutex);

  auto& stats = _timers[tName];
  // If it was running, we don't record the previous timing - just restart
  stats.startTime =
    (tStartTime != std::chrono::steady_clock::time_point{}) ? tStartTime : std::chrono::steady_clock::now();
  stats.isRunning.store(true);
}

void
GlobalMetrics::increment(const std::string& tName, uint64_t tValue)
{
  std::lock_guard<std::mutex> lock(_countersMutex);
  _counters[tName].fetch_add(tValue);
}

void
GlobalMetrics::setCounter(const std::string& tName, uint64_t tValue)
{
  std::lock_guard<std::mutex> lock(_countersMutex);
  _counters[tName].store(tValue);
}

uint64_t
GlobalMetrics::getCounter(const std::string& tName) const
{
  std::lock_guard<std::mutex> lock(_countersMutex);
  auto it = _counters.find(tName);
  return it != _counters.end() ? it->second.load() : 0;
}

void
GlobalMetrics::resetTimer(const std::string& tName)
{
  std::lock_guard<std::mutex> lock(_timersMutex);
  auto it = _timers.find(tName);
  if (it != _timers.end())
  {
    it->second.reset();
  }
}

void
GlobalMetrics::resetCounter(const std::string& tName)
{
  std::lock_guard<std::mutex> lock(_countersMutex);
  auto it = _counters.find(tName);
  if (it != _counters.end())
  {
    it->second.store(0);
  }
}

void
GlobalMetrics::resetAll()
{
  {
    std::lock_guard<std::mutex> lock(_timersMutex);
    for (auto& [name, stats] : _timers)
    {
      stats.reset();
    }
  }

  {
    std::lock_guard<std::mutex> lock(_countersMutex);
    for (auto& [name, counter] : _counters)
    {
      counter.store(0);
    }
  }
}

void
GlobalMetrics::printReport() const
{
  std::cout << getReport() << std::endl;
}

std::string
GlobalMetrics::getReport() const
{
  std::ostringstream report;

  report << "\n=== Global Metrics Report ===\n";
  report << std::string(71, '=') << "\n";

  // Timers section
  {
    std::lock_guard<std::mutex> lock(_timersMutex);
    if (!_timers.empty())
    {
      report << "TIMERS:\n";
      report << std::left << std::setw(25) << "Name" << std::setw(10) << "Count" << std::setw(12) << "Avg (ms)"
             << std::setw(12) << "Min (ms)" << std::setw(12) << "Max (ms)" << "\n";
      report << std::string(71, '=') << "\n";

      // Sort timers by name for consistent output
      std::vector<std::pair<std::string, const TimerStats*>> sortedTimers;
      for (const auto& [name, stats] : _timers)
      {
        sortedTimers.emplace_back(name, &stats);
      }
      std::sort(sortedTimers.begin(), sortedTimers.end());

      for (const auto& [name, stats] : sortedTimers)
      {
        report << std::left << std::setw(25) << name << std::setw(10) << stats->getCount() << std::setw(12)
               << std::fixed << std::setprecision(3) << stats->getAvgMs() << std::setw(12) << std::fixed
               << std::setprecision(3) << stats->getMinMs() << std::setw(12) << std::fixed << std::setprecision(3)
               << stats->getMaxMs() << "\n";
      }
      report << std::string(71, '=') << "\n";
    }
  }

  // Counters section
  {
    std::lock_guard<std::mutex> lock(_countersMutex);
    if (!_counters.empty())
    {
      report << "COUNTERS:\n";
      report << std::left << std::setw(30) << "Name" << std::setw(15) << "Value" << "\n";
      report << std::string(45, '=') << "\n";

      // Sort counters by name for consistent output
      std::vector<std::pair<std::string, uint64_t>> sortedCounters;
      for (const auto& [name, counter] : _counters)
      {
        sortedCounters.emplace_back(name, counter.load());
      }
      std::sort(sortedCounters.begin(), sortedCounters.end());

      for (const auto& [name, value] : sortedCounters)
      {
        report << std::left << std::setw(30) << name << std::setw(15) << value << "\n";
      }
      report << std::string(45, '=') << "\n";
    }
  }

  if (_timers.empty() && _counters.empty())
  {
    report << "No metrics recorded.\n\n";
  }
  return report.str();
}

// === System Monitoring Implementation ===

#if defined(SCV_SYSTEM_MONITORING_ENABLED) && SCV_SYSTEM_MONITORING_ENABLED

  // Static member definitions
  #if defined(__linux__)
thread_local std::unordered_map<int, long> GlobalMetrics::_lastCpuTimes;
long GlobalMetrics::_ticksPerSec = sysconf(_SC_CLK_TCK);
  #endif

void
GlobalMetrics::startSystemMonitoring(double tIntervalMs, const std::string& tOutputPath)
{
  if constexpr (!kEnableSystemMonitoring)
  {
    return; // Compile-time optimization - dead code elimination
  }

  std::lock_guard<std::mutex> lock(_systemMonitoringMutex);

  if (_monitoringActive.load())
  {
    return; // Already running
  }

  _outputPath = tOutputPath;
  _systemSamples.clear();
  _systemSamples.reserve(1000); // Pre-allocate for performance

  // Set reference time
  auto now = std::chrono::high_resolution_clock::now();
  _monitoringStartNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

  _monitoringActive = true;
  _monitoringThread = std::thread(&GlobalMetrics::_systemMonitoringLoop, this, tIntervalMs);
}

void
GlobalMetrics::stopSystemMonitoring(bool tFlushToFile)
{
  if constexpr (!kEnableSystemMonitoring)
  {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(_systemMonitoringMutex);
    if (!_monitoringActive.load())
    {
      return; // Not running
    }
    _monitoringActive = false;
  }

  // Wait for monitoring thread to finish
  if (_monitoringThread.joinable())
  {
    _monitoringThread.join();
  }

  // Flush to file if requested
  if (tFlushToFile && !_outputPath.empty())
  {
    std::lock_guard<std::mutex> lock(_systemMonitoringMutex);
    std::ofstream cpuFileStats(_outputPath, std::ios::out | std::ios::trunc);
    for (const auto& sample : _systemSamples)
    {
      if (sample.timeMs < 3000) // Same filter as original pipeline code
      {
        cpuFileStats << sample.timeMs << " " << sample.activeThreads << " " << sample.cpuCores << "\n";
      }
    }
  }
}

std::vector<GlobalMetrics::SystemSample>
GlobalMetrics::getSystemSamples() const
{
  if constexpr (!kEnableSystemMonitoring)
  {
    return {};
  }

  std::lock_guard<std::mutex> lock(_systemMonitoringMutex);
  return _systemSamples; // Return copy
}

double
GlobalMetrics::getCurrentCpuLoad() const
{
  if constexpr (!kEnableSystemMonitoring)
  {
    return 0.0;
  }

  return _calculateCpuLoad(0.01); // 10ms default interval
}

int
GlobalMetrics::getCurrentActiveThreads() const
{
  if constexpr (!kEnableSystemMonitoring)
  {
    return 0;
  }

  return _getActiveThreadCount();
}

void
GlobalMetrics::logSystemStatsToStream(std::ostream& tOs, double tIntervalSec) const
{
  if constexpr (!kEnableSystemMonitoring)
  {
    return;
  }

  #if defined(__linux__)
  auto currCpuStats = _getThreadCpuStats();
  int activeThreads = 0;
  double totalCpuLoad = 0.0;

  for (const auto& [tid, currTime] : currCpuStats)
  {
    long prevTime = 0;
    if (_lastCpuTimes.count(tid))
    {
      prevTime = _lastCpuTimes.at(tid);
    }

    double deltaSec = double(currTime - prevTime) / _ticksPerSec;
    double cpuLoad = (tIntervalSec > 0.0) ? (deltaSec / tIntervalSec) * 100.0 : 0.0;
    totalCpuLoad += cpuLoad;
    if (cpuLoad > 0.0)
    {
      activeThreads++;
    }
    _lastCpuTimes[tid] = currTime;
  }

  tOs << "ActiveThreads " << activeThreads << " TotalCPU% " << totalCpuLoad;
  #endif
}

bool
GlobalMetrics::isSystemMonitoringActive() const
{
  if constexpr (!kEnableSystemMonitoring)
  {
    return false;
  }

  return _monitoringActive.load();
}

// Private implementation methods

void
GlobalMetrics::_systemMonitoringLoop(double intervalMs)
{
  while (_monitoringActive.load())
  {
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    double timeMs = static_cast<double>(nowNs - _monitoringStartNs.load()) / 1e6;
    int threads = _getActiveThreadCount();
    double cpu = _calculateCpuLoad(intervalMs / 1000.0) / 100.0; // Normalize to 0.0-N.0 cores

    {
      std::lock_guard<std::mutex> lock(_systemMonitoringMutex);
      _systemSamples.push_back({timeMs, threads, cpu});
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(intervalMs)));
  }
}

std::unordered_map<int, long>
GlobalMetrics::_getThreadCpuStats() const
{
  #if defined(__linux__)
  std::unordered_map<int, long> cpuTimes;
  pid_t pid = getpid();
  std::string taskDir = "/proc/" + std::to_string(pid) + "/task";
  DIR* dir = opendir(taskDir.c_str());
  if (!dir)
    return cpuTimes;

  dirent* entry;
  while ((entry = readdir(dir)) != nullptr)
  {
    if (entry->d_name[0] == '.')
      continue;
    int tid = std::stoi(entry->d_name);
    std::ifstream f(taskDir + "/" + entry->d_name + "/stat");
    if (!f.is_open())
      continue;
    std::string line, tmp;
    std::getline(f, line);
    std::istringstream ss(line);
    long utime = 0, stime = 0;
    for (int i = 1; i <= 15; ++i)
    {
      ss >> tmp;
      if (i == 14)
        utime = std::stol(tmp);
      if (i == 15)
        stime = std::stol(tmp);
    }
    cpuTimes[tid] = utime + stime;
  }
  closedir(dir);
  return cpuTimes;
  #else
  return {};
  #endif
}

int
GlobalMetrics::_getActiveThreadCount() const
{
  return static_cast<int>(_getThreadCpuStats().size());
}

double
GlobalMetrics::_calculateCpuLoad(double tIntervalSec) const
{
  #if defined(__linux__)
  auto curr = _getThreadCpuStats();
  double total = 0;
  for (const auto& [tid, currTime] : curr)
  {
    long prevTime = 0;
    if (_lastCpuTimes.count(tid))
    {
      prevTime = _lastCpuTimes.at(tid);
    }
    double deltaSec = double(currTime - prevTime) / _ticksPerSec;
    double cpuLoad = tIntervalSec > 0.0 ? (deltaSec / tIntervalSec) * 100.0 : 0.0;
    total += cpuLoad;
    _lastCpuTimes[tid] = currTime;
  }
  return total;
  #else
  return 0.0;
  #endif
}

#endif // SCV_SYSTEM_MONITORING_ENABLED

} // namespace scv
