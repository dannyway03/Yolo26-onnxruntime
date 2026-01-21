#ifndef SCV_GLOBAL_METRICS_HPP
#define SCV_GLOBAL_METRICS_HPP

#include <chrono>

#include <atomic>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Platform-specific includes for system monitoring
#if defined(__linux__)
  #include <dirent.h>
  #include <sys/types.h>
  #include <unistd.h>
#endif

namespace scv {

/**
 * @brief Unified global metrics system for timing and counting
 *
 * Provides a single source of truth for all performance metrics in the application.
 * Supports both timing measurements and event counting with thread-safe operations.
 *
 * Features:
 * - Named timers with automatic averaging
 * - Atomic counters for events
 * - RAII scoped timing
 * - Thread-safe operations
 * - Low overhead when not actively queried
 */
class GlobalMetrics
{
public:
  /**
   * @brief Get singleton instance
   */
  static GlobalMetrics&
  instance();

  /**
   * @brief Timer statistics
   */
  struct TimerStats
  {
    std::atomic<uint64_t> totalTimeNs{0};
    std::atomic<uint64_t> callCount{0};
    std::atomic<uint64_t> minTimeNs{UINT64_MAX};
    std::atomic<uint64_t> maxTimeNs{0};
    // Note: lastEndTime and startTime are protected by _timersMutex, not atomic
    std::chrono::steady_clock::time_point lastEndTime{};
    std::chrono::steady_clock::time_point startTime{}; // For manual timers
    std::atomic<bool> isRunning{false};                // Track if timer is currently running

    [[nodiscard]] double
    getAvgMs() const
    {
      auto count = callCount.load();
      return count > 0 ? (static_cast<double>(totalTimeNs.load()) / count) / 1e6 : 0.0;
    }

    [[nodiscard]] double
    getMinMs() const
    {
      auto minNs = minTimeNs.load();
      return minNs != UINT64_MAX ? static_cast<double>(minNs) / 1e6 : 0.0;
    }

    [[nodiscard]] double
    getMaxMs() const
    {
      return static_cast<double>(maxTimeNs.load()) / 1e6;
    }

    [[nodiscard]] uint64_t
    getCount() const
    {
      return callCount.load();
    }

    void
    reset()
    {
      totalTimeNs.store(0);
      callCount.store(0);
      minTimeNs.store(UINT64_MAX);
      maxTimeNs.store(0);
      lastEndTime = std::chrono::steady_clock::time_point{};
      startTime = std::chrono::steady_clock::time_point{};
      isRunning.store(false);
    }
  };

  /**
   * @brief RAII scoped timer for automatic timing
   */
  class ScopedTimer
  {
  public:
    explicit ScopedTimer(const std::string& tName) : _name(tName), _start(std::chrono::steady_clock::now()) {}

    ~ScopedTimer()
    {
      _end = std::chrono::steady_clock::now();
      auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(_end - _start).count();
      GlobalMetrics::instance().recordTiming(_name, static_cast<uint64_t>(durationNs), _end);
    }

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer&
    operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer&
    operator=(ScopedTimer&&) = delete;

    std::chrono::steady_clock::time_point
    getEndTime()
    {
      return _end;
    }

    uint64_t
    stopAndGo()
    {
      _end = std::chrono::steady_clock::now();
      auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(_end - _start).count();
      GlobalMetrics::instance().recordTiming(_name, static_cast<uint64_t>(durationNs), _end);

      // restart
      _start = std::chrono::steady_clock::now();
      return static_cast<uint64_t>(durationNs);
    }

  private:
    std::string _name;
    std::chrono::steady_clock::time_point _start, _end;
  };

  // === Timer Interface ===

  /**
   * @brief Record timing for a named timer
   * @param name Timer name
   * @param durationNs Duration in nanoseconds
   * @param endTime End timestamp of the timer
   */
  void
  recordTiming(const std::string& tName, uint64_t tDurationNs, std::chrono::steady_clock::time_point tEndTime = {});

  /**
   * @brief Get average timing in milliseconds
   * @param name Timer name
   * @return Average time in milliseconds, 0.0 if timer doesn't exist
   */
  [[nodiscard]] double
  getAvgTime(const std::string& tName) const;

  /**
   * @brief Get minimum timing in milliseconds
   * @param name Timer name
   * @return Minimum time in milliseconds, 0.0 if timer doesn't exist
   */
  [[nodiscard]] double
  getMinTime(const std::string& tName) const;

  /**
   * @brief Get maximum timing in milliseconds
   * @param name Timer name
   * @return Maximum time in milliseconds, 0.0 if timer doesn't exist
   */
  [[nodiscard]] double
  getMaxTime(const std::string& tName) const;

  /**
   * @brief Get timer call count
   * @param name Timer name
   * @return Number of times timer was called, 0 if timer doesn't exist
   */
  [[nodiscard]] uint64_t
  getTimerCount(const std::string& tName) const;

  /**
   * @brief Get the last end time of a named timer
   * @param name Timer name
   * @return Last end timestamp, default time_point if timer doesn't exist or hasn't run
   */
  [[nodiscard]] std::chrono::steady_clock::time_point
  getLastEndTime(const std::string& tName) const;

  /**
   * @brief Get the last end time of a named timer
   * @param name Timer name
   * @return Last end timestamp, default time_point if timer doesn't exist or hasn't run
   */
  [[nodiscard]] std::chrono::steady_clock::time_point
  getStartTime(const std::string& tName) const;

  // === Manual Timer Interface ===

  /**
   * @brief Start a manual timer
   * @param name Timer name
   * @param startTime Optional start time (default: now)
   */
  void
  startTimer(const std::string& tName, std::chrono::steady_clock::time_point tStartTime = {});

  /**
   * @brief Stop a manual timer and record the elapsed time
   * @param name Timer name
   * @param endTime Optional end time (default: now)
   * @return Elapsed time in milliseconds, 0.0 if timer wasn't running
   */
  double
  stopTimer(const std::string& tName, std::chrono::steady_clock::time_point tEndTime = {});

  /**
   * @brief Get elapsed time for a running timer without stopping it
   * @param name Timer name
   * @param currentTime Optional current time (default: now)
   * @return Elapsed time in milliseconds, 0.0 if timer isn't running
   */
  [[nodiscard]] double
  getElapsedTime(const std::string& tName, std::chrono::steady_clock::time_point tCurrentTime = {}) const;

  /**
   * @brief Check if a timer is currently running
   * @param name Timer name
   * @return True if timer is running, false otherwise
   */
  [[nodiscard]] bool
  isTimerRunning(const std::string& tName) const;

  /**
   * @brief Restart a timer (stop if running, then start again)
   * @param name Timer name
   * @param startTime Optional start time (default: now)
   */
  void
  restartTimer(const std::string& tName, std::chrono::steady_clock::time_point tStartTime = {});

  // === Counter Interface ===

  /**
   * @brief Increment a named counter
   * @param name Counter name
   * @param value Value to add (default: 1)
   */
  void
  increment(const std::string& tName, uint64_t tValue = 1);

  /**
   * @brief Set counter value
   * @param name Counter name
   * @param value New value
   */
  void
  setCounter(const std::string& tName, uint64_t tValue);

  /**
   * @brief Get counter value
   * @param name Counter name
   * @return Counter value, 0 if counter doesn't exist
   */
  [[nodiscard]] uint64_t
  getCounter(const std::string& tName) const;

  // === Management Interface ===

  /**
   * @brief Reset a specific timer
   * @param name Timer name
   */
  void
  resetTimer(const std::string& tName);

  /**
   * @brief Reset a specific counter
   * @param name Counter name
   */
  void
  resetCounter(const std::string& tName);

  /**
   * @brief Reset all metrics
   */
  void
  resetAll();

  /**
   * @brief Print all metrics to console
   */
  void
  printReport() const;

  /**
   * @brief Get formatted report string
   */
  [[nodiscard]] std::string
  getReport() const;

  // === System Monitoring Interface (compile-time controlled) ===

// Compile-time control via CMake (follows profiler.hpp pattern)
#ifdef SCV_SYSTEM_MONITORING_ENABLED
  #if SCV_SYSTEM_MONITORING_ENABLED
  static constexpr bool kEnableSystemMonitoring = true;
  #else
  static constexpr bool kEnableSystemMonitoring = false;
  #endif
#else
  // Default to disabled for zero overhead in production
  static constexpr bool kEnableSystemMonitoring = false;
#endif

  /**
   * @brief System monitoring sample data
   */
  struct SystemSample
  {
    double timeMs;     // Relative to monitoring start
    int activeThreads; // Number of active threads
    double cpuCores;   // Normalized CPU load (0.0 to N cores)
  };

  /**
   * @brief Start system-wide performance sampling
   * @param intervalMs Sampling interval in milliseconds (default: 10.0)
   * @param outputPath Optional output file path for samples
   */
  void
  startSystemMonitoring(double tIntervalMs = 10.0, const std::string& tOutputPath = "");

  /**
   * @brief Stop system monitoring and optionally flush to file
   * @param flushToFile Whether to write samples to file (default: true)
   */
  void
  stopSystemMonitoring(bool tFlushToFile = true);

  /**
   * @brief Get current system monitoring samples
   * @return Vector of system samples, empty if monitoring not enabled
   */
  [[nodiscard]] std::vector<SystemSample>
  getSystemSamples() const;

  /**
   * @brief Get current system CPU load percentage
   * @return Current CPU load as percentage (0.0-100.0 per core)
   */
  [[nodiscard]] double
  getCurrentCpuLoad() const;

  /**
   * @brief Get current active thread count
   * @return Number of active threads
   */
  [[nodiscard]] int
  getCurrentActiveThreads() const;

  /**
   * @brief Write system monitoring data to pipe time log format
   * @param os Output stream to write to
   * @param intervalSec Interval since last measurement for CPU calculation
   */
  void
  logSystemStatsToStream(std::ostream& tOs, double tIntervalSec) const;

  /**
   * @brief Check if system monitoring is currently running
   * @return True if monitoring is active
   */
  [[nodiscard]] bool
  isSystemMonitoringActive() const;

private:
  GlobalMetrics() = default;

  mutable std::mutex _timersMutex;
  mutable std::mutex _countersMutex;

  std::unordered_map<std::string, TimerStats> _timers;
  std::unordered_map<std::string, std::atomic<uint64_t>> _counters;

  // === System Monitoring Private Members ===
#if defined(SCV_SYSTEM_MONITORING_ENABLED) && SCV_SYSTEM_MONITORING_ENABLED
  // System monitoring state (only compiled when enabled)
  mutable std::mutex _systemMonitoringMutex;
  std::thread _monitoringThread;
  std::atomic<bool> _monitoringActive{false};
  std::atomic<uint64_t> _monitoringStartNs{0};
  std::vector<SystemSample> _systemSamples;
  std::string _outputPath;

  // Linux-specific CPU monitoring state
  #if defined(__linux__)
  thread_local static std::unordered_map<int, long> _lastCpuTimes;
  static long _ticksPerSec;
  #endif

  // Platform-specific implementation methods
  void
  _systemMonitoringLoop(double intervalMs);
  std::unordered_map<int, long>
  _getThreadCpuStats() const;
  int
  _getActiveThreadCount() const;
  double
  _calculateCpuLoad(double intervalSec) const;
#endif
};

} // namespace scv

// === Convenience Macros ===

/**
 * @brief Create a scoped timer for the current scope
 * Usage: SCOPED_TIMER("functionName");
 */
#define SCOPED_TIMER(name) scv::GlobalMetrics::ScopedTimer _timer(name)

/**
 * @brief Increment a named counter by 1
 * Usage: COUNT_EVENT("framesProcessed");
 */
#define COUNT_EVENT(name) scv::GlobalMetrics::instance().increment(name)

/**
 * @brief Increment a named counter by a specific value
 * Usage: COUNT_EVENT_VALUE("bytesProcessed", frameSize);
 */
#define COUNT_EVENT_VALUE(name, value) scv::GlobalMetrics::instance().increment(name, value)

/**
 * @brief Set a counter to a specific value
 * Usage: SET_COUNTER("queueSize", queue.size());
 */
#define SET_COUNTER(name, value) scv::GlobalMetrics::instance().setCounter(name, value)

/**
 * @brief Start a manual timer
 * Usage: START_TIMER("fpsTimer");
 */
#define START_TIMER(name) scv::GlobalMetrics::instance().startTimer(name)

/**
 * @brief Stop a manual timer and record elapsed time
 * Usage: double elapsed = STOP_TIMER("fpsTimer");
 */
#define STOP_TIMER(name) scv::GlobalMetrics::instance().stopTimer(name)

/**
 * @brief Get elapsed time for a running timer without stopping it
 * Usage: double elapsed = GET_ELAPSED_TIME("fpsTimer");
 */
#define GET_ELAPSED_TIME(name) scv::GlobalMetrics::instance().getElapsedTime(name)

/**
 * @brief Restart a timer (stop if running, then start again)
 * Usage: RESTART_TIMER("fpsTimer");
 */
#define RESTART_TIMER(name) scv::GlobalMetrics::instance().restartTimer(name)

// === System Monitoring Convenience Macros ===

#ifdef SCV_SYSTEM_MONITORING_ENABLED
  #if SCV_SYSTEM_MONITORING_ENABLED
    /**
     * @brief Start system monitoring with specified interval
     * Usage: START_SYSTEM_MONITORING(5.0);
     */
    #define START_SYSTEM_MONITORING(intervalMs) scv::GlobalMetrics::instance().startSystemMonitoring(intervalMs)

    /**
     * @brief Start system monitoring with output path
     * Usage: START_SYSTEM_MONITORING_FILE(10.0, "pytools/system_stats.dat");
     */
    #define START_SYSTEM_MONITORING_FILE(intervalMs, path)                                                             \
      scv::GlobalMetrics::instance().startSystemMonitoring(intervalMs, path)

    /**
     * @brief Stop system monitoring and flush to file
     * Usage: STOP_SYSTEM_MONITORING();
     */
    #define STOP_SYSTEM_MONITORING() scv::GlobalMetrics::instance().stopSystemMonitoring()

    /**
     * @brief Log system stats to stream in pipe time format
     * Usage: LOG_SYSTEM_STATS_TO_STREAM(std::cout, 0.01);
     */
    #define LOG_SYSTEM_STATS_TO_STREAM(stream, intervalSec)                                                            \
      scv::GlobalMetrics::instance().logSystemStatsToStream(stream, intervalSec)

    /**
     * @brief Enhanced PIPE_STOP macro with system stats logging
     * Preserves all original functionality from pipeline demos
     */
    #define PIPE_STOP_WITH_SYSTEM_STATS(name, id, stream)                                                              \
      do                                                                                                               \
      {                                                                                                                \
        static thread_local auto lastTp = std::chrono::high_resolution_clock::now();                                   \
        auto nowTp = std::chrono::high_resolution_clock::now();                                                        \
        double intervalSec = std::chrono::duration<double>(nowTp - lastTp).count();                                    \
        lastTp = nowTp;                                                                                                \
        double stop = STOP_TIMER(name);                                                                                \
        auto startTp = scv::GlobalMetrics::instance().getStartTime(name);                                              \
        uint64_t startNs = std::chrono::duration_cast<std::chrono::nanoseconds>(startTp.time_since_epoch()).count();   \
        static uint64_t referenceTimeNs = 0;                                                                           \
        if (referenceTimeNs == 0)                                                                                      \
          referenceTimeNs = startNs;                                                                                   \
        double startMs = static_cast<double>(startNs - referenceTimeNs) / 1e6;                                         \
        if (startMs < 3000)                                                                                            \
        {                                                                                                              \
          stream << name << " " << id << " " << startMs << " " << startMs + stop << " ";                               \
          LOG_SYSTEM_STATS_TO_STREAM(stream, intervalSec);                                                             \
          stream << "\n";                                                                                              \
        }                                                                                                              \
      }                                                                                                                \
      while (0)
  #else
  // Compile to no-ops when system monitoring is disabled
    #define START_SYSTEM_MONITORING(intervalMs)                                                                        \
      do                                                                                                               \
      {}                                                                                                               \
      while (0)
    #define START_SYSTEM_MONITORING_FILE(intervalMs, path)                                                             \
      do                                                                                                               \
      {}                                                                                                               \
      while (0)
    #define STOP_SYSTEM_MONITORING()                                                                                   \
      do                                                                                                               \
      {}                                                                                                               \
      while (0)
    #define LOG_SYSTEM_STATS_TO_STREAM(stream, intervalSec)                                                            \
      do                                                                                                               \
      {}                                                                                                               \
      while (0)
    #define PIPE_STOP_WITH_SYSTEM_STATS(name, id, stream)                                                              \
      do                                                                                                               \
      {}                                                                                                               \
      while (0)
  #endif
#else
  // Default to no-ops when macro not defined (backward compatibility)
  #define START_SYSTEM_MONITORING(intervalMs)                                                                          \
    do                                                                                                                 \
    {}                                                                                                                 \
    while (0)
  #define START_SYSTEM_MONITORING_FILE(intervalMs, path)                                                               \
    do                                                                                                                 \
    {}                                                                                                                 \
    while (0)
  #define STOP_SYSTEM_MONITORING()                                                                                     \
    do                                                                                                                 \
    {}                                                                                                                 \
    while (0)
  #define LOG_SYSTEM_STATS_TO_STREAM(stream, intervalSec)                                                              \
    do                                                                                                                 \
    {}                                                                                                                 \
    while (0)
  #define PIPE_STOP_WITH_SYSTEM_STATS(name, id, stream)                                                                \
    do                                                                                                                 \
    {}                                                                                                                 \
    while (0)
#endif

#endif // SCV_GLOBAL_METRICS_HPP