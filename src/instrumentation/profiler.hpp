#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Platform-specific includes for CPU time
#if defined(__linux__)
  #include <ctime>
#elif defined(_WIN32)
  #include <windows.h>
#endif

namespace scv {

// Forward declaration
class Profiler;

// RAII class to automatically time a scope
class ScopeProfiler
{
public:
  ScopeProfiler(const char* tName);
  ~ScopeProfiler();

  // Disable copy/move to prevent unintended behavior
  ScopeProfiler(const ScopeProfiler&) = delete;
  ScopeProfiler&
  operator=(const ScopeProfiler&) = delete;
  ScopeProfiler(ScopeProfiler&&) = delete;
  ScopeProfiler&
  operator=(ScopeProfiler&&) = delete;

private:
  const char* _name;
  std::chrono::steady_clock::time_point _wallStart;
  std::chrono::nanoseconds _cpuStart{};
};

class Profiler
{
public:
  // Aggregated statistics for a single scope
  struct ScopeStats
  {
    uint64_t calls = 0;
    uint64_t threads = 0;
    double totalWallMs = 0;
    double selfWallMs = 0;
    double totalCpuMs = 0;
    double selfCpuMs = 0;
    std::chrono::steady_clock::time_point firstCallTime;
  };

  // Raw data collected for each scope entry per thread
  struct Entry
  {
    uint64_t calls = 0;
    double totalWallMs = 0;
    double totalCpuMs = 0;
    std::chrono::steady_clock::time_point firstCallTime;

    // Incremental tracking for real-time sampling
    uint64_t lastReportedCalls = 0;
    double lastReportedWallMs = 0;
    double lastReportedCpuMs = 0;
  };

  // Data stored per thread
  struct ThreadData
  {
    std::unordered_map<std::string, Entry> entries;
    std::vector<std::string> callStack;
  };

  // Incremental stats tracking for real-time sampling
  struct IncrementalStats
  {
    ScopeStats currentStats = {};
    bool isValid = false;
    uint64_t lastUpdateVersion = 0;
  };

  // Report node structure (needed by GlobalData)
  struct ReportNode
  {
    std::string name;
    ScopeStats stats;
    double childrenWallMs = 0;
    double childrenCpuMs = 0;
    std::vector<std::unique_ptr<ReportNode>> children;    // Preserve call order
    std::unordered_map<std::string, size_t> childIndices; // For quick lookup
  };

  // Template for version-based cache (consolidates cache patterns)
  template<typename T>
  struct VersionedCache
  {
    mutable std::unordered_map<std::string, T> data;
    mutable std::mutex mutex;
    mutable uint64_t version = 0;

    bool
    isValid(uint64_t tCurrentVersion) const
    {
      return version == tCurrentVersion;
    }

    void
    updateVersion(uint64_t tCurrentVersion) const
    {
      version = tCurrentVersion;
    }

    void
    clear() const
    {
      std::lock_guard lock(mutex);
      data.clear();
      version = 0;
    }
  };

  // Global data shared across all threads
  struct GlobalData
  {
    mutable std::shared_mutex dataMutex; // Protects thread_data
    std::unordered_map<std::thread::id, std::unique_ptr<ThreadData>> threadData;

    // Unified version counter for all caches
    mutable std::atomic<uint64_t> dataVersion{0}; // Incremented when profiler data changes

    // Consolidated cache systems using template
    mutable VersionedCache<ScopeStats> statsCache;
    mutable VersionedCache<IncrementalStats> incrementalCache;

    // Call graph cache (different type, kept separate but follows same pattern)
    mutable std::unique_ptr<ReportNode> callGraphCache;
    mutable uint64_t callGraphVersion = 0;
    mutable std::mutex callGraphMutex;

    ThreadData&
    getThreadDataForCurrentThread()
    {
      // Caller must hold data_mutex lock
      std::thread::id tid = std::this_thread::get_id();
      auto it = threadData.find(tid);
      if (it == threadData.end())
      {
        it = threadData.emplace(tid, std::make_unique<ThreadData>()).first;
      }
      return *it->second;
    }

    void
    invalidateCache() const
    {
      dataVersion.fetch_add(1, std::memory_order_relaxed);
    }
  };

  static GlobalData&
  getGlobalData()
  {
    // Meyer's singleton is thread-safe in C++11 and later
    static GlobalData instance;
    return instance;
  }

  // CPU time collection: Compile-time control via CMake
#ifdef SCV_CPU_TIME_ENABLED
  #if SCV_CPU_TIME_ENABLED
  // CMake ON: CPU timing enabled, can be controlled at runtime
  inline static bool kEnableCpuTime = true; // Default on when CMake enables it
  #else
  // CMake DEFAULT: CPU timing compile-time disabled (zero overhead)
  static constexpr bool kEnableCpuTime = false;
  #endif
#else
  // Backward compatibility - default to runtime control when macro not defined
  inline static bool kEnableCpuTime = false;
#endif

  // Profiling control: Compile-time control via CMake
#ifdef SCV_PROFILING_ENABLED
  #if SCV_PROFILING_ENABLED
  // CMake ON: Profiling enabled
  static constexpr bool kEnableProfiling = true;
  #else
  // CMake OFF: Profiling compile-time disabled (zero overhead)
  static constexpr bool kEnableProfiling = false;
  #endif
#else
  // CMake undefined: default to enabled for backward compatibility
  static constexpr bool kEnableProfiling = true;
#endif

  // Called by ScopeProfiler to record data
  static void
  record([[maybe_unused]] const char* tName, std::chrono::steady_clock::time_point tWallStart,
         std::chrono::nanoseconds tWallTime, [[maybe_unused]] std::chrono::nanoseconds tCpuTime)
  {
    // Check if profiling is enabled at runtime
    if (!kEnableProfiling)
    {
      return;
    }

    GlobalData& global = getGlobalData();

    // Use a single lock for the entire record operation to avoid deadlocks
    std::unique_lock lock(global.dataMutex);
    ThreadData& td = global.getThreadDataForCurrentThread();

    // Use thread-local string buffer to avoid repeated allocations
    thread_local static std::string cachedFullName;
    buildHierarchicalName(td.callStack, cachedFullName);

    if (cachedFullName.empty())
      return;

    Entry& entry = td.entries[cachedFullName];
    if (entry.calls == 0)
    {
      entry.firstCallTime = tWallStart;
    }
    entry.calls++;
    entry.totalWallMs += tWallTime.count() / 1e6;

    // CPU time collection: compile-time + runtime checks for zero overhead when disabled
#ifdef SCV_CPU_TIME_ENABLED
  #if SCV_CPU_TIME_ENABLED
    if (kEnableCpuTime)
    {
      entry.totalCpuMs += tCpuTime.count() / 1e6;
    }
  #endif
#endif

    // Invalidate stats cache since data has changed (atomic operation, no lock needed)
    global.invalidateCache();
  }

private:
  // Utility to extract base name from hierarchical name (consolidates duplicate logic)
  static std::string_view
  getBaseName(std::string_view tFullName)
  {
    size_t lastSep = tFullName.rfind("->");
    return (lastSep == std::string::npos) ? tFullName : tFullName.substr(lastSep + 2);
  }

  // Unified stats aggregation function (consolidates duplicate logic from multiple places)
  static void
  aggregateStatsFromAllThreads(const GlobalData& tGlobal, std::unordered_map<std::string, ScopeStats>& tAggregatedStats,
                               std::unordered_map<std::string, std::unordered_set<std::thread::id>>& tHreadsPerScope)
  {

    // Use shared lock for read-only operation
    std::shared_lock dataLock(tGlobal.dataMutex);

    for (const auto& [tid, thread_data] : tGlobal.threadData)
    {
      for (const auto& [full_name, entry] : thread_data->entries)
      {
        ScopeStats& stats = tAggregatedStats[full_name];
        if (entry.calls > 0 && (stats.calls == 0 || entry.firstCallTime < stats.firstCallTime))
        {
          stats.firstCallTime = entry.firstCallTime;
        }
        stats.calls += entry.calls;
        stats.totalWallMs += entry.totalWallMs;
        stats.totalCpuMs += entry.totalCpuMs;
        tHreadsPerScope[full_name].insert(tid);
      }
    }

    // Set thread counts
    for (auto& [name, stats] : tAggregatedStats)
    {
      stats.threads = tHreadsPerScope[name].size();
    }
  }

  // Optimized hierarchical name building with pre-allocated buffer
  static void
  buildHierarchicalName(const std::vector<std::string>& tCallStack, std::string& tOutput)
  {
    tOutput.clear();
    if (tCallStack.empty())
      return;

    // Pre-calculate required size to avoid multiple reallocations
    size_t totalSize = tCallStack[0].length();
    for (size_t i = 1; i < tCallStack.size(); ++i)
    {
      totalSize += 2 + tCallStack[i].length(); // "->" + name
    }

    tOutput.reserve(totalSize);
    tOutput = tCallStack[0];
    for (size_t i = 1; i < tCallStack.size(); ++i)
    {
      tOutput += "->";
      tOutput += tCallStack[i];
    }
  }

public:
  // Get cached call graph, rebuilding only if needed (Task #6 optimization)
  static std::shared_ptr<ReportNode>
  getCachedCallGraph()
  {
    GlobalData& global = getGlobalData();
    std::lock_guard graphLock(global.callGraphMutex);

    uint64_t currentVersion = global.dataVersion.load(std::memory_order_relaxed);

    // Check if we need to rebuild the call graph
    if (!global.callGraphCache || global.callGraphVersion != currentVersion)
    {
      // Need to rebuild - acquire data lock
      std::shared_lock dataLock(global.dataMutex);
      global.callGraphCache = buildReportTreeInternal();
      global.callGraphVersion = currentVersion;
    }

    // Return shared copy of the cached graph
    return {global.callGraphCache.get(), [](ReportNode*)
            {
              // Custom deleter that does nothing - the original unique_ptr manages the
              // memory
            }};
  }

  // Helper to build the call tree from collected data (internal implementation)
  static std::unique_ptr<ReportNode>
  buildReportTreeInternal()
  {
    auto root = std::make_unique<ReportNode>();
    root->name = "[root]";

    std::unordered_map<std::string, ScopeStats> aggregatedStats;
    std::unordered_map<std::string, std::unordered_set<std::thread::id>> threadsPerScope;
    GlobalData& global = getGlobalData();

    // 1. Aggregate stats from all threads
    for (const auto& [tid, thread_data] : global.threadData)
    {
      for (const auto& [full_name, entry] : thread_data->entries)
      {
        ScopeStats& stats = aggregatedStats[full_name];
        if (entry.calls > 0 && (stats.calls == 0 || entry.firstCallTime < stats.firstCallTime))
        {
          stats.firstCallTime = entry.firstCallTime;
        }
        stats.calls += entry.calls;
        stats.totalWallMs += entry.totalWallMs;
        stats.totalCpuMs += entry.totalCpuMs;
        threadsPerScope[full_name].insert(tid);
      }
    }

    auto split = [](const std::string& tS, const std::string& tDelimiter)
    {
      std::vector<std::string> tokens;
      size_t start = 0, end = 0;
      while ((end = tS.find(tDelimiter, start)) != std::string::npos)
      {
        tokens.push_back(tS.substr(start, end - start));
        start = end + tDelimiter.length();
      }
      tokens.push_back(tS.substr(start));
      return tokens;
    };

    // 2. Build tree from aggregated stats, preserving call order
    // First sort by first_call_time to maintain call order
    std::vector<std::pair<std::string, ScopeStats>> sortedStats;
    for (const auto& [full_name, stats] : aggregatedStats)
    {
      sortedStats.emplace_back(full_name, stats);
    }
    std::sort(sortedStats.begin(), sortedStats.end(),
              [](const auto& tA, const auto& tB)
              {
                return tA.second.firstCallTime < tB.second.firstCallTime;
              });

    for (const auto& [full_name, stats] : sortedStats)
    {
      ReportNode* currentNodePtr = root.get();
      for (const auto& part : split(full_name, "->"))
      {
        // Check if child already exists
        auto it = currentNodePtr->childIndices.find(part);
        if (it == currentNodePtr->childIndices.end())
        {
          // Create new child and add to vector
          auto childNode = std::make_unique<ReportNode>();
          childNode->name = part;
          size_t index = currentNodePtr->children.size();
          currentNodePtr->children.push_back(std::move(childNode));
          currentNodePtr->childIndices[part] = index;
          currentNodePtr = currentNodePtr->children[index].get();
        }
        else
        {
          // Use existing child
          currentNodePtr = currentNodePtr->children[it->second].get();
        }
      }
      currentNodePtr->stats = stats;
      if (threadsPerScope.count(full_name))
      {
        currentNodePtr->stats.threads = threadsPerScope.at(full_name).size();
      }
    }

    // 3. Post-order traversal to calculate self-time
    std::function<void(ReportNode*)> calculateSelfTime = [&](ReportNode* tNode)
    {
      double childrenWallTime = 0;
      double childrenCpuTime = 0;
      for (auto& child : tNode->children)
      {
        calculateSelfTime(child.get());
        childrenWallTime += child->stats.totalWallMs;
        childrenCpuTime += child->stats.totalCpuMs;
      }
      tNode->childrenWallMs = childrenWallTime;
      tNode->stats.selfWallMs = tNode->stats.totalWallMs - childrenWallTime;
      tNode->childrenCpuMs = childrenCpuTime;
      tNode->stats.selfCpuMs = tNode->stats.totalCpuMs - childrenCpuTime;
    };
    calculateSelfTime(root.get());

    return root;
  }

  // Backwards-compatible wrapper for buildReportTree (uses caching internally)
  static std::unique_ptr<ReportNode>
  buildReportTree()
  {
    // Use the cached version if available, otherwise build fresh
    GlobalData& global = getGlobalData();

    {
      std::lock_guard graphLock(global.callGraphMutex);
      uint64_t currentVersion = global.dataVersion.load(std::memory_order_relaxed);

      if (global.callGraphCache && global.callGraphVersion == currentVersion)
      {
        // Return the cached version (move ownership to caller)
        auto result = std::move(global.callGraphCache);
        global.callGraphCache = nullptr; // Mark as consumed
        return result;
      }
    }

    // Build fresh (lock acquired inside buildReportTree_Internal)
    std::shared_lock dataLock(global.dataMutex);
    return buildReportTreeInternal();
  }

private:
  // Update incremental stats for a specific scope
  static void
  updateIncrementalStats(const std::string& tScopeName, GlobalData& tGlobal)
  {
    std::lock_guard incLock(tGlobal.incrementalCache.mutex);
    uint64_t currentVersion = tGlobal.dataVersion.load(std::memory_order_relaxed);

    auto& incStats = tGlobal.incrementalCache.data[tScopeName];

    // Check if update needed
    if (incStats.isValid && incStats.lastUpdateVersion == currentVersion)
    {
      return; // Already up to date
    }

    // Compute fresh stats for this scope using optimized base name extraction
    ScopeStats freshStats = {};
    std::unordered_set<std::thread::id> threadsForScope;

    {
      std::shared_lock dataLock(tGlobal.dataMutex);
      for (const auto& [tid, thread_data] : tGlobal.threadData)
      {
        for (const auto& [full_name, entry] : thread_data->entries)
        {
          // Use consolidated getBaseName utility
          if (getBaseName(full_name) == tScopeName)
          {
            if (freshStats.calls == 0 || entry.firstCallTime < freshStats.firstCallTime)
            {
              freshStats.firstCallTime = entry.firstCallTime;
            }
            freshStats.calls += entry.calls;
            freshStats.totalWallMs += entry.totalWallMs;
            freshStats.totalCpuMs += entry.totalCpuMs;
            threadsForScope.insert(tid);
          }
        }
      }
    }

    freshStats.threads = threadsForScope.size();
    // Note: self_wall_ms and self_cpu_ms would need call graph computation for accuracy
    // For real-time sampling, we can approximate or skip this expensive calculation
    freshStats.selfWallMs = freshStats.totalWallMs; // Approximation
    freshStats.selfCpuMs = freshStats.totalCpuMs;   // Approximation

    incStats.currentStats = freshStats;
    incStats.isValid = true;
    incStats.lastUpdateVersion = currentVersion;
  }

public:
  static ScopeStats
  getStatsFast(std::string_view tName)
  {
    GlobalData& global = getGlobalData();
    std::string nameKey(tName);
    uint64_t currentVersion = global.dataVersion.load(std::memory_order_relaxed);

    // Try incremental cache first (fastest path for real-time sampling)
    {
      std::lock_guard incLock(global.incrementalCache.mutex);
      if (global.incrementalCache.isValid(currentVersion))
      {
        auto it = global.incrementalCache.data.find(nameKey);
        if (it != global.incrementalCache.data.end() && it->second.isValid)
        {
          return it->second.currentStats;
        }
      }
    }

    // Update incremental stats if needed
    updateIncrementalStats(nameKey, global);

    // Return updated incremental stats
    {
      std::lock_guard incLock(global.incrementalCache.mutex);
      auto it = global.incrementalCache.data.find(nameKey);
      if (it != global.incrementalCache.data.end() && it->second.isValid)
      {
        return it->second.currentStats;
      }
    }

    // Fallback to stats cache
    {
      std::lock_guard cacheLock(global.statsCache.mutex);
      if (global.statsCache.isValid(currentVersion))
      {
        auto cacheIt = global.statsCache.data.find(nameKey);
        if (cacheIt != global.statsCache.data.end())
        {
          return cacheIt->second;
        }
      }
      else
      {
        global.statsCache.clear();
      }
    }

    // Aggregate stats using unified function (eliminates code duplication)
    std::unordered_map<std::string, ScopeStats> aggregatedStats;
    std::unordered_map<std::string, std::unordered_set<std::thread::id>> threadsPerScope;
    aggregateStatsFromAllThreads(global, aggregatedStats, threadsPerScope);

    // 2. Calculate children time and self time - optimized with pre-allocated containers
    std::unordered_map<std::string, double> childrenWallTime;
    std::unordered_map<std::string, double> childrenCpuTime;
    childrenWallTime.reserve(aggregatedStats.size() / 2); // Estimate parent count
    childrenCpuTime.reserve(aggregatedStats.size() / 2);

    for (const auto& [full_name, stats] : aggregatedStats)
    {
      size_t lastSep = full_name.rfind("->");
      if (lastSep != std::string::npos)
      {
        // Use string_view to avoid substring allocation
        std::string_view parentName(full_name.data(), lastSep);
        std::string parentKey(parentName); // Only allocate when needed for map key
        childrenWallTime[parentKey] += stats.totalWallMs;
        childrenCpuTime[parentKey] += stats.totalCpuMs;
      }
    }

    for (auto& [full_name, stats] : aggregatedStats)
    {
      auto itWall = childrenWallTime.find(full_name);
      auto itCpu = childrenCpuTime.find(full_name);
      stats.selfWallMs = stats.totalWallMs - (itWall != childrenWallTime.end() ? itWall->second : 0.0);
      stats.selfCpuMs = stats.totalCpuMs - (itCpu != childrenCpuTime.end() ? itCpu->second : 0.0);
    }

    // 3. Cache all computed stats (populate cache for future lookups)
    if (!global.statsCache.isValid(currentVersion))
    {
      // Cache by base names for faster lookups
      std::unordered_map<std::string, ScopeStats> baseNameStats;
      std::unordered_map<std::string, std::unordered_set<std::thread::id>> baseNameThreads;

      for (const auto& [full_name, stats] : aggregatedStats)
      {
        const size_t kFullLen = full_name.length();
        size_t startPos = full_name.rfind("->");
        std::string_view baseNameView = (startPos == std::string::npos) ?
                                          std::string_view(full_name) :
                                          std::string_view(full_name.data() + startPos + 2, kFullLen - startPos - 2);

        std::string baseKey(baseNameView);
        baseNameStats[baseKey].calls += stats.calls;
        baseNameStats[baseKey].totalWallMs += stats.totalWallMs;
        baseNameStats[baseKey].selfWallMs += stats.selfWallMs;
        baseNameStats[baseKey].totalCpuMs += stats.totalCpuMs;
        baseNameStats[baseKey].selfCpuMs += stats.selfCpuMs;

        if (threadsPerScope.count(full_name))
        {
          baseNameThreads[baseKey].insert(threadsPerScope.at(full_name).begin(), threadsPerScope.at(full_name).end());
        }
      }

      // Store in cache using consolidated template system
      {
        std::lock_guard cacheLock(global.statsCache.mutex);
        global.statsCache.data.clear();
        global.statsCache.data.reserve(baseNameStats.size());

        for (auto& [base_key, stats] : baseNameStats)
        {
          stats.threads = baseNameThreads[base_key].size();
          global.statsCache.data[base_key] = stats;
        }

        global.statsCache.updateVersion(currentVersion);
      }
    }

    // 4. Return cached result
    {
      std::lock_guard cacheLock(global.statsCache.mutex);
      auto cacheIt = global.statsCache.data.find(nameKey);
      return (cacheIt != global.statsCache.data.end()) ? cacheIt->second : ScopeStats{};
    }
  }

  static ScopeStats
  getStats(std::string_view tName)
  {
    // Task #6: Use cached call graph to avoid rebuilding every time
    auto tree = getCachedCallGraph();

    ScopeStats finalStats = {};
    std::unordered_set<std::thread::id> threadSet;

    for (const auto& [tid, thread_data] : getGlobalData().threadData)
    {
      for (const auto& [full_name, entry] : thread_data->entries)
      {
        size_t lastSep = full_name.rfind("->");
        std::string baseName = (lastSep == std::string::npos) ? full_name : full_name.substr(lastSep + 2);
        if (baseName == tName)
        {
          threadSet.insert(tid);
        }
      }
    }
    finalStats.threads = threadSet.size();

    std::function<void(ReportNode*)> findAndAggregate = [&](ReportNode* tNode)
    {
      if (tNode->name == tName)
      {
        finalStats.calls += tNode->stats.calls;
        finalStats.totalWallMs += tNode->stats.totalWallMs;
        finalStats.selfWallMs += tNode->stats.selfWallMs;
        finalStats.totalCpuMs += tNode->stats.totalCpuMs;
        finalStats.selfCpuMs += tNode->stats.selfCpuMs;
      }
      for (const auto& childNode : tNode->children)
      {
        findAndAggregate(childNode.get());
      }
    };

    findAndAggregate(tree.get());
    return finalStats;
  }

  // Ultra-fast version for real-time sampling (e.g. frame-rate monitoring)
  static double
  getAvgStatRealtime(std::string_view tName, std::string_view tStatName)
  {
    GlobalData& global = getGlobalData();
    std::string nameKey(tName);

    // Check incremental stats with minimal locking
    {
      std::lock_guard incLock(global.incrementalCache.mutex);
      uint64_t currentVersion = global.dataVersion.load(std::memory_order_relaxed);
      auto it = global.incrementalCache.data.find(nameKey);

      if (it != global.incrementalCache.data.end() && it->second.isValid &&
          it->second.lastUpdateVersion == currentVersion)
      {
        // Use cached incremental stats
        const auto& stats = it->second.currentStats;
        if (stats.calls == 0)
          return 0.0;

        if (tStatName == "Wall")
          return stats.totalWallMs / stats.calls;
        else if (tStatName == "CPU")
          return stats.totalCpuMs / stats.calls;
        else if (tStatName == "SelfWall")
          return stats.selfWallMs / stats.calls;
        else if (tStatName == "SelfCPU")
          return stats.selfCpuMs / stats.calls;
      }
    }

    // If incremental stats not available, update them
    updateIncrementalStats(nameKey, global);

    // Try again with updated incremental stats
    {
      std::lock_guard incLock(global.incrementalCache.mutex);
      auto it = global.incrementalCache.data.find(nameKey);
      if (it != global.incrementalCache.data.end() && it->second.isValid)
      {
        const auto& stats = it->second.currentStats;
        if (stats.calls == 0)
          return 0.0;

        if (tStatName == "Wall")
          return stats.totalWallMs / stats.calls;
        else if (tStatName == "CPU")
          return stats.totalCpuMs / stats.calls;
        else if (tStatName == "SelfWall")
          return stats.selfWallMs / stats.calls;
        else if (tStatName == "SelfCPU")
          return stats.selfCpuMs / stats.calls;
      }
    }

    return 0.0; // Default if no data available
  }

  static double
  getAvgStat(std::string_view tName, std::string_view tStatName)
  {
    // For backward compatibility, use real-time optimized version
    return getAvgStatRealtime(tName, tStatName);
  }

  static void
  printFlatReport(std::ostream& tOs = std::cout)
  {
    // Task #6: Use cached call graph
    auto tree = getCachedCallGraph();

    std::unordered_map<std::string, ScopeStats> flatStats;
    std::unordered_map<std::string, std::unordered_set<std::thread::id>> threadsPerFunc;

    for (const auto& [tid, thread_data] : getGlobalData().threadData)
    {
      for (const auto& [full_name, entry] : thread_data->entries)
      {
        size_t lastSep = full_name.rfind("->");
        // Use string_view to avoid allocation, only create string when needed as map key
        std::string_view baseNameView =
          (lastSep == std::string::npos) ?
            std::string_view(full_name) :
            std::string_view(full_name.data() + lastSep + 2, full_name.length() - lastSep - 2);
        std::string baseName(baseNameView);
        threadsPerFunc[baseName].insert(tid);
      }
    }

    std::function<void(ReportNode*)> aggregateFlat = [&](ReportNode* tNode)
    {
      if (tNode->stats.calls > 0)
      {
        ScopeStats& stats = flatStats[tNode->name];
        stats.calls += tNode->stats.calls;
        stats.totalWallMs += tNode->stats.totalWallMs;
        stats.selfWallMs += tNode->stats.selfWallMs;
        stats.totalCpuMs += tNode->stats.totalCpuMs;
        stats.selfCpuMs += tNode->stats.selfCpuMs;
      }
      for (const auto& child : tNode->children)
      {
        aggregateFlat(child.get());
      }
    };
    aggregateFlat(tree.get());

    std::vector<std::pair<std::string, ScopeStats>> sortedStats(flatStats.begin(), flatStats.end());
    std::sort(sortedStats.begin(), sortedStats.end(),
              [](const auto& tA, const auto& tB)
              {
                return tA.second.totalWallMs > tB.second.totalWallMs;
              });

    tOs << "\n--- Profiling Report ---\n";
    tOs << std::string(151, '=') << '\n';
    tOs << std::left << std::setw(28) << "Function" << std::right << std::setw(8) << "Threads" << std::setw(10)
        << "Calls" << std::setw(15) << "TotalWall[ms]" << std::setw(15) << "TotalCPU[ms]" << std::setw(14)
        << "AvgWall[ms]" << std::setw(13) << "AvgCPU[ms]" << std::setw(18) << "AvgSelfWall[ms]" << std::setw(17)
        << "AvgSelfCPU[ms]" << std::setw(13) << "Efficiency" << '\n';
    tOs << std::string(150, '=') << '\n';

    for (auto& [name, stats] : sortedStats)
    {
      stats.threads = threadsPerFunc.count(name) ? threadsPerFunc[name].size() : 0;
      if (stats.calls == 0)
        continue;
      double scale = 1.0;
      double avgWallMs = (stats.totalWallMs * scale) / stats.calls;
      double avgCpuMs = (stats.totalCpuMs * scale) / stats.calls;
      double avgSelfWallMs = (stats.selfWallMs * scale) / stats.calls;
      double avgSelfCpuMs = (stats.selfCpuMs * scale) / stats.calls;
      double efficiency = stats.totalWallMs > 1e-9 ? (stats.totalCpuMs / stats.totalWallMs) * 100.0 : 0.0;

      std::string displayName = name.length() > 26 ? name.substr(0, 23) + "..." : name;
      tOs << std::left << std::setw(28) << displayName << std::right << std::setw(8) << stats.threads << std::setw(10)
          << stats.calls << std::fixed << std::setprecision(3) << std::setw(15) << stats.totalWallMs << std::setw(15)
          << stats.totalCpuMs << std::setprecision(3) << std::setw(14) << avgWallMs << std::setw(13) << avgCpuMs
          << std::setw(18) << avgSelfWallMs << std::setw(17) << avgSelfCpuMs << std::setw(12) << std::setprecision(3)
          << efficiency << "%" << '\n';
    }
    tOs << std::string(151, '=') << '\n';
  }

  static void
  printHierarchyReport(std::ostream& tOs = std::cout)
  {
    // Task #6: Use cached call graph
    auto tree = getCachedCallGraph();

    const int kNameColWidth = 50;
    tOs << "\n--- Hierarchical Report ---\n";
    tOs << std::string(kNameColWidth + 123, '=') << '\n';
    tOs << std::left << std::setw(kNameColWidth) << "Function" << std::right << std::setw(8) << "Threads"
        << std::setw(10) << "Calls" << std::setw(15) << "TotalWall[ms]" << std::setw(15) << "TotalCPU[ms]"
        << std::setw(14) << "AvgWall[ms]" << std::setw(13) << "AvgCPU[ms]" << std::setw(18) << "AvgSelfWall[ms]"
        << std::setw(17) << "AvgSelfCPU[ms]" << std::setw(13) << "Efficiency" << '\n';
    tOs << std::string(kNameColWidth + 123, '=') << '\n';

    auto countUtf8Chars = [](const std::string& tS)
    {
      size_t count = 0;
      for (size_t i = 0; i < tS.length();)
      {
        count++;
        unsigned char c = tS[i];
        if (c <= 127)
          i += 1;
        else if ((c & 0xE0) == 0xC0)
          i += 2;
        else if ((c & 0xF0) == 0xE0)
          i += 3;
        else if ((c & 0xF8) == 0xF0)
          i += 4;
        else
        {
          i++;
        }
      }
      return count;
    };

    std::function<void(ReportNode*, const std::string&, bool)> printNode =
      [&](ReportNode* tNode, const std::string& tPrefix, bool tIsLastSibling)
    {
      if (tNode->stats.calls == 0)
        return;

      std::string linePrefix = tPrefix + (tIsLastSibling ? "\u2514\u2500 " : "\u251C\u2500 ");
      std::string namePart = tNode->name;

      tOs << std::left << linePrefix << namePart;

      size_t visualPrefixWidth = countUtf8Chars(linePrefix);
      size_t visualNameWidth = namePart.length();
      int paddingNeeded = kNameColWidth - (visualPrefixWidth + visualNameWidth);
      if (paddingNeeded > 0)
      {
        tOs << std::string(paddingNeeded, ' ');
      }
      else
      {
        tOs << " ";
      }

      const auto& stats = tNode->stats;
      double scale = 1.0;
      double avgWallMs = (stats.totalWallMs * scale) / stats.calls;
      double avgCpuMs = (stats.totalCpuMs * scale) / stats.calls;
      double avgSelfWallMs = (stats.selfWallMs * scale) / stats.calls;
      double avgSelfCpuMs = (stats.selfCpuMs * scale) / stats.calls;
      double efficiency = stats.totalWallMs > 1e-9 ? (stats.totalCpuMs / stats.totalWallMs) * 100.0 : 0.0;

      tOs << std::right << std::setw(7) << stats.threads << " " << std::setw(9) << stats.calls << " " << std::fixed
          << std::setprecision(3) << std::setw(14) << stats.totalWallMs << " " << std::setw(14) << stats.totalCpuMs
          << " " << std::setprecision(3) << std::setw(13) << avgWallMs << " " << std::setw(12) << avgCpuMs << " "
          << std::setw(17) << avgSelfWallMs << " " << std::setw(16) << avgSelfCpuMs << " " << std::setw(12)
          << efficiency << "%" << '\n';

      // Children are already ordered by first call time due to vector structure
      for (size_t i = 0; i < tNode->children.size(); ++i)
      {
        bool isChildLast = (i == tNode->children.size() - 1);
        ReportNode* childPtr = tNode->children[i].get();
        std::string childPrefix = tPrefix + (tIsLastSibling ? "   " : "\u2502  ");
        printNode(childPtr, childPrefix, isChildLast);
      }
    };

    // Root children are already sorted by first call time
    for (size_t i = 0; i < tree->children.size(); ++i)
    {
      printNode(tree->children[i].get(), "", (i == tree->children.size() - 1));
    }

    tOs << std::string(kNameColWidth + 123, '=') << '\n';
  }

  static void
  report(std::ostream& tOs = std::cout)
  {
    printHierarchyReport(tOs);
    printFlatReport(tOs);
  }
};

// --- Platform specific time getters ---
inline std::chrono::nanoseconds
getThreadCpuTime()
{
#if defined(__linux__)
  timespec ts{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0)
  {
    return std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec);
  }
#elif defined(_WIN32)
  FILETIME createTime, exitTime, kernelTime, userTime;
  if (GetThreadTimes(GetCurrentThread(), &createTime, &exitTime, &kernelTime, &userTime))
  {
    ULARGE_INTEGER time;
    time.LowPart = userTime.dwLowDateTime;
    time.HighPart = userTime.dwHighDateTime;
    return std::chrono::nanoseconds(time.QuadPart * 100);
  }
#endif
  return std::chrono::nanoseconds(0);
}

// --- ScopeProfiler Implementation ---
inline ScopeProfiler::ScopeProfiler(const char* tName) : _name(tName)
{
  // Check if profiling is enabled at runtime
  if (!Profiler::kEnableProfiling)
  {
    return;
  }

  {
    std::unique_lock lock(Profiler::getGlobalData().dataMutex);
    Profiler::getGlobalData().getThreadDataForCurrentThread().callStack.emplace_back(tName);
  }
  _wallStart = std::chrono::steady_clock::now();

  // CPU time measurement: compile-time + runtime control for zero overhead when disabled
#ifdef SCV_CPU_TIME_ENABLED
  #if SCV_CPU_TIME_ENABLED
  if (Profiler::kEnableCpuTime)
  {
    _cpuStart = getThreadCpuTime();
  }
  #endif
#endif
}

inline ScopeProfiler::~ScopeProfiler()
{
  // Check if profiling is enabled at runtime
  if (!Profiler::kEnableProfiling)
  {
    return;
  }

  auto wallEnd = std::chrono::steady_clock::now();
  std::chrono::nanoseconds cpuEnd{};

  // CPU time measurement: compile-time + runtime control for zero overhead when disabled
#ifdef SCV_CPU_TIME_ENABLED
  #if SCV_CPU_TIME_ENABLED
  if (Profiler::kEnableCpuTime)
  {
    cpuEnd = getThreadCpuTime();
  }
  #endif
#endif

  // Record will handle its own locking
  Profiler::record(_name, _wallStart, wallEnd - _wallStart, cpuEnd - _cpuStart);

  {
    std::unique_lock lock(Profiler::getGlobalData().dataMutex);
    auto& callStack = Profiler::getGlobalData().getThreadDataForCurrentThread().callStack;
    if (!callStack.empty())
    {
      callStack.pop_back();
    }
  }
}

} // namespace scv

// --- USER MACROS ---
#define PROF_CONCAT_IMPL(x, y) x##y
#define PROF_CONCAT(x, y)      PROF_CONCAT_IMPL(x, y)
#define PROF_UNIQUE_VAR(base)  PROF_CONCAT(base, __LINE__)

// Conditional profiling macros based on CMake configuration
#ifdef SCV_PROFILING_ENABLED
  #if SCV_PROFILING_ENABLED
    #define PROFILE_SCOPE(name) scv::ScopeProfiler PROF_UNIQUE_VAR(profile_scope_)(name)
    #define PROFILE_FUNCTION()  PROFILE_SCOPE(__func__)

    // Allow access to profiler API when enabled
    #define PROFILER_GET_STATS(name)          scv::Profiler::getStats(name)
    #define PROFILER_GET_AVG_STAT(name, stat) scv::Profiler::getAvgStat(name, stat)
    #define PROFILER_PRINT_REPORT()           scv::Profiler::report()
  #else
  // Profiling disabled - compile to no-ops
    #define PROFILE_SCOPE(name)                                                                                        \
      do                                                                                                               \
      {}                                                                                                               \
      while (0)
    #define PROFILE_FUNCTION()                                                                                         \
      do                                                                                                               \
      {}                                                                                                               \
      while (0)

    // Provide dummy implementations that return safe defaults
    #define PROFILER_GET_STATS(name)                                                                                   \
      scv::Profiler::ScopeStats {}
    #define PROFILER_GET_AVG_STAT(name, stat) 0.0
    #define PROFILER_PRINT_REPORT()                                                                                    \
      do                                                                                                               \
      {}                                                                                                               \
      while (0)
  #endif
#else
  // Default to enabled for backward compatibility when macro is not defined
  #define PROFILE_SCOPE(name)               scv::ScopeProfiler PROF_UNIQUE_VAR(profile_scope_)(name)
  #define PROFILE_FUNCTION()                PROFILE_SCOPE(__func__)
  #define PROFILER_GET_STATS(name)          scv::Profiler::getStats(name)
  #define PROFILER_GET_AVG_STAT(name, stat) scv::Profiler::getAvgStat(name, stat)
  #define PROFILER_PRINT_REPORT()           scv::Profiler::report()
#endif

#endif // PROFILER_H
