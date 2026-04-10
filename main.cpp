// Standard library and MPI/OpenMP headers.
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <array>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <limits>
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#include <omp.h>

using CountType = unsigned int;
using TotalType = std::uint64_t;
using Clock = std::chrono::steady_clock;

// Metric index mapping used throughout the program.
// 0 -> lines, 1 -> words, 2 -> bytes/characters.


int main(int argc, char* argv[]) {
    // This build is intended to run with a fixed 16-rank MPI layout.
    constexpr int expectedRanks = 16;
    constexpr std::size_t timingPartCount = 8;
    constexpr std::array<const char*, timingPartCount - 1> timingLabels = {
        "Local file processing",
        "Row counter reduction",
        "File total reduction",
        "Median count exchange",
        "Median calculation",
        "Final aggregation",
        "Median value exchange"
    };

    // Initialize MPI first so any validation error can abort the whole job.
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto logProgress = [rank](const std::string& message, bool allRanks = false) {
        if (allRanks || rank == 0) {
            std::cout << "[progress][rank " << rank << "] " << message << std::endl;
        }
    };

    // Expect a directory path plus the requested node count.
    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <directory_path> <num_nodes>" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::string directoryPath = argv[1];
    const auto pipelineStart = Clock::now();
    std::array<double, timingPartCount> localTimings = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto elapsedSeconds = [](const Clock::time_point& start, const Clock::time_point& end) {
        return std::chrono::duration<double>(end - start).count();
    };
    // Validate the input directory before scanning it.
    if (!std::filesystem::is_directory(directoryPath)) {
        std::cerr << "Rank " << rank << " error: " << directoryPath << " is not a valid directory." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Collect regular files and sort them so every rank sees the same order.
    std::vector<std::string> fileNames;
    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            fileNames.push_back(entry.path().string());
        }
    }
    std::sort(fileNames.begin(), fileNames.end());
    int numNodes = 0;
    try {
        numNodes = std::stoi(argv[2]);
    } catch (const std::invalid_argument&) {
        if (rank == 0) {
            std::cerr << "Error: num_nodes must be a positive integer." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (const std::out_of_range&) {
        if (rank == 0) {
            std::cerr << "Error: num_nodes value is out of range." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (numNodes <= 0) {
        if (rank == 0) {
            std::cerr << "Error: num_nodes must be a positive integer." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (size != expectedRanks || numNodes != expectedRanks || size != numNodes) {
        if (rank == 0) {
            std::cerr << "Error: this program expects exactly " << expectedRanks
                      << " MPI ranks and num_nodes=" << expectedRanks << "." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    logProgress("Validated MPI world size and input arguments.");

    // Split work with quotient/remainder so every file is assigned exactly once.
    // First `extraFiles` ranks process one additional file.
    const int totalFiles = static_cast<int>(fileNames.size());
    const int baseFilesPerRank = totalFiles / numNodes;
    const int extraFiles = totalFiles % numNodes;
    const int start = rank * baseFilesPerRank + std::min(rank, extraFiles);
    const int localFileCount = baseFilesPerRank + (rank < extraFiles ? 1 : 0);
    const int end = start + localFileCount;
    std::cout << "Node " << rank << " processing files from index " << start << " to " << end - 1 << std::endl;

    // A bounded heap keeps only the top or bottom N entries for a metric.
    class BoundedHeap {
    private:
        using Entry = std::pair<std::string, CountType>;
        using CompareFn = bool (*)(const Entry&, const Entry&);

        std::vector<Entry> heap;
        std::size_t capacity;
        CompareFn compare;
        bool keepLargest;

        static bool minHeapCompare(const Entry& lhs, const Entry& rhs) {
            return lhs.second > rhs.second;
        }

        static bool maxHeapCompare(const Entry& lhs, const Entry& rhs) {
            return lhs.second < rhs.second;
        }

        bool shouldReplace(CountType incoming) const {
            return keepLargest ? (incoming > heap.front().second) : (incoming < heap.front().second);
        }

    public:
        BoundedHeap(std::size_t capacityValue, bool keepLargestValues)
            : capacity(capacityValue),
              compare(keepLargestValues ? &BoundedHeap::minHeapCompare : &BoundedHeap::maxHeapCompare),
              keepLargest(keepLargestValues) {
            heap.reserve(capacity);
        }

        void add(const std::string& fileName, CountType count) {
            if (heap.size() < capacity) {
                heap.emplace_back(fileName, count);
                std::push_heap(heap.begin(), heap.end(), compare);
                return;
            }

            if (heap.empty() || !shouldReplace(count)) {
                return;
            }

            std::pop_heap(heap.begin(), heap.end(), compare);
            heap.back() = {fileName, count};
            std::push_heap(heap.begin(), heap.end(), compare);
        }

        const std::vector<Entry>& entries() const {
            return heap;
        }

        std::vector<Entry> sorted() const {
            std::vector<Entry> ordered = heap;
            if (keepLargest) {
                std::sort(ordered.begin(), ordered.end(), [](const Entry& lhs, const Entry& rhs) {
                    return lhs.second > rhs.second;
                });
            } else {
                std::sort(ordered.begin(), ordered.end(), [](const Entry& lhs, const Entry& rhs) {
                    return lhs.second < rhs.second;
                });
            }
            return ordered;
        }
    };


    struct MedianValue {
        CountType upperMiddle;
        double arithmetic;

        MedianValue() : upperMiddle(0), arithmetic(0.0) {}
    };

    // Aggregated per-rank statistics and the MPI helpers used to merge them.
    class Results {
    private:
        using CountEntry = std::pair<CountType, CountType>;

        struct MetricData {
            std::vector<CountEntry> valueCounts;
            TotalType total;
            double average;
            BoundedHeap top;
            BoundedHeap bottom;
            MedianValue median;

            MetricData()
                : total(0),
                  average(0.0),
                  top(10, true),
                  bottom(10, false) {}
        };

        TotalType totalCount;
        MetricData line;
        MetricData word;
        MetricData character;

        MetricData& metricForIndex(int metricIndex) {
            switch (metricIndex) {
                case 0:
                    return line;
                case 1:
                    return word;
                case 2:
                    return character;
                default:
                    throw std::out_of_range("metricIndex out of range");
            }
        }

        const MetricData& metricForIndex(int metricIndex) const {
            switch (metricIndex) {
                case 0:
                    return line;
                case 1:
                    return word;
                case 2:
                    return character;
                default:
                    throw std::out_of_range("metricIndex out of range");
            }
        }

        static bool parseNonNegativeCount(const std::string& field, CountType& value) {
            try {
                std::size_t parsed = 0;
                long long signedValue = std::stoll(field, &parsed, 10);
                if (parsed != field.size()) {
                    return false;
                }

                if (signedValue < 0) {
                    // Treat negative counts as zero so malformed rows are tolerated.
                    value = 0;
                    return true;
                }

                const auto maxCount = static_cast<long long>(std::numeric_limits<CountType>::max());
                value = static_cast<CountType>(signedValue > maxCount ? maxCount : signedValue);
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }

    public:
        static bool parseCsvLine(
            const std::string& rawLine,
            std::string& name,
            CountType& bytes,
            CountType& words,
            CountType& lines
        ) {
            if (rawLine.empty()) {
                return false;
            }

            auto trim = [](const std::string& value) -> std::string {
                std::size_t first = value.find_first_not_of(" \t\r\n");
                if (first == std::string::npos) {
                    return std::string();
                }
                std::size_t last = value.find_last_not_of(" \t\r\n");
                return value.substr(first, last - first + 1);
            };

            // Find the last 3 delimiters from the right so quoted names can contain commas.
            std::array<std::size_t, 3> delimiterPos = {std::string::npos, std::string::npos, std::string::npos};
            int foundDelimiters = 0;
            bool inQuotes = false;

            for (std::size_t idx = rawLine.size(); idx > 0; --idx) {
                const char current = rawLine[idx - 1];
                if (current == '"') {
                    inQuotes = !inQuotes;
                    continue;
                }
                if (!inQuotes && current == ',') {
                    delimiterPos[static_cast<std::size_t>(2 - foundDelimiters)] = idx - 1;
                    ++foundDelimiters;
                    if (foundDelimiters == 3) {
                        break;
                    }
                }
            }

            if (foundDelimiters != 3) {
                return false;
            }

            name = trim(rawLine.substr(0, delimiterPos[0]));

            std::array<std::string, 3> numericFields;
            numericFields[0] = trim(rawLine.substr(delimiterPos[0] + 1, delimiterPos[1] - delimiterPos[0] - 1));
            numericFields[1] = trim(rawLine.substr(delimiterPos[1] + 1, delimiterPos[2] - delimiterPos[1] - 1));
            numericFields[2] = trim(rawLine.substr(delimiterPos[2] + 1));

            if (name.empty() || numericFields[0].empty() || numericFields[1].empty() || numericFields[2].empty()) {
                return false;
            }

            if (name.front() == '"' && name.back() == '"' && name.size() >= 2) {
                std::string unescaped;
                unescaped.reserve(name.size() - 2);
                for (std::size_t idx = 1; idx + 1 < name.size(); ++idx) {
                    if (name[idx] == '"' && idx + 1 < name.size() - 1 && name[idx + 1] == '"') {
                        unescaped.push_back('"');
                        ++idx;
                    } else {
                        unescaped.push_back(name[idx]);
                    }
                }
                name.swap(unescaped);
            }

            return parseNonNegativeCount(numericFields[0], bytes)
                && parseNonNegativeCount(numericFields[1], words)
                && parseNonNegativeCount(numericFields[2], lines);
        }

    private:
        static void mergeHeaps(BoundedHeap& target, const BoundedHeap& source) {
            // Reinsert source entries to preserve target heap size constraints.
            for (const auto& entry : source.entries()) {
                target.add(entry.first, entry.second);
            }
        }

        static void insertCount(std::vector<CountEntry>& counts, CountType value, CountType frequency = 1) {
            // Keep the histogram sorted by value for efficient median lookup later.
            auto position = std::lower_bound(
                counts.begin(),
                counts.end(),
                value,
                [](const CountEntry& entry, CountType candidateValue) {
                    return entry.first < candidateValue;
                }
            );

            if (position != counts.end() && position->first == value) {
                position->second += frequency;
                return;
            }

            counts.insert(position, {value, frequency});
        }

        static void mergeCounts(std::vector<CountEntry>& target, const std::vector<CountEntry>& source) {
            // Merge two sorted histograms in linear time.
            std::vector<CountEntry> merged;
            merged.reserve(target.size() + source.size());

            std::size_t left = 0;
            std::size_t right = 0;
            while (left < target.size() && right < source.size()) {
                if (target[left].first < source[right].first) {
                    merged.push_back(target[left++]);
                } else if (source[right].first < target[left].first) {
                    merged.push_back(source[right++]);
                } else {
                    merged.push_back({target[left].first, target[left].second + source[right].second});
                    ++left;
                    ++right;
                }
            }

            while (left < target.size()) {
                merged.push_back(target[left++]);
            }

            while (right < source.size()) {
                merged.push_back(source[right++]);
            }

            target.swap(merged);
        }

        static void addMetric(const std::string& fileName, CountType value, TotalType currentTotalCount, MetricData& metric) {
            insertCount(metric.valueCounts, value);
            metric.total += value;
            metric.average = currentTotalCount > 0
                ? static_cast<double>(metric.total) / static_cast<double>(currentTotalCount)
                : 0.0;
            metric.top.add(fileName, value);
            metric.bottom.add(fileName, value);
        }

        static void recomputeAverages(TotalType currentTotalCount, MetricData& metric) {
            metric.average = currentTotalCount > 0
                ? static_cast<double>(metric.total) / static_cast<double>(currentTotalCount)
                : 0.0;
        }

        static MedianValue computeMedianFromCounts(const std::vector<CountEntry>& counts, TotalType sampleCount) {
            MedianValue median;
            if (sampleCount == 0 || counts.empty()) {
                return median;
            }

            const TotalType lowerIndex = (sampleCount - 1) / 2;
            const TotalType upperIndex = sampleCount / 2;
            TotalType runningIndex = 0;
            CountType lowerMiddle = 0;
            CountType upperMiddle = 0;
            bool lowerFound = false;
            bool upperFound = false;

            // Walk histogram buckets until both median positions are covered.
            for (const auto& [value, frequency] : counts) {
                const TotalType nextIndex = runningIndex + frequency - 1;
                if (!lowerFound && lowerIndex >= runningIndex && lowerIndex <= nextIndex) {
                    lowerMiddle = value;
                    lowerFound = true;
                }
                if (!upperFound && upperIndex >= runningIndex && upperIndex <= nextIndex) {
                    upperMiddle = value;
                    upperFound = true;
                }
                if (lowerFound && upperFound) {
                    break;
                }
                runningIndex += frequency;
            }

            median.upperMiddle = upperMiddle;
            median.arithmetic = (sampleCount % 2 == 0)
                ? (static_cast<double>(lowerMiddle) + static_cast<double>(upperMiddle)) / 2.0
                : static_cast<double>(upperMiddle);
            return median;
        }

        static void sendCountMap(const std::vector<CountEntry>& counts, int dest, int baseTag) {
            // Send as (value, frequency) pairs to avoid serializing complex containers.
            int entryCount = static_cast<int>(counts.size());
            MPI_Send(&entryCount, 1, MPI_INT, dest, baseTag, MPI_COMM_WORLD);
            for (const auto& [value, frequency] : counts) {
                const std::array<CountType, 2> entry = {value, frequency};
                MPI_Send(entry.data(), static_cast<int>(entry.size()), MPI_UNSIGNED, dest, baseTag + 1, MPI_COMM_WORLD);
            }
        }

        static void recvCountMap(std::vector<CountEntry>& counts, int src, int baseTag) {
            int entryCount = 0;
            MPI_Recv(&entryCount, 1, MPI_INT, src, baseTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int idx = 0; idx < entryCount; ++idx) {
                std::array<CountType, 2> entry = {0, 0};
                MPI_Recv(entry.data(), static_cast<int>(entry.size()), MPI_UNSIGNED, src, baseTag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                insertCount(counts, entry[0], entry[1]);
            }
        }

        static void sendHeap(const BoundedHeap& heapObj, int dest, int baseTag) {
            // File names have variable length, so send length + bytes for each entry.
            int heapSize = static_cast<int>(heapObj.entries().size());
            MPI_Send(&heapSize, 1, MPI_INT, dest, baseTag, MPI_COMM_WORLD);
            for (const auto& entry : heapObj.entries()) {
                int fileNameLength = static_cast<int>(entry.first.size());
                MPI_Send(&fileNameLength, 1, MPI_INT, dest, baseTag + 1, MPI_COMM_WORLD);
                MPI_Send(entry.first.data(), fileNameLength, MPI_CHAR, dest, baseTag + 2, MPI_COMM_WORLD);
                MPI_Send(&entry.second, 1, MPI_UNSIGNED, dest, baseTag + 3, MPI_COMM_WORLD);
            }
        }

        static void recvHeap(BoundedHeap& heapObj, int src, int baseTag) {
            int heapSize = 0;
            MPI_Recv(&heapSize, 1, MPI_INT, src, baseTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int idx = 0; idx < heapSize; ++idx) {
                int fileNameLength = 0;
                MPI_Recv(&fileNameLength, 1, MPI_INT, src, baseTag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::string fileName(static_cast<size_t>(fileNameLength), '\0');
                MPI_Recv(fileName.data(), fileNameLength, MPI_CHAR, src, baseTag + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CountType count = 0;
                MPI_Recv(&count, 1, MPI_UNSIGNED, src, baseTag + 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                heapObj.add(fileName, count);
            }
        }

    public:
        Results() : totalCount(0) {}

        void addFileResult(const std::string& fileName, CountType lineCount, CountType wordCount, CountType charCount) {
            ++totalCount;
            addMetric(fileName, lineCount, totalCount, line);
            addMetric(fileName, wordCount, totalCount, word);
            addMetric(fileName, charCount, totalCount, character);
        }

        void mergeFrom(const Results& other) {
            // Full merge includes histograms so medians can be recomputed globally.
            totalCount += other.totalCount;

            mergeCounts(line.valueCounts, other.line.valueCounts);
            mergeCounts(word.valueCounts, other.word.valueCounts);
            mergeCounts(character.valueCounts, other.character.valueCounts);

            line.total += other.line.total;
            word.total += other.word.total;
            character.total += other.character.total;

            recomputeAverages(totalCount, line);
            recomputeAverages(totalCount, word);
            recomputeAverages(totalCount, character);

            mergeHeaps(line.top, other.line.top);
            mergeHeaps(line.bottom, other.line.bottom);
            mergeHeaps(word.top, other.word.top);
            mergeHeaps(word.bottom, other.word.bottom);
            mergeHeaps(character.top, other.character.top);
            mergeHeaps(character.bottom, other.character.bottom);
        }

        void mergeSummaryFrom(const Results& other) {
            // Lightweight merge used for final reporting when medians are handled separately.
            totalCount += other.totalCount;

            line.total += other.line.total;
            word.total += other.word.total;
            character.total += other.character.total;

            recomputeAverages(totalCount, line);
            recomputeAverages(totalCount, word);
            recomputeAverages(totalCount, character);

            mergeHeaps(line.top, other.line.top);
            mergeHeaps(line.bottom, other.line.bottom);
            mergeHeaps(word.top, other.word.top);
            mergeHeaps(word.bottom, other.word.bottom);
            mergeHeaps(character.top, other.character.top);
            mergeHeaps(character.bottom, other.character.bottom);
        }

        void computeMedians() {
            line.median = computeMedianFromCounts(line.valueCounts, totalCount);
            word.median = computeMedianFromCounts(word.valueCounts, totalCount);
            character.median = computeMedianFromCounts(character.valueCounts, totalCount);
        }

        void computeMedianForMetric(int metricIndex, TotalType sampleCount) {
            metricForIndex(metricIndex).median = computeMedianFromCounts(metricForIndex(metricIndex).valueCounts, sampleCount);
        }

        void setMedianForMetric(int metricIndex, const MedianValue& median) {
            metricForIndex(metricIndex).median = median;
        }

        MedianValue getMedianForMetric(int metricIndex) const {
            return metricForIndex(metricIndex).median;
        }

        std::array<TotalType, 4> packStats() const {
            return {
                totalCount,
                line.total,
                word.total,
                character.total
            };
        }

        void unpackStats(const std::array<TotalType, 4>& packedStats) {
            totalCount = packedStats[0];
            line.total = packedStats[1];
            word.total = packedStats[2];
            character.total = packedStats[3];

            recomputeAverages(totalCount, line);
            recomputeAverages(totalCount, word);
            recomputeAverages(totalCount, character);
        }

        void sendSummaryToRank(int dest, int statsTag, int heapBaseTag) const {
            // Tag blocks are spaced by +10 to keep each metric channel distinct.
            const auto packedStats = packStats();
            MPI_Send(packedStats.data(), static_cast<int>(packedStats.size()), MPI_UINT64_T, dest, statsTag, MPI_COMM_WORLD);

            sendHeap(line.top, dest, heapBaseTag);
            sendHeap(line.bottom, dest, heapBaseTag + 10);
            sendHeap(word.top, dest, heapBaseTag + 20);
            sendHeap(word.bottom, dest, heapBaseTag + 30);
            sendHeap(character.top, dest, heapBaseTag + 40);
            sendHeap(character.bottom, dest, heapBaseTag + 50);
        }

        void recvSummaryFromRank(int src, int statsTag, int heapBaseTag) {
            std::array<TotalType, 4> packedStats = {0, 0, 0, 0};

            MPI_Recv(packedStats.data(), static_cast<int>(packedStats.size()), MPI_UINT64_T, src, statsTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            unpackStats(packedStats);

            recvHeap(line.top, src, heapBaseTag);
            recvHeap(line.bottom, src, heapBaseTag + 10);
            recvHeap(word.top, src, heapBaseTag + 20);
            recvHeap(word.bottom, src, heapBaseTag + 30);
            recvHeap(character.top, src, heapBaseTag + 40);
            recvHeap(character.bottom, src, heapBaseTag + 50);
        }

        void sendMetricCountsToRank(int metricIndex, int dest, int baseTag) const {
            sendCountMap(metricForIndex(metricIndex).valueCounts, dest, baseTag);
        }

        void recvMetricCountsFromRank(int metricIndex, int src, int baseTag) {
            recvCountMap(metricForIndex(metricIndex).valueCounts, src, baseTag);
        }

        void sendMedianToRank(int dest, int metricIndex, int baseTag) const {
            const MedianValue& median = metricForIndex(metricIndex).median;
            MPI_Send(&median.upperMiddle, 1, MPI_UNSIGNED, dest, baseTag, MPI_COMM_WORLD);
            MPI_Send(&median.arithmetic, 1, MPI_DOUBLE, dest, baseTag + 1, MPI_COMM_WORLD);
        }

        void recvMedianFromRank(int metricIndex, int src, int baseTag) {
            MedianValue median;
            MPI_Recv(&median.upperMiddle, 1, MPI_UNSIGNED, src, baseTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&median.arithmetic, 1, MPI_DOUBLE, src, baseTag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            setMedianForMetric(metricIndex, median);
        }

        void sendToRank(int dest, int statsTag, int countsBaseTag, int heapBaseTag) const {
            const auto packedStats = packStats();
            MPI_Send(packedStats.data(), static_cast<int>(packedStats.size()), MPI_UINT64_T, dest, statsTag, MPI_COMM_WORLD);

            sendCountMap(line.valueCounts, dest, countsBaseTag);
            sendCountMap(word.valueCounts, dest, countsBaseTag + 10);
            sendCountMap(character.valueCounts, dest, countsBaseTag + 20);

            sendHeap(line.top, dest, heapBaseTag);
            sendHeap(line.bottom, dest, heapBaseTag + 10);
            sendHeap(word.top, dest, heapBaseTag + 20);
            sendHeap(word.bottom, dest, heapBaseTag + 30);
            sendHeap(character.top, dest, heapBaseTag + 40);
            sendHeap(character.bottom, dest, heapBaseTag + 50);
        }

        void recvFromRank(int src, int statsTag, int countsBaseTag, int heapBaseTag) {
            std::array<TotalType, 4> packedStats = {0, 0, 0, 0};

            MPI_Recv(packedStats.data(), static_cast<int>(packedStats.size()), MPI_UINT64_T, src, statsTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            unpackStats(packedStats);

            recvCountMap(line.valueCounts, src, countsBaseTag);
            recvCountMap(word.valueCounts, src, countsBaseTag + 10);
            recvCountMap(character.valueCounts, src, countsBaseTag + 20);

            recvHeap(line.top, src, heapBaseTag);
            recvHeap(line.bottom, src, heapBaseTag + 10);
            recvHeap(word.top, src, heapBaseTag + 20);
            recvHeap(word.bottom, src, heapBaseTag + 30);
            recvHeap(character.top, src, heapBaseTag + 40);
            recvHeap(character.bottom, src, heapBaseTag + 50);
        }

        void printSummary(std::ostream& out) const {
            auto printHeapEntries = [&out](const std::string& title, const BoundedHeap& heap) {
                out << title << std::endl;
                for (const auto& entry : heap.sorted()) {
                    out << entry.first << ": " << entry.second << std::endl;
                }
            };

            auto printMetricSummary = [&out, &printHeapEntries](const std::string& label, const MetricData& metric) {
                out << label << " - Total: " << metric.total
                    << ", Average: " << metric.average
                    << ", Median (upper-middle): " << metric.median.upperMiddle
                    << ", Median (arithmetic): " << metric.median.arithmetic << std::endl;
                printHeapEntries("Top 10 " + label + "s:", metric.top);
                printHeapEntries("Bottom 10 " + label + "s:", metric.bottom);
            };

            out << "Stats:" << std::endl;
            out << "Total files processed (rows processed): " << totalCount << std::endl;
            printMetricSummary("line count", line);
            printMetricSummary("word count", word);
            printMetricSummary("character count", character);
        }
    };

    // Per-rank state used for local aggregation before MPI reductions.
    Results nodeResults;
    TotalType nodeRowsSeen = 0;
    TotalType nodeRowsParsed = 0;
    TotalType nodeRowsSkipped = 0;
    logProgress("Starting local file processing.", true);

    const auto localProcessingStart = Clock::now();

    // Process the assigned files in parallel across OpenMP threads.
    #pragma omp parallel
    {
        Results threadResults;
        TotalType threadRowsSeen = 0;
        TotalType threadRowsParsed = 0;
        TotalType threadRowsSkipped = 0;

        #pragma omp for nowait
        for (int i = start; i < end; ++i) {
            std::ifstream file(fileNames[static_cast<std::size_t>(i)]);
            if (!file.is_open()) {
                #pragma omp critical
                {
                    std::cerr << "Rank " << rank
                              << " thread " << omp_get_thread_num()
                              << " error: Could not open file "
                              << fileNames[static_cast<std::size_t>(i)] << std::endl;
                }
                continue;
            }

            std::string line;

            // Each input line is a CSV record in the form: name, bytes, words, lines.
            while (std::getline(file, line)) {
                ++threadRowsSeen;
                std::string name;
                CountType bytes = 0;
                CountType words = 0;
                CountType lines = 0;
                if (Results::parseCsvLine(line, name, bytes, words, lines)) {
                    threadResults.addFileResult(name, lines, words, bytes);
                    ++threadRowsParsed;
                } else {
                    ++threadRowsSkipped;
                }
            }
            file.close();
        }

        // Merge once per thread to keep contention low.
        #pragma omp critical
        {
            nodeResults.mergeFrom(threadResults);
            nodeRowsSeen += threadRowsSeen;
            nodeRowsParsed += threadRowsParsed;
            nodeRowsSkipped += threadRowsSkipped;
        }
    }
    localTimings[0] = elapsedSeconds(localProcessingStart, Clock::now());
    logProgress("Completed local file processing.", true);

    std::array<TotalType, 3> nodeParseCounters = {nodeRowsSeen, nodeRowsParsed, nodeRowsSkipped};
    std::array<TotalType, 3> globalParseCounters = {0, 0, 0};
    const auto parseReduceStart = Clock::now();
    // Aggregate row-level quality counters onto rank 0.
    MPI_Reduce(
        nodeParseCounters.data(),
        globalParseCounters.data(),
        static_cast<int>(nodeParseCounters.size()),
        MPI_UINT64_T,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );
    localTimings[1] = elapsedSeconds(parseReduceStart, Clock::now());

    const TotalType localFileTotal = nodeResults.packStats()[0];
    TotalType globalFileTotal = 0;
    const auto globalTotalReduceStart = Clock::now();
    // Every rank needs the global sample count for median computation.
    MPI_Allreduce(&localFileTotal, &globalFileTotal, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    localTimings[2] = elapsedSeconds(globalTotalReduceStart, Clock::now());

    constexpr std::array<int, 3> medianOwners = {1, 2, 3};
    constexpr int medianCountsTagBase = 200;
    constexpr int medianValueTagBase = 260;

    // Distribute median work: one owner rank per metric collects and computes.
    const auto medianCountExchangeStart = Clock::now();
    for (int metricIndex = 0; metricIndex < 3; ++metricIndex) {
        const int ownerRank = medianOwners[static_cast<std::size_t>(metricIndex)];
        const int countTagBase = medianCountsTagBase + metricIndex * 10;

        if (rank == ownerRank) {
            for (int src = 0; src < size; ++src) {
                if (src == ownerRank) {
                    continue;
                }
                nodeResults.recvMetricCountsFromRank(metricIndex, src, countTagBase);
            }
            nodeResults.computeMedianForMetric(metricIndex, globalFileTotal);
        } else {
            nodeResults.sendMetricCountsToRank(metricIndex, ownerRank, countTagBase);
        }
    }
    localTimings[3] = elapsedSeconds(medianCountExchangeStart, Clock::now());

    const auto medianCalculationStart = Clock::now();
    // Owner ranks recompute from merged histograms to ensure deterministic medians.
    for (int metricIndex = 0; metricIndex < 3; ++metricIndex) {
        const int ownerRank = medianOwners[static_cast<std::size_t>(metricIndex)];
        if (rank == ownerRank) {
            nodeResults.computeMedianForMetric(metricIndex, globalFileTotal);
        }
    }
    localTimings[4] = elapsedSeconds(medianCalculationStart, Clock::now());

    // Gather non-median summary data to rank 0 for the final report.
    Results finalResults;
    const auto finalAggregationStart = Clock::now();
    if (rank == 0) {
        logProgress("Starting final aggregation on rank 0.");
        finalResults.mergeSummaryFrom(nodeResults);
        for (int i = 1; i < size; ++i) {
            Results recvResults;

            recvResults.recvSummaryFromRank(i, 100, 140);
            finalResults.mergeSummaryFrom(recvResults);
        }
        logProgress("Final aggregation complete.");
    } else if (rank > 0) {
        nodeResults.sendSummaryToRank(0, 100, 140);
    }
    localTimings[5] = elapsedSeconds(finalAggregationStart, Clock::now());

    MPI_Barrier(MPI_COMM_WORLD);

    const auto medianValueExchangeStart = Clock::now();
    // Send finalized median values from owner ranks to rank 0 for final output.
    if (rank == 0) {
        for (int metricIndex = 0; metricIndex < 3; ++metricIndex) {
            finalResults.recvMedianFromRank(metricIndex, medianOwners[static_cast<std::size_t>(metricIndex)], medianValueTagBase + metricIndex * 10);
        }
    } else if (rank >= 1 && rank <= 3) {
        const int metricIndex = rank - 1;
        nodeResults.sendMedianToRank(0, metricIndex, medianValueTagBase + metricIndex * 10);
    }
    localTimings[6] = elapsedSeconds(medianValueExchangeStart, Clock::now());

    MPI_Barrier(MPI_COMM_WORLD);
    localTimings[7] = elapsedSeconds(pipelineStart, Clock::now());

    std::vector<double> gatheredTimings;
    if (rank == 0) {
        gatheredTimings.resize(static_cast<std::size_t>(size) * timingPartCount, 0.0);
    }
    // Gather per-rank stage timings so rank 0 can print totals and averages.
    MPI_Gather(
        localTimings.data(),
        static_cast<int>(timingPartCount),
        MPI_DOUBLE,
        rank == 0 ? gatheredTimings.data() : nullptr,
        static_cast<int>(timingPartCount),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Cleanly shut down MPI before printing the final report.
    MPI_Finalize();

    // Only rank 0 prints the final aggregated results.
    if (rank == 0) {
        finalResults.printSummary(std::cout);
        std::cout << "rows_seen: " << globalParseCounters[0] << std::endl;
        std::cout << "rows_parsed: " << globalParseCounters[1] << std::endl;
        std::cout << "rows_skipped: " << globalParseCounters[2] << std::endl;

        std::cout << "Timing summary:" << std::endl;
        std::array<double, timingPartCount - 1> stageTotals = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int node = 0; node < size; ++node) {
            const std::size_t baseIndex = static_cast<std::size_t>(node) * timingPartCount;
            for (std::size_t stage = 0; stage + 1 < timingPartCount; ++stage) {
                stageTotals[stage] += gatheredTimings[baseIndex + stage];
            }
        }

        for (std::size_t stage = 0; stage + 1 < timingPartCount; ++stage) {
            const double stageAverage = stageTotals[stage] / static_cast<double>(size);
            std::cout << timingLabels[stage] << " total: " << std::fixed << std::setprecision(6)
                      << stageTotals[stage] << " s"
                      << " (avg per node: " << stageAverage << " s)" << std::endl;
        }

        double endToEndTotal = 0.0;
        for (int node = 0; node < size; ++node) {
            const std::size_t baseIndex = static_cast<std::size_t>(node) * timingPartCount;
            endToEndTotal = std::max(endToEndTotal, gatheredTimings[baseIndex + timingPartCount - 1]);
        }
        std::cout << "End-to-end total: " << std::fixed << std::setprecision(6)
                  << endToEndTotal << " s (max node time)" << std::endl;

        std::cout << "Node times:" << std::endl;
        for (int node = 0; node < size; ++node) {
            const std::size_t baseIndex = static_cast<std::size_t>(node) * timingPartCount;
            std::cout << "Node " << node << " total: " << std::fixed << std::setprecision(6)
                      << gatheredTimings[baseIndex + timingPartCount - 1] << " s" << std::endl;
        }
    }

    return 0;
}