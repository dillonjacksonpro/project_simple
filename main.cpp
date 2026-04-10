// get all header info
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <array>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#include <omp.h>


int main(int argc, char* argv[]) {
    constexpr int expectedRanks = 4;

    // init mpi first so any fatal validation can fail collectively
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // args: directory path and expected node count
    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <directory_path> <num_nodes>" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::string directoryPath = argv[1];
    // check if the provided path is a valid directory
    if (!std::filesystem::is_directory(directoryPath)) {
        std::cerr << "Rank " << rank << " error: " << directoryPath << " is not a valid directory." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // use fstat to get entries in the directory
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

    // split the file names into ranges for each node
    std::vector<std::pair<int, int>> nodeRanges;
    int filesPerNode = static_cast<int>(fileNames.size() / static_cast<std::size_t>(numNodes));
    for (int i = 0; i < numNodes; ++i) {
        int start = i * filesPerNode;
        int end = (i == numNodes - 1) ? static_cast<int>(fileNames.size()) : (i + 1) * filesPerNode;
        nodeRanges.emplace_back(start, end);
    }

    if (size != expectedRanks || numNodes != expectedRanks || size != numNodes) {
        if (rank == 0) {
            std::cerr << "Error: this program expects exactly " << expectedRanks
                      << " MPI ranks and num_nodes=" << expectedRanks << "." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // using rank, get the range of files to process for this node
    int start = nodeRanges[static_cast<std::size_t>(rank)].first;
    int end = nodeRanges[static_cast<std::size_t>(rank)].second;
    std::cout << "Node " << rank << " processing files from index " << start << " to " << end - 1 << std::endl;

    // define a high performance simple heap to keep track of the top/bottom 10 values for line count, word count, and character count
    // this heap should be able to efficiently maintain the top/bottom 10 values as new values are added

    class BoundedHeap {
    private:
        using Entry = std::pair<std::string, int>;
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

        bool shouldReplace(int incoming) const {
            return keepLargest ? (incoming > heap.front().second) : (incoming < heap.front().second);
        }

    public:
        BoundedHeap(std::size_t capacityValue, bool keepLargestValues)
            : capacity(capacityValue),
              compare(keepLargestValues ? &BoundedHeap::minHeapCompare : &BoundedHeap::maxHeapCompare),
              keepLargest(keepLargestValues) {
            heap.reserve(capacity);
        }

        void add(const std::string& fileName, int count) {
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

    // define a node unordered map to store results
    // map needs a field for:
    // total count - only a single value
    // line count, word count, character count
    // each of those fields needs a vector for raw values, a running total, an average, and a top/bottom 10 simple heap
    // add median to the results struct as well, but we will calculate it later after we have all the raw values

    struct MedianValue {
        int upperMiddle;
        double arithmetic;

        MedianValue() : upperMiddle(0), arithmetic(0.0) {}
    };

    class Results {
    private:
        struct MetricData {
            std::vector<int> rawCounts;
            int total;
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

        int totalCount;
        MetricData line;
        MetricData word;
        MetricData character;

        static void mergeHeaps(BoundedHeap& target, const BoundedHeap& source) {
            for (const auto& entry : source.entries()) {
                target.add(entry.first, entry.second);
            }
        }

        static void addMetric(const std::string& fileName, int value, int currentTotalCount, MetricData& metric) {
            metric.rawCounts.push_back(value);
            metric.total += value;
            metric.average = currentTotalCount > 0 ? static_cast<double>(metric.total) / currentTotalCount : 0.0;
            metric.top.add(fileName, value);
            metric.bottom.add(fileName, value);
        }

        static void recomputeAverages(int currentTotalCount, MetricData& metric) {
            metric.average = currentTotalCount > 0 ? static_cast<double>(metric.total) / currentTotalCount : 0.0;
        }

        static void sendHeap(const BoundedHeap& heapObj, int dest, int baseTag) {
            int heapSize = static_cast<int>(heapObj.entries().size());
            MPI_Send(&heapSize, 1, MPI_INT, dest, baseTag, MPI_COMM_WORLD);
            for (const auto& entry : heapObj.entries()) {
                int fileNameLength = static_cast<int>(entry.first.size());
                MPI_Send(&fileNameLength, 1, MPI_INT, dest, baseTag + 1, MPI_COMM_WORLD);
                MPI_Send(entry.first.data(), fileNameLength, MPI_CHAR, dest, baseTag + 2, MPI_COMM_WORLD);
                MPI_Send(&entry.second, 1, MPI_INT, dest, baseTag + 3, MPI_COMM_WORLD);
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
                int count = 0;
                MPI_Recv(&count, 1, MPI_INT, src, baseTag + 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                heapObj.add(fileName, count);
            }
        }

    public:
        Results() : totalCount(0) {}

        void addFileResult(const std::string& fileName, int lineCount, int wordCount, int charCount) {
            ++totalCount;
            addMetric(fileName, lineCount, totalCount, line);
            addMetric(fileName, wordCount, totalCount, word);
            addMetric(fileName, charCount, totalCount, character);
        }

        void mergeFrom(const Results& other) {
            totalCount += other.totalCount;

            line.rawCounts.insert(line.rawCounts.end(), other.line.rawCounts.begin(), other.line.rawCounts.end());
            word.rawCounts.insert(word.rawCounts.end(), other.word.rawCounts.begin(), other.word.rawCounts.end());
            character.rawCounts.insert(character.rawCounts.end(), other.character.rawCounts.begin(), other.character.rawCounts.end());

            line.total += other.line.total;
            word.total += other.word.total;
            character.total += other.character.total;

            recomputeAverages(totalCount, line);
            recomputeAverages(totalCount, word);
            recomputeAverages(totalCount, character);

            // Only one rank computes each median field; summation acts as selection during merge.
            line.median.upperMiddle += other.line.median.upperMiddle;
            line.median.arithmetic += other.line.median.arithmetic;
            word.median.upperMiddle += other.word.median.upperMiddle;
            word.median.arithmetic += other.word.median.arithmetic;
            character.median.upperMiddle += other.character.median.upperMiddle;
            character.median.arithmetic += other.character.median.arithmetic;

            mergeHeaps(line.top, other.line.top);
            mergeHeaps(line.bottom, other.line.bottom);
            mergeHeaps(word.top, other.word.top);
            mergeHeaps(word.bottom, other.word.bottom);
            mergeHeaps(character.top, other.character.top);
            mergeHeaps(character.bottom, other.character.bottom);
        }

        const std::vector<int>& lineCounts() const { return line.rawCounts; }
        const std::vector<int>& wordCounts() const { return word.rawCounts; }
        const std::vector<int>& charCounts() const { return character.rawCounts; }

        void clearRawCounts() {
            line.rawCounts.clear();
            word.rawCounts.clear();
            character.rawCounts.clear();
        }

        void setMedians(const MedianValue& lineMedian, const MedianValue& wordMedian, const MedianValue& charMedian) {
            line.median = lineMedian;
            word.median = wordMedian;
            character.median = charMedian;
        }

        std::array<int, 7> packStats() const {
            return {
                totalCount,
                line.total,
                word.total,
                character.total,
                line.median.upperMiddle,
                word.median.upperMiddle,
                character.median.upperMiddle
            };
        }

        std::array<double, 3> packMedianArithmetic() const {
            return {
                line.median.arithmetic,
                word.median.arithmetic,
                character.median.arithmetic
            };
        }

        void unpackStats(const std::array<int, 7>& packedStats,
                         const std::array<double, 3>& packedMedianArithmetic) {
            totalCount = packedStats[0];
            line.total = packedStats[1];
            word.total = packedStats[2];
            character.total = packedStats[3];
            line.median.upperMiddle = packedStats[4];
            word.median.upperMiddle = packedStats[5];
            character.median.upperMiddle = packedStats[6];

            line.median.arithmetic = packedMedianArithmetic[0];
            word.median.arithmetic = packedMedianArithmetic[1];
            character.median.arithmetic = packedMedianArithmetic[2];

            recomputeAverages(totalCount, line);
            recomputeAverages(totalCount, word);
            recomputeAverages(totalCount, character);
        }

        void sendToRank(int dest, int statsTag, int medianTag, int heapBaseTag) const {
            const auto packedStats = packStats();
            const auto packedMedian = packMedianArithmetic();
            MPI_Send(packedStats.data(), static_cast<int>(packedStats.size()), MPI_INT, dest, statsTag, MPI_COMM_WORLD);
            MPI_Send(packedMedian.data(), static_cast<int>(packedMedian.size()), MPI_DOUBLE, dest, medianTag, MPI_COMM_WORLD);

            sendHeap(line.top, dest, heapBaseTag);
            sendHeap(line.bottom, dest, heapBaseTag + 10);
            sendHeap(word.top, dest, heapBaseTag + 20);
            sendHeap(word.bottom, dest, heapBaseTag + 30);
            sendHeap(character.top, dest, heapBaseTag + 40);
            sendHeap(character.bottom, dest, heapBaseTag + 50);
        }

        void recvFromRank(int src, int statsTag, int medianTag, int heapBaseTag) {
            std::array<int, 7> packedStats = {0, 0, 0, 0, 0, 0, 0};
            std::array<double, 3> packedMedianArithmetic = {0.0, 0.0, 0.0};

            MPI_Recv(packedStats.data(), static_cast<int>(packedStats.size()), MPI_INT, src, statsTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(packedMedianArithmetic.data(), static_cast<int>(packedMedianArithmetic.size()), MPI_DOUBLE, src, medianTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            unpackStats(packedStats, packedMedianArithmetic);

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

            out << "Total files processed: " << totalCount << std::endl;
            printMetricSummary("line count", line);
            printMetricSummary("word count", word);
            printMetricSummary("character count", character);
        }
    };

    // create a results object for this node
    Results nodeResults;

    // use openmp to parallelize the processing of files for this node
    #pragma omp parallel
    {
        Results threadResults;

        #pragma omp for nowait
        for (int i = start; i < end; ++i) {
            // process the file to get line count, word count, and character count
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

            // the open file is formatted in a csv like this
            // name, bytes, words, lines
            // we need to parse the line to get the counts
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string name;
                int bytes = 0;
                int words = 0;
                int lines = 0;
                if (std::getline(ss, name, ',') && ss >> bytes && ss.ignore() && ss >> words && ss.ignore() && ss >> lines) {
                    threadResults.addFileResult(name, lines, words, bytes);
                }
            }
            file.close();
        }

        // merge once per thread to reduce contention
        #pragma omp critical
        {
            nodeResults.mergeFrom(threadResults);
        }
    }

    // define median variable for each node to store the median values for line count, word count, and char count
    MedianValue medianLineCount;
    MedianValue medianWordCount;
    MedianValue medianCharCount;

    // define a quick select function to calculate the median of a vector of integers
    auto partition = [](std::vector<int>& nums, int left, int right) -> int {
        int pivot = nums[static_cast<std::size_t>(right)];
        int i = left;
        for (int j = left; j < right; ++j) {
            if (nums[static_cast<std::size_t>(j)] < pivot) {
                std::swap(nums[static_cast<std::size_t>(i)], nums[static_cast<std::size_t>(j)]);
                ++i;
            }
        }
        std::swap(nums[static_cast<std::size_t>(i)], nums[static_cast<std::size_t>(right)]);
        return i;
    };

    auto quickSelect = [&partition](std::vector<int>& nums, int k) -> int {
        int left = 0;
        int right = static_cast<int>(nums.size()) - 1;
        while (left <= right) {
            int pivotIndex = partition(nums, left, right);
            if (pivotIndex == k) {
                return nums[static_cast<std::size_t>(pivotIndex)];
            } else if (pivotIndex < k) {
                left = pivotIndex + 1;
            } else {
                right = pivotIndex - 1;
            }
        }
        return -1; // should never reach here
    };

    auto computeMedianValues = [&quickSelect](const std::vector<int>& values) -> MedianValue {
        MedianValue median;
        if (values.empty()) {
            return median;
        }

        std::vector<int> upperValues = values;
        int upperIndex = static_cast<int>(upperValues.size() / 2);
        median.upperMiddle = quickSelect(upperValues, upperIndex);

        if (values.size() % 2 == 1) {
            median.arithmetic = static_cast<double>(median.upperMiddle);
        } else {
            std::vector<int> lowerValues = values;
            int lowerMiddle = quickSelect(lowerValues, upperIndex - 1);
            median.arithmetic = (static_cast<double>(lowerMiddle) + static_cast<double>(median.upperMiddle)) / 2.0;
        }

        return median;
    };

    auto gatherValuesToReducer = [rank, size](const std::vector<int>& localValues, int reducerRank) {
        int localSize = static_cast<int>(localValues.size());

        std::vector<int> recvCounts;
        if (rank == reducerRank) {
            recvCounts.resize(static_cast<std::size_t>(size), 0);
        }

        MPI_Gather(
            &localSize,
            1,
            MPI_INT,
            rank == reducerRank ? recvCounts.data() : nullptr,
            1,
            MPI_INT,
            reducerRank,
            MPI_COMM_WORLD
        );

        std::vector<int> displacements;
        int totalCount = 0;
        if (rank == reducerRank) {
            displacements.resize(static_cast<std::size_t>(size), 0);
            for (int i = 0; i < size; ++i) {
                displacements[static_cast<std::size_t>(i)] = totalCount;
                totalCount += recvCounts[static_cast<std::size_t>(i)];
            }
        }

        std::vector<int> gatheredValues;
        if (rank == reducerRank) {
            gatheredValues.resize(static_cast<std::size_t>(totalCount));
        }

        MPI_Gatherv(
            localSize > 0 ? localValues.data() : nullptr,
            localSize,
            MPI_INT,
            rank == reducerRank ? gatheredValues.data() : nullptr,
            rank == reducerRank ? recvCounts.data() : nullptr,
            rank == reducerRank ? displacements.data() : nullptr,
            MPI_INT,
            reducerRank,
            MPI_COMM_WORLD
        );

        return gatheredValues;
    };

    // Gather each metric to its designated reducer rank.
    std::vector<int> allLineCounts = gatherValuesToReducer(nodeResults.lineCounts(), 1);
    std::vector<int> allWordCounts = gatherValuesToReducer(nodeResults.wordCounts(), 2);
    std::vector<int> allCharCounts = gatherValuesToReducer(nodeResults.charCounts(), 3);

    // We no longer need local raw vectors after global median inputs are gathered.
    nodeResults.clearRawCounts();

    if (rank == 1 && !allLineCounts.empty()) {
        medianLineCount = computeMedianValues(allLineCounts);
    }
    if (rank == 2 && !allWordCounts.empty()) {
        medianWordCount = computeMedianValues(allWordCounts);
    }
    if (rank == 3 && !allCharCounts.empty()) {
        medianCharCount = computeMedianValues(allCharCounts);
    }

    auto broadcastMedian = [](MedianValue& median, int reducerRank) {
        MPI_Bcast(&median.upperMiddle, 1, MPI_INT, reducerRank, MPI_COMM_WORLD);
        MPI_Bcast(&median.arithmetic, 1, MPI_DOUBLE, reducerRank, MPI_COMM_WORLD);
    };

    // Broadcast medians from each reducer so every rank has a complete median set.
    broadcastMedian(medianLineCount, 1);
    broadcastMedian(medianWordCount, 2);
    broadcastMedian(medianCharCount, 3);

    if (rank == 1) {
        std::cout << "Median line count (upper-middle): " << medianLineCount.upperMiddle
                  << ", arithmetic: " << medianLineCount.arithmetic << std::endl;
    }
    if (rank == 2) {
        std::cout << "Median word count (upper-middle): " << medianWordCount.upperMiddle
                  << ", arithmetic: " << medianWordCount.arithmetic << std::endl;
    }
    if (rank == 3) {
        std::cout << "Median char count (upper-middle): " << medianCharCount.upperMiddle
                  << ", arithmetic: " << medianCharCount.arithmetic << std::endl;
    }

    // set median values to results struct for this node
    nodeResults.setMedians(medianLineCount, medianWordCount, medianCharCount);

    // gather each node results to rank 0 and merge them into a final results object
    Results finalResults;
    if (rank == 0) {
        finalResults.mergeFrom(nodeResults);
        for (int i = 1; i < size; ++i) {
            Results recvResults;

            recvResults.recvFromRank(i, 100, 101, 110);
            finalResults.mergeFrom(recvResults);
        }
    } else {
        nodeResults.sendToRank(0, 100, 101, 110);
    }

    // finalize mpi
    MPI_Finalize();

    // output final results only on rank 0
    if (rank == 0) {
        finalResults.printSummary(std::cout);
    }

    return 0;
}