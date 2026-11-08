#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <nvxk.h>

using namespace std ;
using namespace std::chrono;

bool verify_result(const vector<float>& C_ref, const vector<float>& C_opt,
                   size_t count, const string& name) {
    const double absolute_tolerance = 1e-5;
    const double relative_tolerance = 1e-4;
    
    for(size_t i=0; i < count; ++i) {
        const double diff = fabs(static_cast<double>(C_ref[i]) - C_opt[i]);
        const double scale = max(fabs(static_cast<double>(C_ref[i])),
                                 fabs(static_cast<double>(C_opt[i])));
        if (!isfinite(C_ref[i]) || !isfinite(C_opt[i]) ||
            diff > absolute_tolerance + relative_tolerance * scale) {
            cout << "\n[FAILED] "<< name << " - Sai so tai vi tri " << i << "!\n";
            cout << "Gia tri dung (Naive): " << C_ref[i] << "\n";
            cout << "Gia tri sai: " << C_opt[i] << "\n";
            cout << "Do chenh lech: " << diff << "\n";
            return false;
        }
    }
    return true;
}

template<typename Func>
double measure_time(Func f,int num_runs=5) {
    f();
    
    vector<double> times;
    const double min_sample_seconds = 0.05;
    for(int i=0; i < num_runs; ++i){
        size_t iterations = 1;
        double elapsed = 0.0;

        do {
            auto start = steady_clock::now();
            for(size_t j=0; j < iterations; ++j) f();
            auto end = steady_clock::now();
            elapsed = duration<double>(end-start).count();

            if (elapsed < min_sample_seconds) {
                if (iterations > numeric_limits<size_t>::max() / 2) {
                    break;
                }
                iterations *= 2;
            }
        } while (elapsed < min_sample_seconds);

        if (elapsed <= 0.0 || !isfinite(elapsed)) {
            return 0.0;
        }
        times.push_back(elapsed / iterations);
    }
    
    sort(times.begin(), times.end());
    return times[num_runs/2]; // Lấy giá trị trung vị
}

template<typename T>
bool benchmark_vector_add_type(size_t count, int runs, ofstream& csv) {
    vector<T> a, b, output, expected;
    try {
        a.resize(count);
        b.resize(count);
        output.resize(count);
        expected.resize(count);
    } catch (const exception& error) {
        cerr << "[FAILED] Cannot allocate vector-add buffers for size "
             << count << ": " << error.what() << "\n";
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<T>(static_cast<int>(i % 101) - 50) / static_cast<T>(7);
        b[i] = static_cast<T>(static_cast<int>(i % 67) - 33) / static_cast<T>(11);
        expected[i] = a[i] + b[i];
    }

    auto calculate = [&]() {
        vector_add_avx2(a.data(), b.data(), output.data(), count);
    };
    const double seconds = measure_time(calculate, runs);
    if (seconds <= 0.0 || !isfinite(seconds)) {
        cerr << "[FAILED] Invalid vector-add timing for size " << count << "\n";
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const double scale = max(1.0, fabs(static_cast<double>(expected[i])));
        const double tolerance = 4.0 * numeric_limits<T>::epsilon() * scale;
        if (!isfinite(output[i]) ||
            fabs(static_cast<double>(output[i]) - expected[i]) > tolerance) {
            cerr << "[FAILED] vector-add mismatch for "
                 << (sizeof(T) == sizeof(float) ? "float" : "double")
                 << " at index " << i << ": expected " << expected[i]
                 << ", got " << output[i] << "\n";
            return false;
        }
    }

    const double gigabytes_per_second =
        (3.0 * static_cast<double>(count) * sizeof(T) / seconds) / 1e9;
    const string type_name = sizeof(T) == sizeof(float) ? "float" : "double";
    cout << left << setw(12) << type_name << setw(14) << count
         << setw(18) << fixed << setprecision(6) << seconds
         << setw(16) << setprecision(3) << gigabytes_per_second << "\n";
    if (csv) {
        csv << "vector-add," << type_name << "," << count << ","
            << setprecision(10) << seconds << ","
            << setprecision(6) << gigabytes_per_second << "\n";
    }
    return true;
}

struct Options {
    vector<size_t> sizes = {64, 67, 128, 130, 256, 512, 1024};
    vector<size_t> block_sizes;
    vector<int> thread_counts = {1, 2, 4};
    int runs = 5;
    string algorithm = "all";
    string csv_path;
    size_t m = 0;
    size_t k = 0;
    size_t n = 0;
    bool has_shape = false;
    bool has_square_sizes = false;
    bool custom_block_sizes = false;
    bool tune_block_size = false;
};

void print_help(const char* program) {
    cout << "Usage: " << program << " [options]\n"
         << "  --size N              Run one matrix size\n"
         << "  --sizes N,N,...       Run several matrix sizes\n"
         << "  --shape M,K,N         Run rectangular A(MxK) * B(KxN)\n"
         << "  --runs N              Number of timed samples (default: 5)\n"
         << "  --threads N,N,...     OpenMP thread counts (default: 1,2,4)\n"
         << "  --tune-block-size     Benchmark block sizes for the blocked algorithm\n"
         << "  --block-sizes N,...   Override automatically generated tuning candidates\n"
         << "  --algorithm NAME      all, naive, ikj, avx2, blocked, omp, auto, or vector-add\n"
         << "                        vector-add benchmarks float and double array addition\n"
         << "  --csv FILE            Write measurements to a CSV file\n"
         << "  --help                Show this help\n";
}

vector<size_t> parse_sizes(const string& text) {
    vector<size_t> values;
    string token;
    stringstream input(text);
    while (getline(input, token, ',')) {
        if (token.empty() || token.find_first_not_of("0123456789") != string::npos) {
            throw invalid_argument("invalid matrix size: " + token);
        }
        size_t used = 0;
        unsigned long long value = stoull(token, &used);
        if (used != token.size() || value == 0 ||
            value > numeric_limits<size_t>::max() ||
            value > numeric_limits<size_t>::max() / value) {
            throw invalid_argument("invalid matrix size: " + token);
        }
        values.push_back(static_cast<size_t>(value));
    }
    if (values.empty()) throw invalid_argument("at least one matrix size is required");
    return values;
}

size_t parse_positive_size(const string& token, const string& description) {
    if (token.empty() || token.find_first_not_of("0123456789") != string::npos) {
        throw invalid_argument("invalid " + description + ": " + token);
    }
    size_t used = 0;
    unsigned long long value = stoull(token, &used);
    if (used != token.size() || value == 0 ||
        value > numeric_limits<size_t>::max()) {
        throw invalid_argument("invalid " + description + ": " + token);
    }
    return static_cast<size_t>(value);
}

void parse_shape(const string& text, Options& options) {
    stringstream input(text);
    string token;
    vector<size_t> dimensions;
    while (getline(input, token, ',')) {
        dimensions.push_back(parse_positive_size(token, "matrix dimension"));
    }
    if (dimensions.size() != 3) {
        throw invalid_argument("--shape expects M,K,N");
    }
    xk_detail::checked_element_count(dimensions[0], dimensions[1]);
    xk_detail::checked_element_count(dimensions[1], dimensions[2]);
    xk_detail::checked_element_count(dimensions[0], dimensions[2]);
    options.m = dimensions[0];
    options.k = dimensions[1];
    options.n = dimensions[2];
    options.has_shape = true;
}

vector<size_t> generated_block_sizes(size_t m, size_t k, size_t n) {
    const size_t largest_dimension = max(m, max(k, n));
    const size_t limit = min(largest_dimension, static_cast<size_t>(256));
    vector<size_t> candidates;
    for (size_t block_size = 8; block_size <= limit; block_size *= 2) {
        candidates.push_back(block_size);
    }
    if (candidates.empty()) candidates.push_back(min(largest_dimension, static_cast<size_t>(8)));
    return candidates;
}

vector<int> parse_threads(const string& text) {
    vector<int> values;
    string token;
    stringstream input(text);
    while (getline(input, token, ',')) {
        if (token.empty() || token.find_first_not_of("0123456789") != string::npos) {
            throw invalid_argument("invalid thread count: " + token);
        }
        size_t used = 0;
        long value = stol(token, &used);
        if (used != token.size() || value < 1 || value > numeric_limits<int>::max()) {
            throw invalid_argument("invalid thread count: " + token);
        }
        values.push_back(static_cast<int>(value));
    }
    if (values.empty()) throw invalid_argument("at least one thread count is required");
    return values;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help") {
            print_help(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (arg == "--tune-block-size") {
            options.tune_block_size = true;
        } else if (arg == "--size" || arg == "--sizes" || arg == "--shape" || arg == "--runs" ||
                   arg == "--threads" || arg == "--algorithm" || arg == "--csv" ||
                   arg == "--block-sizes") {
            if (i + 1 >= argc) throw invalid_argument("missing value for " + arg);
            string value = argv[++i];
            if (arg == "--size") {
                options.sizes = parse_sizes(value);
                options.has_square_sizes = true;
                if (options.sizes.size() != 1) throw invalid_argument("--size expects one value");
            } else if (arg == "--sizes") {
                options.sizes = parse_sizes(value);
                options.has_square_sizes = true;
            } else if (arg == "--shape") {
                parse_shape(value, options);
            } else if (arg == "--runs") {
                if (value.empty() || value.find_first_not_of("0123456789") != string::npos) {
                    throw invalid_argument("runs must be a positive integer");
                }
                size_t used = 0;
                long runs = stol(value, &used);
                if (used != value.size() || runs < 1 || runs > numeric_limits<int>::max()) {
                    throw invalid_argument("runs must be a positive integer");
                }
                options.runs = static_cast<int>(runs);
            } else if (arg == "--threads") {
                options.thread_counts = parse_threads(value);
            } else if (arg == "--algorithm") {
                options.algorithm = value;
            } else if (arg == "--csv") {
                options.csv_path = value;
            } else if (arg == "--block-sizes") {
                options.block_sizes = parse_sizes(value);
                options.custom_block_sizes = true;
                options.tune_block_size = true;
            }
        } else {
            throw invalid_argument("unknown option: " + arg);
        }
    }
    if (options.algorithm != "all" && options.algorithm != "naive" &&
        options.algorithm != "ikj" && options.algorithm != "avx2" &&
        options.algorithm != "blocked" && options.algorithm != "omp" &&
        options.algorithm != "auto" && options.algorithm != "vector-add") {
        throw invalid_argument("algorithm must be all, naive, ikj, avx2, blocked, omp, auto, or vector-add");
    }
    if (options.tune_block_size && options.algorithm != "all" &&
        options.algorithm != "blocked") {
        throw invalid_argument("--tune-block-size requires --algorithm all or blocked");
    }
    if (options.has_shape && options.has_square_sizes) {
        throw invalid_argument("--shape cannot be combined with --size or --sizes");
    }
    if (options.has_shape && options.algorithm == "vector-add") {
        throw invalid_argument("--shape cannot be used with --algorithm vector-add");
    }
    if (options.algorithm == "vector-add" && options.tune_block_size) {
        throw invalid_argument("block-size tuning cannot be used with vector-add");
    }
    return options;
}

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const exception& error) {
        cerr << "Argument error: " << error.what() << "\n";
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    cout << "========================================================\n";
    cout << "System & Compilation Info:\n";
    const bool has_avx2_fma = XK_has_avx2_fma();
    cout << "AVX2 + FMA : " << (has_avx2_fma ? "available" : "not available; using fallback") << "\n";
    cout << "Compiler   : AVX2 is selected at runtime when supported\n";
    cout << "========================================================\n\n";

    ofstream csv;
    if (!options.csv_path.empty()) {
        csv.open(options.csv_path.c_str());
        if (!csv) {
            cerr << "Cannot open CSV file: " << options.csv_path << "\n";
            return EXIT_FAILURE;
        }
        if (options.algorithm == "vector-add") {
            csv << "operation,type,size,median_seconds,gigabytes_per_second\n";
        } else {
            csv << "M,K,N,algorithm,threads,median_seconds,gflops\n";
        }
    }

    if (options.algorithm == "vector-add") {
        cout << left << setw(12) << "Type" << setw(14) << "Elements"
             << setw(18) << "Seconds" << setw(16) << "GB/s" << "\n";
        cout << string(60, '-') << "\n";
        for (size_t count : options.sizes) {
            if (!benchmark_vector_add_type<float>(count, options.runs, csv) ||
                !benchmark_vector_add_type<double>(count, options.runs, csv)) {
                return EXIT_FAILURE;
            }
        }
        if (csv && !csv.flush()) {
            cerr << "Failed while writing CSV file: " << options.csv_path << "\n";
            return EXIT_FAILURE;
        }
        cout << "\n[SUCCESS] All requested vector-add benchmarks passed.\n";
        return EXIT_SUCCESS;
    }

    cout << left << setw(18) << "M x K x N" << setw(16) << "Algorithm"
         << setw(12) << "Threads" << setw(18) << "Seconds"
         << setw(14) << "GFLOPS" << "\n";
    cout << string(68, '-') << "\n";

    const vector<string> algorithms = {"naive", "ikj", "avx2", "blocked", "omp", "auto"};
    vector<vector<size_t> > shapes;
    if (options.has_shape) {
        shapes.push_back(vector<size_t>{options.m, options.k, options.n});
    } else {
        for (size_t size : options.sizes) {
            shapes.push_back(vector<size_t>{size, size, size});
        }
    }

    for (const vector<size_t>& shape : shapes) {
        const size_t M = shape[0];
        const size_t K = shape[1];
        const size_t N = shape[2];
        const size_t a_count = xk_detail::checked_element_count(M, K);
        const size_t b_count = xk_detail::checked_element_count(K, N);
        const size_t c_count = xk_detail::checked_element_count(M, N);
        const string shape_name = to_string(M) + "x" + to_string(K) + "x" + to_string(N);
        vector<float> A;
        vector<float> B;
        vector<float> C_naive;
        vector<float> output;
        try {
            A.resize(a_count);
            B.resize(b_count);
            C_naive.assign(c_count, 0.0f);
            output.resize(c_count);
        } catch (const exception& error) {
            cerr << "[FAILED] Cannot allocate matrices for shape "
                 << shape_name << ": " << error.what() << "\n";
            return EXIT_FAILURE;
        }

        for(size_t i=0; i < a_count; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
        for(size_t i=0; i < b_count; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

        XK_gemm_naive(A.data(), B.data(), C_naive.data(), M, K, N);
        const double ops = 2.0 * static_cast<double>(M) *
                           static_cast<double>(K) * static_cast<double>(N);

        for (const string& algorithm : algorithms) {
            if (options.algorithm != "all" && options.algorithm != algorithm) continue;
            vector<int> thread_counts = algorithm == "omp" ? options.thread_counts : vector<int>(1, 1);
            vector<size_t> block_sizes =
                algorithm == "blocked" && options.tune_block_size
                    ? (options.custom_block_sizes ? options.block_sizes
                                                  : generated_block_sizes(M, K, N))
                    : vector<size_t>(1, 64);
            double best_block_seconds = numeric_limits<double>::max();
            size_t best_block_size = 0;
            for (int thread_count : thread_counts) {
                for (size_t block_size : block_sizes) {
                    auto calculate = [&]() {
                        if (algorithm == "naive")
                            XK_gemm_naive(A.data(), B.data(), output.data(), M, K, N);
                        else if (algorithm == "ikj")
                            XK_gemm_ikj(A.data(), B.data(), output.data(), M, K, N);
                        else if (algorithm == "avx2")
                            XK_gemm_avx2(A.data(), B.data(), output.data(), M, K, N);
                        else if (algorithm == "blocked")
                            XK_gemm_blocked(A.data(), B.data(), output.data(), M, K, N, block_size);
                        else if (algorithm == "omp")
                            XK_gemm_omp(A.data(), B.data(), output.data(), M, K, N, thread_count);
                        else XK_gemm(A.data(), B.data(), output.data(), M, K, N);
                    };

                    double seconds = measure_time(calculate, options.runs);
                    if (seconds <= 0.0 || !isfinite(seconds)) {
                        cerr << "[FAILED] Invalid timing for " << algorithm
                             << " at shape " << shape_name << "\n";
                        return EXIT_FAILURE;
                    }
                    if (!verify_result(C_naive, output, c_count, algorithm)) {
                        return EXIT_FAILURE;
                    }

                    string result_name = algorithm;
                    if (algorithm == "blocked" && options.tune_block_size) {
                        result_name += "-" + to_string(block_size);
                    }
                    if (!has_avx2_fma && (algorithm == "avx2" || algorithm == "blocked")) {
                        result_name += "-fallback";
                    } else if (algorithm == "omp") {
                        result_name = has_avx2_fma ? "omp-avx2" : "omp-ikj";
                    } else if (algorithm == "auto") {
                        result_name = has_avx2_fma ? "auto-avx2" : "auto-ikj";
                    }

                    double gflops = (ops / seconds) * 1e-9;
                    cout << left << setw(18) << shape_name << setw(16) << result_name
                         << setw(12) << thread_count << setw(18) << fixed << setprecision(6) << seconds
                         << setw(14) << setprecision(2) << gflops << "\n";
                    if (csv) {
                        csv << M << "," << K << "," << N << "," << result_name << "," << thread_count << ","
                            << setprecision(10) << seconds << "," << setprecision(6) << gflops << "\n";
                    }
                    if (algorithm == "blocked" && options.tune_block_size &&
                        seconds < best_block_seconds) {
                        best_block_seconds = seconds;
                        best_block_size = block_size;
                    }
                }
            }
            if (algorithm == "blocked" && options.tune_block_size) {
                cout << "[BEST] shape=" << shape_name << " block size=" << best_block_size
                     << " (" << fixed << setprecision(6) << best_block_seconds << " s)\n";
            }
        }
    }

    if (csv && !csv.flush()) {
        cerr << "Failed while writing CSV file: " << options.csv_path << "\n";
        return EXIT_FAILURE;
    }

    cout << "\n[SUCCESS] All requested benchmarks passed.\n";
    return EXIT_SUCCESS;
}