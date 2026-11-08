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

bool verify_result(const vector<float>& C_ref, const vector<float>& C_opt,size_t N, const string& name){
    const float epsilon = 1e-2f; 
    
    for(size_t i=0; i < N*N; ++i) {
        float diff = fabs(C_ref[i] - C_opt[i]);
        if (!isfinite(C_ref[i]) || !isfinite(C_opt[i]) || diff > epsilon) {
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

struct Options {
    vector<size_t> sizes = {64, 67, 128, 130, 256, 512, 1024};
    vector<int> thread_counts = {1, 2, 4};
    int runs = 5;
    string algorithm = "all";
    string csv_path;
};

void print_help(const char* program) {
    cout << "Usage: " << program << " [options]\n"
         << "  --size N              Run one matrix size\n"
         << "  --sizes N,N,...       Run several matrix sizes\n"
         << "  --runs N              Number of timed samples (default: 5)\n"
         << "  --threads N,N,...     OpenMP thread counts (default: 1,2,4)\n"
         << "  --algorithm NAME      all, naive, ikj, avx2, blocked, omp, or auto\n"
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
        } else if (arg == "--size" || arg == "--sizes" || arg == "--runs" ||
                   arg == "--threads" || arg == "--algorithm" || arg == "--csv") {
            if (i + 1 >= argc) throw invalid_argument("missing value for " + arg);
            string value = argv[++i];
            if (arg == "--size") {
                options.sizes = parse_sizes(value);
                if (options.sizes.size() != 1) throw invalid_argument("--size expects one value");
            } else if (arg == "--sizes") {
                options.sizes = parse_sizes(value);
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
            }
        } else {
            throw invalid_argument("unknown option: " + arg);
        }
    }
    if (options.algorithm != "all" && options.algorithm != "naive" &&
        options.algorithm != "ikj" && options.algorithm != "avx2" &&
        options.algorithm != "blocked" && options.algorithm != "omp" &&
        options.algorithm != "auto") {
        throw invalid_argument("algorithm must be all, naive, ikj, avx2, blocked, omp, or auto");
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
        csv << "size,algorithm,threads,median_seconds,gflops\n";
    }

    cout << left << setw(8) << "N" << setw(16) << "Algorithm"
         << setw(12) << "Threads" << setw(18) << "Seconds"
         << setw(14) << "GFLOPS" << "\n";
    cout << string(68, '-') << "\n";

    const vector<string> algorithms = {"naive", "ikj", "avx2", "blocked", "omp", "auto"};
    for(size_t N : options.sizes){
        vector<float> A(N * N);
        vector<float> B(N * N);
        vector<float> C_naive(N * N, 0.0f);

        for(size_t i=0; i < N*N; ++i) {
            A[i] = static_cast<float>(rand()) / RAND_MAX;
            B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        XK_naive(A.data(), B.data(), C_naive.data(), N);
        vector<float> output(N * N);
        const double ops = 2.0 * static_cast<double>(N) * N * N;

        for (const string& algorithm : algorithms) {
            if (options.algorithm != "all" && options.algorithm != algorithm) continue;
            vector<int> thread_counts = algorithm == "omp" ? options.thread_counts : vector<int>(1, 1);
            for (int thread_count : thread_counts) {
                auto calculate = [&]() {
                    if (algorithm == "naive") XK_naive(A.data(), B.data(), output.data(), N);
                    else if (algorithm == "ikj") XK_ikj(A.data(), B.data(), output.data(), N);
                    else if (algorithm == "avx2") XK_avx2(A.data(), B.data(), output.data(), N);
                    else if (algorithm == "blocked") XK_blocked(A.data(), B.data(), output.data(), N);
                    else if (algorithm == "omp") XK_omp(A.data(), B.data(), output.data(), N, thread_count);
                    else XK_auto(A.data(), B.data(), output.data(), N);
                };

                double seconds = measure_time(calculate, options.runs);
                if (seconds <= 0.0 || !isfinite(seconds)) {
                    cerr << "[FAILED] Invalid timing for " << algorithm << " at N=" << N << "\n";
                    return EXIT_FAILURE;
                }
                if (!verify_result(C_naive, output, N, algorithm)) {
                    return EXIT_FAILURE;
                }

                string result_name = algorithm;
                if (!has_avx2_fma && (algorithm == "avx2" || algorithm == "blocked")) {
                    result_name += "-fallback";
                } else if (algorithm == "omp") {
                    result_name = has_avx2_fma ? "omp-avx2" : "omp-ikj";
                } else if (algorithm == "auto") {
                    result_name = has_avx2_fma ? "auto-avx2" : "auto-ikj";
                }

                double gflops = (ops / seconds) * 1e-9;
                cout << left << setw(8) << N << setw(16) << result_name
                     << setw(12) << thread_count << setw(18) << fixed << setprecision(6) << seconds
                     << setw(14) << setprecision(2) << gflops << "\n";
                if (csv) {
                    csv << N << "," << result_name << "," << thread_count << ","
                        << setprecision(10) << seconds << "," << setprecision(6) << gflops << "\n";
                }
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