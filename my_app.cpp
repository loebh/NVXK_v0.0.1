#include <cmath>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#include <nvxk.h>

using namespace std;

bool read_matrix(const char* name, vector<float>& matrix,
                 size_t rows, size_t columns) {
    cout << "Nhap ma tran " << name << " theo tung hang:\n";
    for (size_t i = 0; i < rows; ++i) {
        cout << "Hang " << i + 1 << ": ";
        for (size_t j = 0; j < columns; ++j) {
            float value;
            if (!(cin >> value) || !isfinite(value)) {
                cerr << "Du lieu khong hop le. Hay nhap cac so huu han.\n";
                return false;
            }
            matrix[i * columns + j] = value;
        }
    }
    return true;
}

bool read_dimension(const char* name, size_t& value) {
    long long input;
    cout << name;
    if (!(cin >> input) || input <= 0 ||
        static_cast<unsigned long long>(input) >
            numeric_limits<size_t>::max()) {
        cerr << "Kich thuoc phai la so nguyen duong hop le.\n";
        return false;
    }
    value = static_cast<size_t>(input);
    return true;
}

size_t checked_count(size_t rows, size_t columns, size_t max_elements) {
    if (columns != 0 && rows > max_elements / columns) {
        throw length_error("Kich thuoc ma tran qua lon.");
    }
    return rows * columns;
}

int main() {
    size_t m, k, n;
    if (!read_dimension("Nhap so hang M cua A (M x K): ", m) ||
        !read_dimension("Nhap so cot K cua A / so hang cua B (K x N): ", k) ||
        !read_dimension("Nhap so cot N cua B (K x N): ", n)) {
        return 1;
    }

    const size_t max_elements = vector<float>().max_size();
    try {
        vector<float> a(checked_count(m, k, max_elements));
        vector<float> b(checked_count(k, n, max_elements));
        vector<float> c(checked_count(m, n, max_elements));

        if (!read_matrix("A", a, m, k) || !read_matrix("B", b, k, n)) {
            return 1;
        }

        XK_gemm(a.data(), b.data(), c.data(), m, k, n);

        cout << "Ket qua A x B (" << m << " x " << n << "):\n";
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                cout << c[i * n + j] << (j + 1 == n ? '\n' : ' ');
            }
        }
    } catch (const bad_alloc&) {
        cerr << "Khong du bo nho de tao cac ma tran nay.\n";
        return 1;
    } catch (const length_error& error) {
        cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
