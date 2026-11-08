#include <cmath>
#include <iostream>
#include <limits>
#include <new>
#include <vector>
#include <nvxk.h>

using namespace std;

bool read_matrix(const char* name, vector<float>& matrix, size_t n) {
    cout << "Nhap ma tran " << name << " theo tung hang:\n";
    for (size_t i = 0; i < n; ++i) {
        cout << "Hang " << i + 1 << ": ";
        for (size_t j = 0; j < n; ++j) {
            float value;
            if (!(cin >> value) || !isfinite(value)) {
                cerr << "Du lieu khong hop le. Hay nhap cac so huu han.\n";
                return false;
            }
            matrix[i * n + j] = value;
        }
    }
    return true;
}

int main() {
    long long dimension;
    cout << "Nhap kich thuoc ma tran vuong N (N x N): ";
    if (!(cin >> dimension) || dimension <= 0) {
        cerr << "Kich thuoc phai la so nguyen duong.\n";
        return 1;
    }

    if (static_cast<unsigned long long>(dimension) >
        numeric_limits<size_t>::max()) {
        cerr << "Kich thuoc qua lon.\n";
        return 1;
    }

    const size_t n = static_cast<size_t>(dimension);
    const size_t max_elements = vector<float>().max_size();
    if (n > max_elements / n) {
        cerr << "Kich thuoc ma tran qua lon.\n";
        return 1;
    }

    const size_t element_count = n * n;
    try {
        vector<float> a(element_count);
        vector<float> b(element_count);
        vector<float> c(element_count);

        if (!read_matrix("A", a, n) || !read_matrix("B", b, n)) {
            return 1;
        }

        XK_auto(a.data(), b.data(), c.data(), n);

        cout << "Ket qua A x B:\n";
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                cout << c[i * n + j] << (j + 1 == n ? '\n' : ' ');
            }
        }
    } catch (const bad_alloc&) {
        cerr << "Khong du bo nho de tao cac ma tran nay.\n";
        return 1;
    }

    return 0;
}
