# NVXK

NVXK là dự án C++ thử nghiệm cộng vector và nhân ma trận trên CPU. Dự án so sánh nhiều cách cài đặt GEMM, từ vòng lặp scalar đến AVX2/FMA, chia khối và OpenMP.

## Tính năng

- Nhân ma trận vuông và chữ nhật: `A[M×K] * B[K×N] = C[M×N]`.
- Runtime dispatch AVX2/FMA với fallback scalar khi CPU không hỗ trợ.
- Các lựa chọn GEMM: naive, ikj, AVX2, blocked, OpenMP và tự chọn.
- GEMM `int32_t` vào `int64_t` qua `XK_gemm_i32_i64`.
- Cộng vector `float`/`double` bằng `vector_add_avx2`.
- Benchmark có kiểm tra kết quả, đo thời gian và xuất CSV; block-size tuning có thể tự sinh ứng viên.
- Console app nhập hai ma trận chữ nhật.

## Build (Windows / MSYS2 UCRT64)

Mở PowerShell tại thư mục dự án:

```powershell
Set-Location D:\gun\NVXK
$env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\usr\bin;$env:PATH"

g++ -std=c++11 -O3 -Wall -fopenmp -I. main.cpp nvxk.cpp -o nvxk.exe
g++ -std=c++11 -O3 -Wall -fopenmp -I. my_app.cpp nvxk.cpp -o my_app.exe
```

## Dùng thử

Chạy GEMM benchmark trên ma trận chữ nhật:

```powershell
.\nvxk.exe --shape 128,256,64 --algorithm all --runs 3
```

Benchmark cộng vector `float` và `double`:

```powershell
.\nvxk.exe --size 1000000 --algorithm vector-add --runs 5 --csv vector-results.csv
```

Chạy bộ kiểm thử:

```powershell
g++ -std=c++11 -O2 -Wall -I. nvxk_test.cpp nvxk.cpp -o nvxk-test.exe
.\nvxk-test.exe
```

Chạy ứng dụng nhập ma trận:

```powershell
.\my_app.exe
```

Nhập lần lượt `M`, `K`, `N`, tiếp theo là `M` hàng của A (mỗi hàng `K` số) và `K` hàng của B (mỗi hàng `N` số).

## API GEMM mẫu

```cpp
#include <nvxk.h>

float A[] = {1, 2, 3, 4, 5, 6};       // 2x3
float B[] = {7, 8, 9, 10, 11, 12};    // 3x2
float C[4];
XK_gemm(A, B, C, 2, 3, 2);            // C = {58, 64, 139, 154}
```

Xem [README.txt.md](README.txt.md) để biết đầy đủ API, các tùy chọn benchmark, định dạng CSV và lưu ý về độ chính xác/tràn số.
