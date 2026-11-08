# NVXK

NVXK là dự án C++ thử nghiệm và so sánh các cách cộng vector, nhân ma trận vuông trên CPU. Dự án tập trung vào thay đổi thứ tự truy cập bộ nhớ, SIMD AVX2/FMA, chia khối và chạy đa luồng bằng OpenMP.

Tên tiền tố `XK` trong các hàm nhân ma trận được dùng để ghi nhận người đóng góp quan trọng của dự án.

## Mục tiêu

Dự án trả lời các câu hỏi:

- Các cách cài đặt phép nhân ma trận có cho kết quả đúng như nhau không?
- Thay đổi thứ tự vòng lặp có ảnh hưởng tốc độ không?
- AVX2/FMA và OpenMP cải thiện hiệu năng như thế nào?
- Chia ma trận thành khối có giúp ích trên kích thước và CPU đang thử nghiệm không?

Kết quả phụ thuộc vào CPU, compiler, cờ biên dịch và tải của hệ thống. Benchmark là công cụ đo trên máy hiện tại, không phải cam kết tốc độ trên mọi máy.

## Cấu trúc dự án

| Tệp | Chức năng |
|---|---|
| `main.cpp` | CLI benchmark: đọc tuỳ chọn, tạo ma trận thử, kiểm tra kết quả, đo thời gian, in GFLOPS và tuỳ chọn xuất CSV. |
| `nvxk.cpp` | Phần cài đặt các hàm cộng vector và nhân ma trận. |
| `nvxk.h` | Khai báo công khai các hàm để benchmark hoặc chương trình C++ khác gọi. |
| `my_app.cpp` | Ứng dụng console cho nhập hai ma trận vuông, nhân chúng bằng `XK_auto` và in kết quả. |
| `makefile` | Các mục tiêu build benchmark, ví dụ và dọn các executable do Makefile tạo. |
| `nvxk-portable.exe` | Executable benchmark build tĩnh nếu tệp này được phân phối kèm bản phát hành. |

## Các hàm

Các hàm nhân ma trận nhận hai ma trận vuông đầu vào `A` và `B`, ghi tích `A × B` vào `C`, với mỗi ma trận có kích thước `N × N`.

### `vector_add_avx2`

```cpp
void vector_add_avx2(const float* a, const float* b, float* c, size_t n);
```

Cộng hai mảng `a` và `b`, mỗi mảng có `n` phần tử, rồi ghi kết quả vào `c`. AVX2 xử lý từng nhóm tối đa 8 số `float`; các phần tử cuối không đủ thành nhóm 8 được xử lý tuần tự. Hàm này hiện có trong thư viện nhưng chưa được benchmark trong `main.cpp`.

### `XK_naive`

```cpp
void XK_naive(const float* A, const float* B, float* C, size_t N);
```

Phiên bản nhân ma trận cơ bản dùng thứ tự vòng lặp `i-j-k`. Benchmark dùng hàm này làm kết quả tham chiếu để so sánh tính đúng đắn. Cách duyệt cột của `B` có thể không tận dụng tốt tính liên tục trong bộ nhớ hàng.

### `XK_ikj`

```cpp
void XK_ikj(const float* A, const float* B, float* C, size_t N);
```

Phiên bản tuần tự dùng thứ tự vòng lặp `i-k-j`. Hàm khởi tạo `C` bằng 0, sau đó cập nhật lần lượt theo hàng. Đây là mốc để khảo sát ảnh hưởng của thứ tự truy cập dữ liệu mà không viết intrinsic AVX2 trực tiếp.

### `XK_avx2`

```cpp
void XK_avx2(const float* A, const float* B, float* C, size_t N);
```

Phiên bản đơn luồng dùng AVX2 và FMA nếu CPU hỗ trợ. Mỗi lượt vector xử lý 8 giá trị `float`; phần dư của hàng được xử lý tuần tự. Nếu CPU không hỗ trợ, hàm tự chạy `XK_ikj` để tránh lệnh AVX2 không hợp lệ.

### Phát hiện và tự chọn backend

```cpp
bool XK_has_avx2_fma();
void XK_auto(const float* A, const float* B, float* C, size_t N);
```

`XK_has_avx2_fma()` kiểm tra khả năng AVX2/FMA của CPU khi chạy. `XK_auto()` tự dùng AVX2/FMA nếu có, nếu không thì dùng `XK_ikj`. Đây là lựa chọn nên dùng trong ứng dụng để tự thích ứng với máy người dùng.

### `XK_avx2_omp`

```cpp
void XK_avx2_omp(
    const float* A,
    const float* B,
    float* C,
    size_t N,
    int thread_count
);
```

Phiên bản đa luồng chia các hàng của `C` bằng OpenMP. Nếu CPU hỗ trợ AVX2/FMA, mỗi luồng dùng AVX2/FMA; nếu không, chương trình dùng bản i-k-j đa luồng tương thích. `thread_count` nhỏ hơn 1 được đổi thành 1. Benchmark mặc định thử 1, 2 và 4 luồng; có thể chỉ định các giá trị khác bằng CLI.

### `XK_omp`

```cpp
void XK_omp(const float* A, const float* B, float* C, size_t N, int thread_count);
```

Tên gọi tiện dụng cho backend đa luồng có tự chọn AVX2/FMA hoặc fallback scalar/OpenMP.

Nhiều luồng không luôn nhanh hơn: ma trận nhỏ có thể bị ảnh hưởng bởi chi phí điều phối luồng; số luồng hữu ích cũng phụ thuộc vào CPU.

### `XK_blocked`

```cpp
void XK_blocked(const float* A, const float* B, float* C, size_t N);
```

Phiên bản chia các vòng lặp ma trận thành khối với `BLOCK = 64`, sau đó dùng AVX2/FMA trong khối nếu CPU hỗ trợ. Nếu không hỗ trợ, hàm tự dùng `XK_ikj`. Các phần tử ở cạnh ma trận được xử lý bằng vòng lặp dư. Block size hiện đang cố định trong mã nguồn và chưa chắc là tối ưu cho mọi CPU.

## Biểu diễn ma trận và điều kiện gọi hàm

Ma trận được lưu dạng row-major: các phần tử của một hàng nằm liên tiếp trong bộ nhớ. Ví dụ ma trận:

```text
1 2
3 4
```

được lưu dưới dạng `{1, 2, 3, 4}`.

Với ma trận kích thước `N × N`, `A`, `B` và `C` cần vùng nhớ chứa ít nhất `N * N` phần tử `float`. `C` là vùng nhận kết quả.

Các hàm công khai tự kiểm tra AVX2/FMA trước khi gọi mã SIMD. Nếu CPU không hỗ trợ, những hàm này chạy fallback tương thích. Chương trình vẫn có thể chạy trên CPU không hỗ trợ AVX2/FMA; hiệu năng khi đó có thể thấp hơn.

## Yêu cầu build

- Windows với MSYS2 UCRT64 GCC (hoặc GCC tương thích).
- GCC hỗ trợ OpenMP và GNU function target attributes (ví dụ MSYS2 UCRT64 GCC).
- AVX2/FMA chỉ cần thiết để dùng backend SIMD nhanh; CPU không hỗ trợ vẫn chạy được nhờ fallback.

Cờ chính:

- `-O3`: bật tối ưu hoá compiler.
- `-fopenmp`: bật OpenMP và liên kết runtime OpenMP.
- `-I.`: thêm thư mục hiện tại để tìm `nvxk.h`.
- Các hàm SIMD được biên dịch riêng với target AVX2/FMA; không bật `-mavx2` hoặc `-mfma` toàn chương trình để giữ khả năng fallback trên CPU cũ.

## Build trên Windows bằng PowerShell

Mở PowerShell tại thư mục dự án. Nếu MSYS2 được cài ở đường dẫn mặc định:

```powershell
cd D:\gun\NVXK
$env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\usr\bin;$env:PATH"
```

Build benchmark:

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -I. -O3 -fopenmp -Wall main.cpp nvxk.cpp -o nvxk.exe
```

Chỉ chạy executable nếu build thành công:

```powershell
if ($LASTEXITCODE -eq 0) { .\nvxk.exe }
```

Build ứng dụng nhập ma trận:

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -I. -O3 -fopenmp -Wall my_app.cpp nvxk.cpp -o my_app.exe
if ($LASTEXITCODE -eq 0) { .\my_app.exe }
```

Chạy ứng dụng, nhập kích thước `N`, sau đó nhập lần lượt từng hàng của ma trận A và B. Ví dụ với `N = 2`, nhập các hàng `1 2`, `3 4` cho A và `5 6`, `7 8` cho B; kết quả là `19 22` và `43 50`. Ứng dụng hiện nhận ma trận vuông.

Để tạo một bản ứng dụng liên kết tĩnh nhằm gửi cho máy Windows khác:

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -I. -O3 -fopenmp -Wall -static my_app.cpp nvxk.cpp -o my_app-portable.exe
```

Gửi `my_app-portable.exe`; người nhận mở file và nhập ma trận khi chương trình hỏi. `-static` đóng các thư viện GCC/OpenMP cần thiết vào executable, nên không cần cài MSYS2/GCC trên máy nhận. Nên thử file trên máy Windows khác trước khi gửi. `my_app.cpp` và `main.cpp` đều có hàm `main()`, vì vậy build chúng thành hai executable riêng; không đưa cả hai vào cùng một lệnh build.

## Build bằng Makefile

Nếu `make` và GCC MSYS2 đã có trong `PATH`:

```powershell
make
```

Các mục tiêu:

```powershell
make nvxk.exe
make my_app.exe
make clean
```

Makefile hiện dùng `CXX = g++`; hãy đảm bảo lệnh `g++` trỏ tới compiler MSYS2 UCRT64 phù hợp. Nếu máy đang chọn một MinGW cũ khác, dùng các lệnh build trực tiếp ở trên hoặc chỉnh `PATH` trước khi gọi `make`.

## Chạy benchmark CLI

Chạy tất cả thuật toán với danh sách kích thước mặc định `64, 67, 128, 130, 256, 512, 1024`:

```powershell
.\nvxk.exe
```

Xem trợ giúp:

```powershell
.\nvxk.exe --help
```

Tuỳ chọn hỗ trợ:

| Tuỳ chọn | Mô tả | Mặc định |
|---|---|---|
| `--size N` | Chạy một kích thước ma trận. | — |
| `--sizes N,N,...` | Chạy nhiều kích thước, phân cách bằng dấu phẩy. | `64,67,128,130,256,512,1024` |
| `--runs N` | Số mẫu thời gian; phải là số nguyên dương. | `5` |
| `--threads N,N,...` | Số luồng thử cho thuật toán `omp`; mỗi số phải dương. | `1,2,4` |
| `--algorithm NAME` | Chọn `all`, `naive`, `ikj`, `avx2`, `blocked`, `omp` hoặc `auto`. | `all` |
| `--csv FILE` | Ghi kết quả dạng CSV vào tệp chỉ định. | Không ghi CSV |
| `--help` | In hướng dẫn lệnh. | — |

Ví dụ: chạy kích thước 512 với OpenMP 1, 2 và 4 luồng, dùng 5 mẫu và xuất CSV:

```powershell
.\nvxk.exe --size 512 --algorithm omp --threads 1,2,4 --runs 5 --csv omp-results.csv
```

Chạy các thuật toán đơn luồng tại vài kích thước:

```powershell
.\nvxk.exe --sizes 128,256,512,1024 --algorithm all --runs 5 --csv results.csv
```

Khi `--algorithm all`, chương trình chạy tất cả thuật toán; tuỳ chọn `--threads` áp dụng cho `omp`. Các thuật toán còn lại được báo với cột số luồng bằng 1.

## Kết quả benchmark

Benchmark thực hiện một lượt làm nóng, sau đó đo nhiều mẫu. Nếu một mẫu ngắn hơn khoảng 0.05 giây, hàm được lặp nhiều lần trong mẫu đó để tăng độ phân giải của phép đo. Thời gian mỗi lần gọi được tính từ tổng thời gian chia cho số lần lặp; kết quả báo cáo là trung vị các mẫu.

Chương trình kiểm tra từng đầu ra so với `XK_naive`, với sai số tuyệt đối tối đa `0.01`. Nếu kết quả có NaN/vô cực hoặc vượt sai số, chương trình báo `[FAILED]` và thoát với mã lỗi.

Các cột kết quả:

- `Seconds`: thời gian trung vị cho một lần gọi hàm, tính bằng giây.
- `GFLOPS`: số tỷ phép tính dấu phẩy động ước lượng mỗi giây. Benchmark dùng công thức `2 * N^3` phép tính cho phép nhân hai ma trận vuông.

CSV có các cột:

```text
size,algorithm,threads,median_seconds,gflops
```

Số GFLOPS cao hơn thường nghĩa là nhiều phép tính được thực hiện mỗi giây trong lần đo đó. Hãy so sánh cùng kích thước trên cùng máy. `XK_naive` có thứ tự vòng lặp khác với `XK_ikj` và các bản AVX2, nên chênh lệch giữa chúng phản ánh cả thứ tự truy cập dữ liệu lẫn cách triển khai; không nên quy toàn bộ khác biệt cho SIMD. Đầu ra cho biết AVX2/FMA có sẵn hay không. Khi không có, nhãn `avx2-fallback`, `blocked-fallback` hoặc `omp-ikj` cho biết kết quả được đo từ fallback tương thích.

## Dùng NVXK trong ứng dụng C++ khác

Ứng dụng include header và build cùng phần cài đặt `nvxk.cpp`:

```cpp
#include <iostream>
#include <vector>
#include <nvxk.h>

int main() {
    const size_t N = 2;
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};
    std::vector<float> C(N * N);

    XK_auto(A.data(), B.data(), C.data(), N);

    for (float value : C) {
        std::cout << value << " ";
    }
}
```

Build ứng dụng cùng `nvxk.cpp`:

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -I. -O3 -fopenmp -Wall my_app.cpp nvxk.cpp -o my_app.exe
```

Với đầu vào trên, kết quả mong đợi là:

```text
19 22 43 50
```

## Bản executable portable

Có thể tạo executable liên kết tĩnh để người dùng không cần cài GCC/MSYS2:

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -I. -O3 -fopenmp -Wall -static main.cpp nvxk.cpp -o nvxk-portable.exe
```

Executable tự chọn AVX2/FMA hoặc fallback khi chạy, nên không bắt buộc CPU có AVX2/FMA. Nên thử trên máy mục tiêu trước khi phát hành.

## Giới hạn hiện tại

- `XK_blocked` dùng block size cố định là 64; chưa có tuỳ chọn CLI để điều chỉnh hoặc tự dò block size.
- CLI phát hiện hỗ trợ AVX2/FMA nhưng chưa tự hiển thị tên model CPU/compiler.
- Fallback cho CPU cũ dùng i-k-j tuần tự hoặc OpenMP; hiệu năng có thể thấp hơn AVX2/FMA.
- `vector_add_avx2` chưa có bài kiểm tra và benchmark riêng.
- OpenMP có chi phí điều phối; dùng nhiều luồng có thể chậm hơn với ma trận nhỏ.
- `--runs` là số mẫu đo, không phải số lần gọi hàm trong một mẫu. Số lần gọi nội bộ được tự điều chỉnh để đạt thời lượng mẫu tối thiểu.

## Gợi ý tái lập kết quả

Khi ghi kết quả vào báo cáo, lưu lại:

- model CPU và số nhân/luồng phần cứng;
- phiên bản GCC;
- lệnh/cờ biên dịch;
- kích thước ma trận và số mẫu;
- CSV đầu ra;
- trạng thái các ứng dụng khác đang chạy trong lúc benchmark.

Chạy thử nhiều lần và báo cáo xu hướng hoặc trung vị giữa các lần chạy. Không xem một lần đo là tốc độ cố định của thuật toán trên mọi máy.
