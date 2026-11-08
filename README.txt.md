# NVXK

NVXK là dự án C++ thử nghiệm và so sánh các cách cộng vector, nhân ma trận trên CPU. Dự án tập trung vào thay đổi thứ tự truy cập bộ nhớ, SIMD AVX2/FMA, chia khối và chạy đa luồng bằng OpenMP.

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
| `my_app.cpp` | Ứng dụng console nhập `M`, `K`, `N`, nhân A[M×K] với B[K×N] bằng `XK_gemm` rồi in kết quả. |
| `makefile` | Các mục tiêu build benchmark, ví dụ và dọn các executable do Makefile tạo. |
| `nvxk-portable.exe` | Executable benchmark build tĩnh nếu tệp này được phân phối kèm bản phát hành. |

## Các hàm

Các hàm `XK_*` cũ nhận ma trận vuông `N × N`. API `XK_gemm*` mới hỗ trợ phép nhân chữ nhật `A[M × K] × B[K × N] = C[M × N]`.

### GEMM chữ nhật và kiểu dữ liệu

`XK_gemm_naive` và `XK_gemm_ikj` là template, hỗ trợ `float`, `double` và số nguyên có dấu không lớn hơn `int`. `XK_gemm` tự chọn AVX2/FMA cho `float` và `double`, fallback về scalar nếu CPU không hỗ trợ; số nguyên dùng template scalar có kiểm tra tràn. Các ma trận lưu dạng row-major; `A` cần `M*K`, `B` cần `K*N`, và `C` cần `M*N` phần tử. Khi cần nhân nhanh số nguyên 32-bit, `XK_gemm_i32_i64` nhận hai đầu vào `int32_t` và ghi kết quả `int64_t`, không kiểm tra tràn bên trong vòng lặp. Người gọi phải bảo đảm từng tổng kết quả nằm trong miền `int64_t`. Các biến thể chia khối và OpenMP hiện dành cho `float`.

```cpp
float A[] = {1, 2, 3, 4, 5, 6};
float B[] = {7, 8, 9, 10, 11, 12};
float C[4];
XK_gemm(A, B, C, 2, 3, 2);
```

Kết quả là `{58, 64, 139, 154}`. Các API vuông cũ vẫn được giữ để tương thích với chương trình hiện tại.

Ví dụ số nguyên, đầu ra rộng hơn đầu vào:

```cpp
int32_t A[] = {50000, 50000};
int32_t B[] = {50000, -50000};
int64_t C[1];
XK_gemm_i32_i64(A, B, C, 1, 2, 1);
```

Kết quả là `0`. Mỗi tích được mở rộng sang `int64_t` trước khi nhân. Hàm không kiểm tra tổng bị tràn, vì vậy mọi tổng trung gian theo thứ tự tích lũy cũng như kết quả cuối phải biểu diễn được bằng `int64_t`.

### `vector_add_avx2`

```cpp
void vector_add_avx2(const float* a, const float* b, float* c, size_t n);
```

Cộng hai mảng `a` và `b`, mỗi mảng có `n` phần tử, rồi ghi kết quả vào `c`. Có overload `float` (8 phần tử mỗi vector) và `double` (4 phần tử mỗi vector); phần dư được xử lý tuần tự. Bài test bao gồm kích thước có phần dư và benchmark riêng chạy bằng `--algorithm vector-add`.

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

Phiên bản đa luồng chia các hàng của `C` bằng OpenMP. Nếu CPU hỗ trợ AVX2/FMA, mỗi luồng dùng AVX2/FMA; nếu không, chương trình dùng bản i-k-j đa luồng tương thích. `thread_count` nhỏ hơn 1 được đổi thành 1. Với ma trận vuông `N < 256`, hàm chạy đơn luồng để tránh overhead; benchmark mặc định thử 1, 2 và 4 luồng, có thể chỉ định các giá trị khác bằng CLI.

### `XK_omp`

```cpp
void XK_omp(const float* A, const float* B, float* C, size_t N, int thread_count);
```

Tên gọi tiện dụng cho backend đa luồng có tự chọn AVX2/FMA hoặc fallback scalar/OpenMP. API chữ nhật `XK_gemm_omp` cũng dùng OpenMP; với công việc nhỏ hơn xấp xỉ `M*K*N = 256^3`, hàm chuyển sang AVX2 đơn luồng.

Nhiều luồng không luôn nhanh hơn: ma trận nhỏ có thể bị ảnh hưởng bởi chi phí điều phối luồng; số luồng hữu ích cũng phụ thuộc vào CPU.

### `XK_blocked`

```cpp
void XK_blocked(const float* A, const float* B, float* C, size_t N);
void XK_blocked(const float* A, const float* B, float* C, size_t N, size_t block_size);
```

Phiên bản chia các vòng lặp ma trận thành khối, sau đó dùng AVX2/FMA trong khối nếu CPU hỗ trợ. Overload cũ mặc định dùng block size 64; overload mới nhận block size dương tùy chọn. `XK_gemm_blocked` cung cấp cùng phép tính cho ma trận chữ nhật. Các phần tử ở cạnh ma trận được xử lý an toàn; block size phù hợp vẫn phụ thuộc CPU.

## Biểu diễn ma trận và điều kiện gọi hàm

Ma trận được lưu dạng row-major: các phần tử của một hàng nằm liên tiếp trong bộ nhớ. Ví dụ ma trận:

```text
1 2
3 4
```

được lưu dưới dạng `{1, 2, 3, 4}`.

Với API vuông, `A`, `B` và `C` cần vùng nhớ chứa ít nhất `N * N` phần tử `float`. Với API chữ nhật, kích thước vùng nhớ lần lượt là `M*K`, `K*N` và `M*N`. `C` là vùng nhận kết quả.

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

Chạy ứng dụng, nhập lần lượt `M`, `K`, `N`, sau đó nhập `M` hàng của A (mỗi hàng `K` số) và `K` hàng của B (mỗi hàng `N` số). Ví dụ nhập:

```text
2
3
2
1 2 3
4 5 6
7 8
9 10
11 12
```

Kết quả là `58 64` và `139 154`.

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
make test
make clean
```

`make test` chạy bài kiểm tra GEMM chữ nhật, biên AVX2, cộng vector `double` và đầu ra số nguyên `int64_t`.

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
| `--shape M,K,N` | Chạy một phép nhân chữ nhật `A[M×K] * B[K×N]`. Không dùng cùng `--size/--sizes`. | — |
| `--runs N` | Số mẫu thời gian; phải là số nguyên dương. | `5` |
| `--threads N,N,...` | Số luồng thử cho thuật toán `omp`; mỗi số phải dương. | `1,2,4` |
| `--tune-block-size` | Tự sinh block-size ứng viên; mỗi ứng viên được warm-up rồi benchmark cho thuật toán `blocked`. Chỉ dùng với `all` hoặc `blocked`. | Tắt |
| `--block-sizes N,...` | Tùy chọn thay danh sách block-size tự sinh; tự bật tuning. | Lũy thừa của 2 từ 8 đến 256, giới hạn theo kích thước lớn nhất; với kích thước nhỏ hơn 8 thử đúng kích thước lớn nhất |
| `--algorithm NAME` | Chọn `all`, `naive`, `ikj`, `avx2`, `blocked`, `omp`, `auto` hoặc `vector-add`. | `all` |
| `--csv FILE` | Ghi kết quả dạng CSV vào tệp chỉ định. | Không ghi CSV |
| `--help` | In hướng dẫn lệnh. | — |

Benchmark riêng cộng vector trên cả `float` và `double`; `--size`/`--sizes` là số phần tử mỗi vector:

```powershell
.\nvxk.exe --size 1000000 --algorithm vector-add --runs 5 --csv vector-results.csv
```

Kết quả được kiểm tra với phép cộng scalar tương ứng trước khi báo thời gian trung vị và băng thông GB/s. Băng thông ước tính tính ba lần truy cập dữ liệu trên mỗi phần tử (hai lần đọc, một lần ghi). Benchmark vector dùng CSV `operation,type,size,median_seconds,gigabytes_per_second`; CSV GEMM giữ schema riêng.

Ví dụ: chạy kích thước 512 với OpenMP 1, 2 và 4 luồng, dùng 5 mẫu và xuất CSV:

```powershell
.\nvxk.exe --size 512 --algorithm omp --threads 1,2,4 --runs 5 --csv omp-results.csv
```

Chạy các thuật toán đơn luồng tại vài kích thước:

```powershell
.\nvxk.exe --sizes 128,256,512,1024 --algorithm all --runs 5 --csv results.csv
```

Tự benchmark các block-size, in cấu hình nhanh nhất theo từng kích thước:

```powershell
.\nvxk.exe --sizes 256,512,1024 --algorithm blocked --tune-block-size --block-sizes 32,64,128,256
```

Benchmark ma trận chữ nhật:

```powershell
.\nvxk.exe --shape 128,256,64 --algorithm all --runs 3
```

Khi `--algorithm all`, chương trình chạy tất cả thuật toán; tuỳ chọn `--threads` áp dụng cho `omp`. Các thuật toán còn lại được báo với cột số luồng bằng 1.

## Kết quả benchmark

Benchmark thực hiện một lượt làm nóng, sau đó đo nhiều mẫu. Nếu một mẫu ngắn hơn khoảng 0.05 giây, hàm được lặp nhiều lần trong mẫu đó để tăng độ phân giải của phép đo. Thời gian mỗi lần gọi được tính từ tổng thời gian chia cho số lần lặp; kết quả báo cáo là trung vị các mẫu.

Chương trình kiểm tra từng đầu ra so với GEMM naive bằng sai số hỗn hợp: sai số tuyệt đối tối đa `1e-5 + 1e-4 * max(|reference|, |result|)`. Cách này áp dụng sai số tương đối cho giá trị lớn mà vẫn có ngưỡng tuyệt đối hữu ích gần 0. Nếu kết quả có NaN/vô cực hoặc vượt ngưỡng, chương trình báo `[FAILED]` và thoát với mã lỗi.

Các cột kết quả:

- `Seconds`: thời gian trung vị cho một lần gọi hàm, tính bằng giây.
- `GFLOPS`: số tỷ phép tính dấu phẩy động ước lượng mỗi giây. Công thức là `2 * M * K * N`; cột kích thước hiển thị `M x K x N`.

CSV có các cột:

```text
M,K,N,algorithm,threads,median_seconds,gflops
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

- Ứng dụng nhập ma trận và benchmark hiện nhận kích thước chữ nhật; benchmark giữ `--size`/`--sizes` cho ma trận vuông.
- GEMM AVX2/FMA có cho `float`/`double`; chia khối và OpenMP hiện chỉ tối ưu cho `float`. Số nguyên 32→64 có API riêng scalar; các template số nguyên hiện có kiểm tra tràn và dành cho trường hợp cần kiểm tra.
- Benchmark tự sinh block-size ứng viên khi tuning; lựa chọn tốt nhất vẫn phụ thuộc CPU và tải hệ thống. `XK_blocked` vuông mặc định tiếp tục dùng block size 64.
- CLI phát hiện hỗ trợ AVX2/FMA nhưng chưa tự hiển thị tên model CPU/compiler.
- Fallback cho CPU cũ dùng i-k-j tuần tự hoặc OpenMP; hiệu năng có thể thấp hơn AVX2/FMA.
- `vector_add_avx2` có bài kiểm tra `float`/`double` cho kích thước nhỏ, phần dư SIMD và mảng rỗng; benchmark riêng dùng `--algorithm vector-add`.
- OpenMP tự chuyển sang đường chạy đơn luồng khi số phép nhân-cộng ước tính thấp hơn ngưỡng tương đương `256^3`; ngưỡng này hiện được cố định trong mã.
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
