#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#ifdef USE_MKL
  #include <mkl_cblas.h>
#else
  #include <cblas.h>
#endif

#include <omp.h>

#define N      4096
#define BLOCK  64
#define SEED   42

// размеры блоков для варианта 3
#define MC  128
#define KC  256
#define NR  16
#define MR  6

// таймер через WinAPI
double get_time() {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}

float* alloc_matrix() {
    float* p = (float*)_aligned_malloc((size_t)N * N * sizeof(float), 64);
    if (!p) {
        printf("Ошибка выделения памяти!\n");
        exit(1);
    }
    return p;
}

void fill_random(float* A) {
    for (int i = 0; i < N * N; i++)
        A[i] = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);
}

// сложность по формуле c = 2*n^3
double flop_count() {
    return 2.0 * (double)N * (double)N * (double)N;
}

// производительность в MFlops
double calc_mflops(double flops, double seconds) {
    return flops / seconds * 1e-6;
}

void verify(const char* label, const float* ref, const float* got) {
    float max_err = 0.0f;
    for (int i = 0; i < 4096; i++) {
        float diff = fabsf(ref[i] - got[i]);
        float base = fabsf(ref[i]) + 1e-6f;
        float err = diff / base;
        if (err > max_err) max_err = err;
    }
    printf("  Проверка %s: макс. погрешность = %.2e  %s\n",
        label, (double)max_err, max_err < 5e-3f ? "[OK]" : "[ВНИМАНИЕ]");
}

// Вариант 1: наивный алгоритм по формуле из линейной алгебры
// C[i][j] = сумма A[i][k] * B[k][j] по всем k
void matmul_naive(const float* A, const float* B, float* C) {
    memset(C, 0, (size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// Вариант 2: cblas_sgemm из Intel MKL
void matmul_blas(const float* A, const float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N,
        1.0f, A, N,
              B, N,
        0.0f, C, N);
}

/* Вариант 3: оптимизированный алгоритм
Оптимизации:
1. Упаковка панелей матриц A и B в непрерывные буферы
чтобы микроядро читало данные последовательно без промахов кэша
2. Микроядро 6x16 на AVX2 с инструкциями FMA
за одну итерацию делает 96 умножений-сложений
3. OpenMP для параллельного выполнения по блокам строк
*/

// вспомогательное микроядро для краевых блоков (неполные MR/NR)
void micro_kernel_generic(int kc, const float* Ap, const float* Bp,
                           float* C, int ldc, int mr, int nr) {
    float tmp[MR * NR];
    memset(tmp, 0, sizeof(tmp));
    for (int k = 0; k < kc; k++)
        for (int i = 0; i < mr; i++) {
            float a = Ap[i * kc + k];
            for (int j = 0; j < nr; j++)
                tmp[i * NR + j] += a * Bp[k * NR + j];
        }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            C[i * ldc + j] += tmp[i * NR + j];
}

// основное микроядро 6x16 на AVX2
// использует 12 ymm-регистров-аккумуляторов и FMA инструкции
void micro_kernel_avx2(int kc, const float* Ap, const float* Bp,
                        float* C, int ldc) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int k = 0; k < kc; k++) {
        __m256 b0 = _mm256_loadu_ps(Bp + k * NR);
        __m256 b1 = _mm256_loadu_ps(Bp + k * NR + 8);

        __m256 a;
        a = _mm256_set1_ps(Ap[0 * kc + k]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_set1_ps(Ap[1 * kc + k]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_set1_ps(Ap[2 * kc + k]); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_set1_ps(Ap[3 * kc + k]); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_set1_ps(Ap[4 * kc + k]); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_set1_ps(Ap[5 * kc + k]); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    // записываем результат в C (накапливаем к существующим значениям)
    #define STORE(row, r0, r1) { \
        __m256 cc0 = _mm256_loadu_ps(C + (row)*ldc);     \
        __m256 cc1 = _mm256_loadu_ps(C + (row)*ldc + 8); \
        _mm256_storeu_ps(C + (row)*ldc,     _mm256_add_ps(cc0, r0)); \
        _mm256_storeu_ps(C + (row)*ldc + 8, _mm256_add_ps(cc1, r1)); }

    STORE(0, c00, c01) STORE(1, c10, c11) STORE(2, c20, c21)
    STORE(3, c30, c31) STORE(4, c40, c41) STORE(5, c50, c51)
    #undef STORE
}

void matmul_optimized(const float* A, const float* B, float* C) {
    memset(C, 0, (size_t)N * N * sizeof(float));

    int max_threads = omp_get_max_threads();

    // буфер панели B общий для всех потоков (только читается)
    float* Bp_buf = (float*)_aligned_malloc((size_t)KC * N * sizeof(float), 64);

    // у каждого потока своя копия панели A
    float** Ap_bufs = (float**)malloc(max_threads * sizeof(float*));
    for (int t = 0; t < max_threads; t++)
        Ap_bufs[t] = (float*)_aligned_malloc((size_t)MC * KC * sizeof(float), 64);

    // внешний цикл по блокам k
    for (int kc_start = 0; kc_start < N; kc_start += KC) {
        int kc = (kc_start + KC < N) ? KC : N - kc_start;

        // упаковываем панель B в буфер по NR-столбцам
        for (int jj = 0; jj < N; jj += NR) {
            int jc = (jj + NR < N) ? NR : N - jj;
            for (int kk = 0; kk < kc; kk++) {
                for (int j2 = 0; j2 < jc; j2++)
                    Bp_buf[(jj / NR) * KC * NR + kk * NR + j2] =
                        B[(kc_start + kk) * N + jj + j2];
                for (int j2 = jc; j2 < NR; j2++)
                    Bp_buf[(jj / NR) * KC * NR + kk * NR + j2] = 0.0f;
            }
        }

        int jb;
        #pragma omp parallel for schedule(static)
        for (jb = 0; jb < N; jb += MC) {
            int tid = omp_get_thread_num();
            int mc = (jb + MC < N) ? MC : N - jb;
            float* Ap = Ap_bufs[tid];

            // упаковываем панель A для текущего потока
            for (int i2 = 0; i2 < mc; i2++)
                for (int k2 = 0; k2 < kc; k2++)
                    Ap[i2 * KC + k2] = A[(jb + i2) * N + kc_start + k2];

            // вызываем микроядра
            for (int jj = 0; jj < N; jj += NR) {
                int jc = (jj + NR < N) ? NR : N - jj;
                float* Bp = Bp_buf + (jj / NR) * KC * NR;

                for (int i2 = 0; i2 < mc; i2 += MR) {
                    int mr = (i2 + MR < mc) ? MR : mc - i2;
                    float* Cptr = C + (jb + i2) * N + jj;
                    float* Aptr = Ap + i2 * KC;

                    if (mr == MR && jc == NR)
                        micro_kernel_avx2(kc, Aptr, Bp, Cptr, N);
                    else
                        micro_kernel_generic(kc, Aptr, Bp, Cptr, N, mr, jc);
                }
            }
        }
    }

    for (int t = 0; t < max_threads; t++)
        _aligned_free(Ap_bufs[t]);
    free(Ap_bufs);
    _aligned_free(Bp_buf);
}


int main() {
    printf(" Корепанов Никита Артемович, группа 090304-РПИб-о25\n\n");
    printf(" Лабораторная работа: перемножение матриц %dx%d \n\n", N, N);

    float* A  = alloc_matrix();
    float* B  = alloc_matrix();
    float* C1 = alloc_matrix();
    float* C2 = alloc_matrix();
    float* C3 = alloc_matrix();

    srand(SEED);
    fill_random(A);
    fill_random(B);
    printf("Матрицы заполнены случайными числами (seed=%d).\n\n", SEED);

    double t_start, t_end;
    double t_naive = 0, t_blas, t_opt;

    // вариант 2 запускаем первым как эталон
    printf("Запуск варианта 2 (cblas_sgemm)...\n");
    t_start = get_time();
    matmul_blas(A, B, C2);
    t_end = get_time();
    t_blas = t_end - t_start;
    printf("Вариант 2: cblas_sgemm (MKL)   время = %.3f с   произв. = %.1f MFlops\n\n",
        t_blas, calc_mflops(flop_count(), t_blas));

    // вариант 1 #define SKIP_NAIVE уже раскоменнтирован, чтобы пропустить 
#define SKIP_NAIVE 
#ifndef SKIP_NAIVE
    printf("Запуск варианта 1 (наивный)...\n");
    t_start = get_time();
    matmul_naive(A, B, C1);
    t_end = get_time();
    t_naive = t_end - t_start;
    printf("Вариант 1: наивный (ijk)       время = %.3f с   произв. = %.1f MFlops\n",
        t_naive, calc_mflops(flop_count(), t_naive));
    verify("C_naive vs C_blas", C2, C1);
    printf("\n");
#else
    printf("Вариант 1 (наивный) пропущен.\n\n");
#endif

    // вариант 3
    printf("Запуск варианта 3 (AVX2 + упаковка панелей + OpenMP)...\n");
    printf("  OpenMP: %d поток(ов)\n", omp_get_max_threads());
    t_start = get_time();
    matmul_optimized(A, B, C3);
    t_end = get_time();
    t_opt = t_end - t_start;
    printf("Вариант 3: AVX2+pack+OpenMP    время = %.3f с   произв. = %.1f MFlops\n",
        t_opt, calc_mflops(flop_count(), t_opt));
    verify("C_opt vs C_blas", C2, C3);

    // итоговая сводка
    double c = flop_count();
    double p_blas = calc_mflops(c, t_blas);
    double p_opt  = calc_mflops(c, t_opt);
    double pct    = p_opt / p_blas * 100.0;

    printf("\nИТОГОВАЯ СВОДКА\n");
    printf("Сложность: c = 2*n^3 = 2*%d^3 = %.3e FLop\n\n", N, c);
    printf("%-30s  %8s   %10s\n", "Вариант", "Время, с", "MFlops");
    printf("%-30s  %8s   %10s\n", "-------", "--------", "------");
#ifndef SKIP_NAIVE
    printf("%-30s  %8.3f   %10.1f\n", "1. Наивный (ijk)", t_naive, calc_mflops(c, t_naive));
#endif
    printf("%-30s  %8.3f   %10.1f\n", "2. cblas_sgemm (MKL)", t_blas, p_blas);
    printf("%-30s  %8.3f   %10.1f\n", "3. AVX2+pack+OpenMP",  t_opt,  p_opt);
    printf("\nВариант 3 / BLAS: %.1f%%  %s\n", pct,
        pct >= 30.0 ? "[>= 30%]" : "[< 30%]");

    _aligned_free(A); _aligned_free(B);
    _aligned_free(C1); _aligned_free(C2); _aligned_free(C3);

    return 0;
}
