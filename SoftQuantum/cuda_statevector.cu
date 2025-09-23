// cuda_statevector.cu
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <stdint.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct C128 { double x, y; }; // real=x, imag=y

__device__ __forceinline__ C128 cmul(const C128 a, const C128 b){
    return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
}
__device__ __forceinline__ C128 cadd(const C128 a, const C128 b){ return {a.x+b.x, a.y+b.y}; }

__device__ __forceinline__ uint64_t insert_bit(uint64_t base, int bit, int val){
    uint64_t low = base & ((1ULL << bit) - 1ULL);
    uint64_t high = base >> bit;
    return (high << (bit+1)) | (uint64_t(val) << bit) | low;
}

// ------------------------------ 1q / 2q kernels ----------------------------
__global__ void k_apply_1q(C128* psi, int n, int q, C128 u00, C128 u01, C128 u10, C128 u11){
    uint64_t pairs = 1ULL << (n-1);
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) return;
    uint64_t i0 = insert_bit(idx, q, 0);
    uint64_t i1 = i0 | (1ULL << q);
    C128 a0 = psi[i0];
    C128 a1 = psi[i1];
    C128 b0 = cadd(cmul(u00,a0), cmul(u01,a1));
    C128 b1 = cadd(cmul(u10,a0), cmul(u11,a1));
    psi[i0] = b0; psi[i1] = b1;
}

__global__ void k_apply_2q(C128* psi, int n, int q1, int q2, const C128* U){
    int lo = min(q1,q2), hi = max(q1,q2);
    uint64_t quads = 1ULL << (n-2);
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= quads) return;
    // build base index with target bits zeroed
    uint64_t low = idx & ((1ULL << lo) - 1ULL);
    uint64_t mid = (idx >> lo) & ((1ULL << (hi-lo-1)) - 1ULL);
    uint64_t high = idx >> (hi-1);
    uint64_t base = (high << (hi+1)) | (mid << (lo+1)) | low;
    uint64_t i00 = base;
    uint64_t i01 = base | (1ULL << lo);
    uint64_t i10 = base | (1ULL << hi);
    uint64_t i11 = base | (1ULL << lo) | (1ULL << hi);

    C128 a00 = psi[i00]; C128 a01 = psi[i01]; C128 a10 = psi[i10]; C128 a11 = psi[i11];
    const C128 *Urm = U; // row-major 4x4
    C128 b00 = cadd(cmul(Urm[0],a00), cadd(cmul(Urm[1],a01), cadd(cmul(Urm[2],a10), cmul(Urm[3],a11))));
    C128 b01 = cadd(cmul(Urm[4],a00), cadd(cmul(Urm[5],a01), cadd(cmul(Urm[6],a10), cmul(Urm[7],a11))));
    C128 b10 = cadd(cmul(Urm[8],a00), cadd(cmul(Urm[9],a01), cadd(cmul(Urm[10],a10), cmul(Urm[11],a11))));
    C128 b11 = cadd(cmul(Urm[12],a00), cadd(cmul(Urm[13],a01), cadd(cmul(Urm[14],a10), cmul(Urm[15],a11))));

    psi[i00] = b00; psi[i01] = b01; psi[i10] = b10; psi[i11] = b11;
}

__global__ void k_apply_cmask_1q(C128* psi, int n, uint64_t cmask, int q, C128 u00, C128 u01, C128 u10, C128 u11){
    uint64_t pairs = 1ULL << (n-1);
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) return;
    uint64_t i0 = insert_bit(idx, q, 0);
    if ((i0 & cmask) != cmask) return; // require all controls=1
    uint64_t i1 = i0 | (1ULL << q);
    C128 a0 = psi[i0];
    C128 a1 = psi[i1];
    C128 b0 = cadd(cmul(u00,a0), cmul(u01,a1));
    C128 b1 = cadd(cmul(u10,a0), cmul(u11,a1));
    psi[i0] = b0; psi[i1] = b1;
}

__global__ void k_apply_cmask_2q(C128* psi, int n, uint64_t cmask, int q1, int q2, const C128* U){
    int lo = min(q1,q2), hi = max(q1,q2);
    uint64_t quads = 1ULL << (n-2);
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= quads) return;
    uint64_t low = idx & ((1ULL << lo) - 1ULL);
    uint64_t mid = (idx >> lo) & ((1ULL << (hi-lo-1)) - 1ULL);
    uint64_t high = idx >> (hi-1);
    uint64_t base = (high << (hi+1)) | (mid << (lo+1)) | low;
    if ((base & cmask) != cmask) return;

    uint64_t i00 = base;
    uint64_t i01 = base | (1ULL << lo);
    uint64_t i10 = base | (1ULL << hi);
    uint64_t i11 = base | (1ULL << lo) | (1ULL << hi);

    C128 a00 = psi[i00]; C128 a01 = psi[i01]; C128 a10 = psi[i10]; C128 a11 = psi[i11];
    const C128 *Urm = U;
    C128 b00 = cadd(cmul(Urm[0],a00), cadd(cmul(Urm[1],a01), cadd(cmul(Urm[2],a10), cmul(Urm[3],a11))));
    C128 b01 = cadd(cmul(Urm[4],a00), cadd(cmul(Urm[5],a01), cadd(cmul(Urm[6],a10), cmul(Urm[7],a11))));
    C128 b10 = cadd(cmul(Urm[8],a00), cadd(cmul(Urm[9],a01), cadd(cmul(Urm[10],a10), cmul(Urm[11],a11))));
    C128 b11 = cadd(cmul(Urm[12],a00), cadd(cmul(Urm[13],a01), cadd(cmul(Urm[14],a10), cmul(Urm[15],a11))));

    psi[i00] = b00; psi[i01] = b01; psi[i10] = b10; psi[i11] = b11;
}

// -------------------------- Generic k-qubit kernel -------------------------
// Applies an arbitrary m x m unitary (m=2^k) to `targets` subset for all
// configurations of the remaining (n-k) qubits. Uses an out-of-place buffer.
// WARNING: Memory for U scales as 4^k. k=16 -> ~64 GiB for complex128.

__device__ __forceinline__ uint64_t scatter_bits(uint64_t packed, const int* pos, int npos){
    uint64_t out = 0ULL;
    for(int i=0;i<npos;i++){
        if ( (packed >> i) & 1ULL ) out |= (1ULL << pos[i]);
    }
    return out;
}

extern "C" __global__ void k_apply_kq(
    const C128* __restrict__ psi_in,
    C128* __restrict__ psi_out,
    int n,
    const int* __restrict__ targets,
    const int* __restrict__ rest,
    int k,
    const C128* __restrict__ U,
    uint64_t B) // number of base configurations = 2^(n-k)
{
    const uint64_t m = (1ULL << k);
    const uint64_t tid = threadIdx.x;
    const uint64_t stride = blockDim.x * gridDim.x;
    extern __shared__ __align__(16) unsigned char smem[];
    C128* sX = reinterpret_cast<C128*>(smem);
    const int TILE = blockDim.x;

    for (uint64_t base_idx = blockIdx.x * blockDim.x + tid; base_idx < B; base_idx += stride){
        uint64_t base = scatter_bits(base_idx, rest, n-k);
        // For each output index j in [0, m): compute y_j = sum_i U[j,i] * x_i
        for (uint64_t j = tid; j < m; j += blockDim.x){
            C128 acc = {0.0, 0.0};
            for (uint64_t i0 = 0; i0 < m; i0 += TILE){
                uint64_t ii = i0 + tid;
                if (ii < m){
                    uint64_t addr = base;
                    uint64_t temp = ii;
                    #pragma unroll 1
                    for(int t=0;t<k;t++){
                        if (temp & 1ULL) addr |= (1ULL << targets[t]);
                        temp >>= 1ULL;
                    }
                    sX[tid] = psi_in[addr];
                }
                __syncthreads();
                uint64_t ilim = (i0 + TILE <= m) ? TILE : (m - i0);
                for (uint64_t t = 0; t < ilim; ++t){
                    C128 u = U[j * m + (i0 + t)];
                    acc = cadd(acc, cmul(u, sX[t]));
                }
                __syncthreads();
            }
            uint64_t out_addr = base;
            uint64_t tempj = j;
            #pragma unroll 1
            for(int t=0;t<k;t++){
                if (tempj & 1ULL) out_addr |= (1ULL << targets[t]);
                tempj >>= 1ULL;
            }
            psi_out[out_addr] = acc;
        }
    }
}

static void check(bool ok, const char* msg){ if(!ok) throw std::runtime_error(msg); }

// Host wrappers --------------------------------------------------------------
void apply_1q(py::array_t<std::complex<double>, py::array::c_style> state, int n, int q, py::array_t<std::complex<double>, py::array::c_style> U){
    if (state.ndim()!=1) throw std::runtime_error("state must be 1D");
    if (!(U.ndim()==2 && U.shape(0)==2 && U.shape(1)==2)) throw std::runtime_error("U must be 2x2");
    size_t dim = (size_t)1 << n;
    if ((size_t)state.shape(0)!=dim) throw std::runtime_error("state length mismatch");

    C128 *d_state=nullptr; C128 *d_U=nullptr;
    cudaMalloc(&d_state, dim*sizeof(C128));
    cudaMalloc(&d_U, 4*sizeof(C128));
    cudaMemcpy(d_state, state.data(), dim*sizeof(C128), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U.data(), 4*sizeof(C128), cudaMemcpyHostToDevice);

    dim3 block(256); dim3 grid((unsigned)((((size_t)1<<(n-1))+255)/256));
    k_apply_1q<<<grid,block>>>(d_state, n, q, ((C128*)d_U)[0], ((C128*)d_U)[1], ((C128*)d_U)[2], ((C128*)d_U)[3]);
    cudaDeviceSynchronize();

    cudaMemcpy(state.mutable_data(), d_state, dim*sizeof(C128), cudaMemcpyDeviceToHost);
    cudaFree(d_state); cudaFree(d_U);
}

void apply_2q(py::array_t<std::complex<double>, py::array::c_style> state, int n, int q1, int q2, py::array_t<std::complex<double>, py::array::c_style> U){
    if (!(U.ndim()==2 && U.shape(0)==4 && U.shape(1)==4)) throw std::runtime_error("U must be 4x4");
    size_t dim = (size_t)1 << n;
    if ((size_t)state.shape(0)!=dim) throw std::runtime_error("state length mismatch");

    C128 *d_state=nullptr; C128 *d_U=nullptr;
    cudaMalloc(&d_state, dim*sizeof(C128));
    cudaMalloc(&d_U, 16*sizeof(C128));
    cudaMemcpy(d_state, state.data(), dim*sizeof(C128), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U.data(), 16*sizeof(C128), cudaMemcpyHostToDevice);

    dim3 block(256); dim3 grid((unsigned)((((size_t)1<<(n-2))+255)/256));
    k_apply_2q<<<grid,block>>>(d_state, n, q1, q2, d_U);
    cudaDeviceSynchronize();

    cudaMemcpy(state.mutable_data(), d_state, dim*sizeof(C128), cudaMemcpyDeviceToHost);
    cudaFree(d_state); cudaFree(d_U);
}

void apply_c1q(py::array_t<std::complex<double>, py::array::c_style> state, int n, uint64_t cmask, int q, py::array_t<std::complex<double>, py::array::c_style> U){
    size_t dim = (size_t)1 << n;
    C128 *d_state=nullptr; C128 *d_U=nullptr;
    cudaMalloc(&d_state, dim*sizeof(C128));
    cudaMalloc(&d_U, 4*sizeof(C128));
    cudaMemcpy(d_state, state.data(), dim*sizeof(C128), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U.data(), 4*sizeof(C128), cudaMemcpyHostToDevice);

    dim3 block(256); dim3 grid((unsigned)((((size_t)1<<(n-1))+255)/256));
    k_apply_cmask_1q<<<grid,block>>>(d_state, n, cmask, q, ((C128*)d_U)[0], ((C128*)d_U)[1], ((C128*)d_U)[2], ((C128*)d_U)[3]);
    cudaDeviceSynchronize();

    cudaMemcpy(state.mutable_data(), d_state, dim*sizeof(C128), cudaMemcpyDeviceToHost);
    cudaFree(d_state); cudaFree(d_U);
}

void apply_c2q(py::array_t<std::complex<double>, py::array::c_style> state, int n, uint64_t cmask, int q1, int q2, py::array_t<std::complex<double>, py::array::c_style> U){
    size_t dim = (size_t)1 << n;
    C128 *d_state=nullptr; C128 *d_U=nullptr;
    cudaMalloc(&d_state, dim*sizeof(C128));
    cudaMalloc(&d_U, 16*sizeof(C128));
    cudaMemcpy(d_state, state.data(), dim*sizeof(C128), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U.data(), 16*sizeof(C128), cudaMemcpyHostToDevice);

    dim3 block(256); dim3 grid((unsigned)((((size_t)1<<(n-2))+255)/256));
    k_apply_cmask_2q<<<grid,block>>>(d_state, n, cmask, q1, q2, d_U);
    cudaDeviceSynchronize();

    cudaMemcpy(state.mutable_data(), d_state, dim*sizeof(C128), cudaMemcpyDeviceToHost);
    cudaFree(d_state); cudaFree(d_U);
}

void apply_kq(py::array_t<std::complex<double>, py::array::c_style> state,
              int n,
              py::array_t<int, py::array::c_style> targets,
              py::array_t<std::complex<double>, py::array::c_style> U){
    if (targets.ndim()!=1) throw std::runtime_error("targets must be 1D int array");
    int k = (int)targets.shape(0);
    if (k < 3 || k > 16) throw std::runtime_error("k must be between 3 and 16 for apply_kq");
    size_t dim = (size_t)1 << n;
    if ((size_t)state.shape(0)!=dim) throw std::runtime_error("state length mismatch");
    uint64_t m = (1ULL << k);
    if (!(U.ndim()==2 && (uint64_t)U.shape(0)==m && (uint64_t)U.shape(1)==m)) throw std::runtime_error("U must be m x m");

    // Build rest indices on host
    std::vector<int> t_host(k);
    for (int i=0;i<k;i++) t_host[i] = targets.data()[i];
    std::vector<int> rest_host; rest_host.reserve(n-k);
    std::vector<char> is_t(n, 0);
    for (int i=0;i<k;i++) is_t[t_host[i]] = 1;
    for (int q=0;q<n;q++) if (!is_t[q]) rest_host.push_back(q);

    // Device buffers
    C128 *d_in=nullptr, *d_out=nullptr, *d_U=nullptr;
    int *d_t=nullptr, *d_r=nullptr;
    cudaMalloc(&d_in,  dim*sizeof(C128));
    cudaMalloc(&d_out, dim*sizeof(C128));
    cudaMalloc(&d_U,   (size_t)m*(size_t)m*sizeof(C128));
    cudaMalloc(&d_t,   k*sizeof(int));
    cudaMalloc(&d_r,   (n-k)*sizeof(int));

    cudaMemcpy(d_in, state.data(), dim*sizeof(C128), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U,  U.data(), (size_t)m*(size_t)m*sizeof(C128), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t,  t_host.data(), k*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,  rest_host.data(), (n-k)*sizeof(int), cudaMemcpyHostToDevice);

    uint64_t B = (1ULL << (n - k));
    dim3 block(256);
    dim3 grid((unsigned)((B + block.x - 1)/block.x));
    size_t shmem = block.x * sizeof(C128);
    k_apply_kq<<<grid, block, shmem>>>(d_in, d_out, n, d_t, d_r, k, d_U, B);
    cudaDeviceSynchronize();

    cudaMemcpy(state.mutable_data(), d_out, dim*sizeof(C128), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_U); cudaFree(d_t); cudaFree(d_r);
}

PYBIND11_MODULE(_svcuda, m){
    m.def("apply_1q", &apply_1q);
    m.def("apply_2q", &apply_2q);
    m.def("apply_c1q", &apply_c1q);
    m.def("apply_c2q", &apply_c2q);
    m.def("apply_kq", &apply_kq);
}
