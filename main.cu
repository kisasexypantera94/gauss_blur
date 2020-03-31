#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <exception>

inline auto f(const double x, const double sigma) -> double {
    return 1 / sqrt(2 * M_PI * sigma*sigma) * exp(-x*x / (2 * sigma*sigma));
}

auto comp_kernel(const size_t R) -> std::vector<double> {
    const double sigma = double(R) / 3;

    std::vector<double> kernel(R + 1);
    double sum = 0;
    for (size_t t = 1; t <= R; ++t) {
        kernel[t] = f(t, sigma);
        sum += kernel[t];
    }
    kernel[0] = 1.0 - 2 * sum;

    return kernel;
}

struct Image {
    std::vector<uchar4> img;
    int height;
    int width;

    Image() = default;
    void load(const std::string &filename) {
        {
            std::ifstream ifs(filename, std::ios::binary | std::ios::in);
            if (!ifs.is_open()) {
                std::throw_with_nested(std::runtime_error("could not open file " + filename));
            }

            ifs.read(reinterpret_cast<char *>(&width), sizeof(int));
            ifs.read(reinterpret_cast<char *>(&height), sizeof(int));
            img.resize(height * width);
            ifs.read(reinterpret_cast<char *>(img.data()), height * width * sizeof(uchar4));
        }
    }

    void save(const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary | std::ios::out);
        if (!ofs.is_open()) {
            std::throw_with_nested(std::runtime_error("could not open file " + filename));
        }

        ofs.write(reinterpret_cast<const char *>(&width), sizeof(int));
        ofs.write(reinterpret_cast<const char *>(&height), sizeof(int));
        ofs.write(reinterpret_cast<const char *>(img.data()), height * width * sizeof(uchar4));
    }
};

inline void CHECK_ERR(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
        exit(0);
    }
}

texture<uchar4, 2> tex_in;
texture<uchar4, 2> tex_out;

__device__ void swap(int *x, int *y) {
    int tmp = *x;
    *x = *y;
    *y = tmp;
}

__global__ void gauss_blur(int height,
                           int width,
                           const double *const kernel,
                           const int R,
                           uchar4 *const out,
                           bool horizontal) {
    for (int x = threadIdx.x + blockDim.x * blockIdx.x; x < width; x += blockDim.x * gridDim.x) {
        for (int y = threadIdx.y + blockDim.y * blockIdx.y; y < height; y += blockDim.y * gridDim.y) {
            uchar4 weighted_sum = make_uchar4(0, 0, 0, 255);
            for (int r = -R; r <= R; ++r) {
                int tex_x = x + (horizontal ? r : 0);
                int tex_y = y + (horizontal ? 0 : r);

                const uchar4 tex = tex2D(horizontal ? tex_out : tex_in, tex_x, tex_y);
                const double scale = kernel[abs(r)];
                weighted_sum.x += scale * tex.x;
                weighted_sum.y += scale * tex.y;
                weighted_sum.z += scale * tex.z;
            }

            const int offset = width * y + x;
            out[offset] = weighted_sum;
        }
    }
}

int main() {
    std::string in;
    std::string out;
    size_t R;

    std::cin >> in;
    std::cin >> out;
    std::cin >> R;

    // Prepare images
    Image h_in;
    Image h_out;

    try {
        h_in.load(in);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " + std::string(e.what()) << std::endl;
        return 1;
    }

    // Prepare filter
    auto kernel = comp_kernel(R);
    for (const auto x : kernel) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // Prepare device buffers
    uchar4 *d_out;
    double *d_kernel;
    CHECK_ERR(cudaMalloc(&d_out, h_in.img.size() * sizeof(uchar4)));
    CHECK_ERR(cudaMalloc(&d_kernel, kernel.size() * sizeof(double)));

    CHECK_ERR(cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(double), cudaMemcpyHostToDevice));

    // Bind textures to device buffers
    cudaArray *arr_in;
    cudaArray *arr_out;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    CHECK_ERR(cudaMallocArray(&arr_in, &desc, h_in.width, h_in.height));
    CHECK_ERR(cudaMallocArray(&arr_out, &desc, h_in.width, h_in.height));
    CHECK_ERR(cudaMemcpyToArray(arr_in, 0, 0, h_in.img.data(), h_in.img.size() * sizeof(uchar4), cudaMemcpyHostToDevice));

    tex_in.addressMode[0] = cudaAddressModeClamp;
    tex_in.addressMode[1] = cudaAddressModeClamp;
    tex_in.channelDesc = desc;
    tex_in.filterMode = cudaFilterModePoint;
    tex_in.normalized = false;

    CHECK_ERR(cudaBindTextureToArray(tex_in, arr_in, desc));
    CHECK_ERR(cudaBindTextureToArray(tex_out, arr_out, desc));

    // Run kernel
    auto block_dim = dim3(32, 32);
    auto grid_dim = dim3(32, 32);
    gauss_blur<<<grid_dim, block_dim>>>(h_in.height, h_in.width, d_kernel, R, d_out, false);
    CHECK_ERR(cudaDeviceSynchronize());
    CHECK_ERR(cudaGetLastError());

    CHECK_ERR(cudaMemcpyToArray(arr_out, 0, 0, d_out, h_in.img.size() * sizeof(uchar4), cudaMemcpyDeviceToDevice));
    gauss_blur<<<grid_dim, block_dim>>>(h_in.height, h_in.width, d_kernel, R, d_out, true);
    CHECK_ERR(cudaDeviceSynchronize());
    CHECK_ERR(cudaGetLastError());

    // Get results
    h_out.img.resize(h_in.img.size());
    h_out.height = h_in.height;
    h_out.width = h_in.width;
    CHECK_ERR(cudaMemcpy(h_out.img.data(), d_out, h_in.img.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));

    h_out.save(out);

    CHECK_ERR(cudaUnbindTexture(tex_in));
    CHECK_ERR(cudaUnbindTexture(tex_out));

    CHECK_ERR(cudaFreeArray(arr_in));
    CHECK_ERR(cudaFreeArray(arr_out));
    CHECK_ERR(cudaFree(d_out));
    CHECK_ERR(cudaFree(d_kernel));
}
