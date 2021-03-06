#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <exception>
#include <stdexcept>

// =========================================================================
texture<uchar4, 2> tex_in;
texture<uchar4, 2> tex_out;

__global__
void gauss_blur(const int height,
                const int width,
                const double *const filter, // TODO: maybe constant memory?
                const int R,
                uchar4 *const out,
                const bool horizontal) {
    for (int x = threadIdx.x + blockDim.x * blockIdx.x; x < width; x += blockDim.x * gridDim.x) {
        for (int y = threadIdx.y + blockDim.y * blockIdx.y; y < height; y += blockDim.y * gridDim.y) {
            double3 weighted_sum = make_double3(0, 0, 0);
            for (int r = -R; r <= R; ++r) {
                const int tex_x = x + (horizontal ? r : 0);
                const int tex_y = y + (horizontal ? 0 : r);

                const uchar4 tex = tex2D(horizontal ? tex_out : tex_in, tex_x, tex_y);
                const double scale = filter[abs(r)];
                weighted_sum.x += scale * double(tex.x);
                weighted_sum.y += scale * double(tex.y);
                weighted_sum.z += scale * double(tex.z);
            }

            out[width * y + x] = make_uchar4(weighted_sum.x, weighted_sum.y, weighted_sum.z, 0);
        }
    }
}
// =========================================================================


inline auto f(const double x, const double sigma) -> double {
    return 1.0 / sqrt(2.0 * M_PI * sigma*sigma) * exp(-x*x / (2.0 * sigma*sigma));
}

auto comp_filter(const size_t R) -> std::vector<double> {
    const double sigma = R;

    std::vector<double> filter(R + 1);
    double sum = 0;
    for (size_t t = 0; t <= R; ++t) {
        filter[t] = f(t, sigma);
        sum += filter[t];
    }
    sum -= filter[0];

    for (size_t t = 0; t <= R; ++t) {
        static double tmp = filter[0];
        filter[t] /= (sum * 2 + tmp);
    }

    return filter;
}

struct Image : std::vector<uchar4> {
    int height;
    int width;

    void load(const std::string &filename) {
        std::ifstream ifs(filename, std::ios::binary | std::ios::in);
        if (!ifs.is_open()) {
            std::throw_with_nested(std::runtime_error("could not open file " + filename));
        }

        ifs.read(reinterpret_cast<char *>(&width), sizeof(int));
        ifs.read(reinterpret_cast<char *>(&height), sizeof(int));
        this->resize(height * width);
        ifs.read(reinterpret_cast<char *>(this->data()), height * width * sizeof(uchar4));
    }

    void save(const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary | std::ios::out);
        if (!ofs.is_open()) {
            std::throw_with_nested(std::runtime_error("could not open file " + filename));
        }

        ofs.write(reinterpret_cast<const char *>(&width), sizeof(int));
        ofs.write(reinterpret_cast<const char *>(&height), sizeof(int));
        ofs.write(reinterpret_cast<const char *>(this->data()), height * width * sizeof(uchar4));
    }
};

inline void CHECK_ERR(const cudaError_t err) {
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
        exit(0);
    }
}

auto gauss_blur_image(const Image &h_in, const size_t R) -> Image {
    // Prepare filter
    const auto filter = comp_filter(R);

    // Prepare device buffers
    uchar4 *d_out;
    double *d_filter;
    CHECK_ERR(cudaMalloc(&d_out, h_in.size() * sizeof(uchar4)));
    CHECK_ERR(cudaMalloc(&d_filter, filter.size() * sizeof(double)));

    // Copy filter to device
    CHECK_ERR(cudaMemcpy(d_filter, filter.data(), filter.size() * sizeof(double), cudaMemcpyHostToDevice));

    // Bind textures to device buffers
    cudaArray *arr_in;
    cudaArray *arr_out;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    CHECK_ERR(cudaMallocArray(&arr_in, &desc, h_in.width, h_in.height));
    CHECK_ERR(cudaMallocArray(&arr_out, &desc, h_in.width, h_in.height));
    CHECK_ERR(cudaMemcpyToArray(arr_in, 0, 0, h_in.data(), h_in.size() * sizeof(uchar4), cudaMemcpyHostToDevice));

    CHECK_ERR(cudaBindTextureToArray(tex_in, arr_in, desc));
    CHECK_ERR(cudaBindTextureToArray(tex_out, arr_out, desc));

    // Run kernel
    auto block_dim = dim3(16, 16);
    auto grid_dim = dim3(16, 16);
    // Vertical
    gauss_blur<<<grid_dim, block_dim>>>(h_in.height, h_in.width, d_filter, R, d_out, false);
    CHECK_ERR(cudaDeviceSynchronize());
    CHECK_ERR(cudaGetLastError());

    CHECK_ERR(cudaMemcpyToArray(arr_out, 0, 0, d_out, h_in.size() * sizeof(uchar4), cudaMemcpyDeviceToDevice));
    // Horizontal
    gauss_blur<<<grid_dim, block_dim>>>(h_in.height, h_in.width, d_filter, R, d_out, true);
    CHECK_ERR(cudaDeviceSynchronize());
    CHECK_ERR(cudaGetLastError());

    // Get results
    Image h_out;
    h_out.resize(h_in.size());
    h_out.height = h_in.height;
    h_out.width = h_in.width;
    CHECK_ERR(cudaMemcpy(h_out.data(), d_out, h_in.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));

    CHECK_ERR(cudaUnbindTexture(tex_in));
    CHECK_ERR(cudaUnbindTexture(tex_out));

    CHECK_ERR(cudaFreeArray(arr_in));
    CHECK_ERR(cudaFreeArray(arr_out));
    CHECK_ERR(cudaFree(d_out));
    CHECK_ERR(cudaFree(d_filter));

    return h_out;
}

int main() {
    std::string in;
    std::string out;
    size_t R;

    std::cin >> in;
    std::cin >> out;
    std::cin >> R;

    Image h_in;
    try {
        h_in.load(in);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " + std::string(e.what()) << std::endl;
        return 1;
    }

    if (R == 0) {
        try {
            h_in.save(out);
        } catch (const std::exception &e) {
            std::cerr << "ERROR: " + std::string(e.what()) << std::endl;
            return 1;
        }

        return 0;
    }

    auto h_out = gauss_blur_image(h_in, R);
    try {
        h_out.save(out);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " + std::string(e.what()) << std::endl;
        return 1;
    }
}
