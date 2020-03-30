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

auto load_img(const std::string &filename) -> std::vector<uchar4> {
    int height;
    int width;
    std::vector<uchar4> img;

    {
        std::ifstream ifs(filename, std::ios::binary | std::ios::in);
        if (!ifs.is_open()) {
            std::throw_with_nested(std::runtime_error("could not open file " + filename));
        }

        ifs.read(reinterpret_cast<char *>(&height), sizeof(int));
        ifs.read(reinterpret_cast<char *>(&width), sizeof(int));
        img.resize(height * width);
        ifs.read(reinterpret_cast<char *>(img.data()), height * width * sizeof(uchar4));
    }

    return img;
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
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // change direction
    if (horizontal) {
        swap(&height, &width);
    }

    while (tid < width) {
        // build initial window
        uchar4 cur_sum = make_uchar4(0, 0, 0, 0);
        for (int i = -R; i <= R; ++i) {
            // prepare texture coordinates
            int tex_x = tid;
            int tex_y = i;
            if (horizontal) {
                swap(&tex_x, &tex_y);
            }

            uchar4 tex = tex2D(horizontal ? tex_out : tex_in, tex_x, tex_y);
            cur_sum.x += kernel[abs(R)] * tex.x;
            cur_sum.y += kernel[abs(R)] * tex.y;
            cur_sum.z += kernel[abs(R)] * tex.z;
        }

        // iterate over column/row with sliding window
        for (int i = 0; i < height; ++i) {
            int offset = horizontal ? tid * width+ i : i * height + tid;
            out[offset].x = cur_sum.x;
            out[offset].y = cur_sum.y;
            out[offset].z = cur_sum.z;

            // prepare coordinates for both ends of the window
            int tex_last_x = tid;
            int tex_last_y = i - R;
            int tex_new_x = tid;
            int tex_new_y = i + R;

            if (horizontal) {
                swap(&tex_last_x, &tex_last_y);
                swap(&tex_new_x, &tex_new_y);
            }

            uchar4 tex_last = tex2D(horizontal ? tex_out : tex_in, tex_last_x, tex_last_y);
            uchar4 tex_new = tex2D(horizontal ? tex_out : tex_in, tex_new_x, tex_new_y);

            cur_sum.x += tex_new.x - tex_last.x;
            cur_sum.y += tex_new.y - tex_last.y;
            cur_sum.z += tex_new.z - tex_last.z;
        }

        tid += blockDim.x * gridDim.x;
    }
}

int main() {
    std::string in;
    std::string out;
    size_t R;

    std::cin >> in;
    std::cin >> out;
    std::cin >> R;

    std::vector<uchar4> h_in;
    try {
        h_in = load_img(in);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " + std::string(e.what()) << std::endl;
        return 1;
    }

    auto kernel = comp_kernel(R);
    for (const auto x : kernel) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    uchar4 *d_in;
    uchar4 *d_out;
    cudaMalloc(&d_in, h_in.size() * sizeof(uchar4));
    cudaMalloc(&d_out, h_in.size() * sizeof(uchar4));

    cudaBindTexture(NULL, tex_in, d_in, h_in.size() * sizeof(uchar4));
    cudaBindTexture(NULL, tex_out, d_out, h_in.size() * sizeof(uchar4));


    cudaMemcpy(&d_in, &h_in, h_in.size() * sizeof(uchar4), cudaMemcpyHostToDevice);

    cudaFree(d_in);
}
