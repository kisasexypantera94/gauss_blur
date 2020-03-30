#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <exception>

inline auto f(const double x, const double sigma) -> double {
    return 1 / sqrt(2 * M_PI * sigma*sigma) * exp(-x*x / (2 * sigma*sigma));
}

auto comp_kernel(const size_t R) -> std::vector<double> {
    const size_t sigma = R / 3;

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

__global__ void gauss_blur(int height, 
                           int width, 
                           const double *const kernel,
                           const int R,
                           double *const out,
                           bool horizontal) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // change direction
    if (horizontal) {
        int tmp = height;
        height = width;
        width = tmp;
    }

    while (tid < width) {
        // build initial window
        uchar4 cur_scale = make_uchar4(0, 0, 0, 0);
        for (int i = -R; i <= R; ++i) {
            if (horizontal) {
                uchar4 tex = tex2D(tex_out, i, tid);
                cur_scale.x += kernel[abs(R)] * tex.x;
                cur_scale.y += kernel[abs(R)] * tex.y;
                cur_scale.z += kernel[abs(R)] * tex.z;
            } else {
                uchar4 tex = tex2D(tex_in, tid, i);
                cur_scale.x += kernel[abs(R)] * tex.x;
                cur_scale.y += kernel[abs(R)] * tex.y;
                cur_scale.z += kernel[abs(R)] * tex.z;
            }
        }

        // iterate over column/row with sliding window
        for (int i = 0; i < height; ++i) {
            
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
