#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

extern "C" {
#include "turboq.h"
}

namespace py = pybind11;

// encode(arr: np.ndarray[float16], layer_idx, head_idx) -> bytes
// arr must be 1-D, dtype float16, length divisible by 256
static py::bytes encode(py::array_t<uint16_t, py::array::c_style | py::array::forcecast> arr,
                        uint32_t layer_idx, uint32_t head_idx, int bit_width) {
    if (arr.ndim() != 1)
        throw std::invalid_argument("arr must be 1-D");

    size_t n = (size_t)arr.size();
    if (n % 256 != 0)
        throw std::invalid_argument("length must be divisible by 256");

    size_t n_blocks = n / 256;
    size_t block_bytes = (bit_width == 4) ? sizeof(block_tbq4_0) : sizeof(block_tbq3_0);
    size_t out_bytes = n_blocks * block_bytes;

    std::vector<char> buf(out_bytes);
    const ggml_half *src = reinterpret_cast<const ggml_half *>(arr.data());

    char *dst = buf.data();
    for (size_t i = 0; i < n_blocks; ++i) {
        turboq_encode_f16(src + i * 256, dst + i * block_bytes, 256, bit_width,
                          layer_idx, head_idx);
    }

    return py::bytes(buf.data(), out_bytes);
}

// decode(data: bytes, n: int, layer_idx, head_idx, bit_width) -> np.ndarray[float16]
static py::array_t<uint16_t> decode(py::bytes data, size_t n,
                                    uint32_t layer_idx, uint32_t head_idx,
                                    int bit_width) {
    if (n % 256 != 0)
        throw std::invalid_argument("n must be divisible by 256");

    std::string raw = data;
    size_t n_blocks = n / 256;
    size_t block_bytes = (bit_width == 4) ? sizeof(block_tbq4_0) : sizeof(block_tbq3_0);
    if (raw.size() < n_blocks * block_bytes)
        throw std::invalid_argument("data too short for declared n");

    py::array_t<uint16_t> out(n);
    ggml_half *dst = reinterpret_cast<ggml_half *>(out.mutable_data());

    for (size_t i = 0; i < n_blocks; ++i) {
        turboq_decode_f16(raw.data() + i * block_bytes, dst + i * 256, 256,
                          bit_width, layer_idx, head_idx);
    }

    return out;
}

PYBIND11_MODULE(turboq_ext, m) {
    m.doc() = "TurboQuant KV-cache compression — encode/decode for float16 vectors";

    m.def("encode",
          [](py::array_t<uint16_t, py::array::c_style | py::array::forcecast> arr,
             uint32_t layer_idx, uint32_t head_idx, int bit_width) {
              return encode(arr, layer_idx, head_idx, bit_width);
          },
          py::arg("arr"), py::arg("layer_idx") = 0, py::arg("head_idx") = 0,
          py::arg("bit_width") = 4,
          "Encode a float16 array (len % 256 == 0) → compressed bytes");

    m.def("decode",
          [](py::bytes data, size_t n, uint32_t layer_idx, uint32_t head_idx,
             int bit_width) {
              return decode(data, n, layer_idx, head_idx, bit_width);
          },
          py::arg("data"), py::arg("n"), py::arg("layer_idx") = 0,
          py::arg("head_idx") = 0, py::arg("bit_width") = 4,
          "Decode compressed bytes → float16 numpy array of length n");

    m.attr("BLOCK_SIZE") = py::int_(256);
    m.attr("BYTES_PER_BLOCK_4BIT") = py::int_((int)sizeof(block_tbq4_0));
    m.attr("BYTES_PER_BLOCK_3BIT") = py::int_((int)sizeof(block_tbq3_0));
}
