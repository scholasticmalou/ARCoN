// Counted broad phase for the approved soft self-target comparison.
// Uses the existing, read-only CCCollisions compatible CUDA implementation.
#include <torch/extension.h>
#include <vector>
void build_bvh(at::Tensor triangles, at::Tensor* boxes, at::Tensor* tree);
void counted_candidates_cuda(at::Tensor boxes, at::Tensor tree,
                             at::Tensor pairs, at::Tensor count);
std::vector<at::Tensor> bvh(at::Tensor triangles) {
    TORCH_CHECK(triangles.is_cuda() && triangles.is_contiguous());
    TORCH_CHECK(triangles.dim() == 4 && triangles.size(0) == 1 && triangles.size(1) > 1);
    auto n = triangles.size(1);
    auto boxes = at::full({1, 2*n-1, 6}, -1, triangles.options());
    auto tree = at::full({1, 2*n-1, 4}, -1, triangles.options().dtype(at::kLong));
    build_bvh(triangles, &boxes, &tree);
    return {boxes, tree};
}
std::vector<at::Tensor> candidates(at::Tensor boxes, at::Tensor tree, int capacity) {
    TORCH_CHECK(boxes.is_cuda() && boxes.is_contiguous() && tree.is_contiguous());
    TORCH_CHECK(boxes.size(0) == 1 && capacity > 0);
    auto pairs = at::full({capacity, 2}, -1, tree.options());
    auto count = at::zeros({1}, tree.options().dtype(at::kInt));
    counted_candidates_cuda(boxes, tree, pairs, count);
    return {pairs, count};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bvh", &bvh);
    m.def("candidates", &candidates);
}
