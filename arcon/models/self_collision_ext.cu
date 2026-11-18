// The upstream broad-phase kernel already counts all hits, even when its
// circular storage overflows. Expose that count; Python must re-query until
// capacity is sufficient. No partial candidate set is accepted.
#include "collisions_static.cu"
void counted_candidates_cuda(at::Tensor boxes, at::Tensor tree,
                             at::Tensor pairs, at::Tensor count) {
    int n = (boxes.size(1)+1)/2;
    int nodes = boxes.size(1);
    int capacity = pairs.size(0);
    AT_DISPATCH_FLOATING_TYPES(boxes.scalar_type(), "pb4u_counted_broad_phase", ([&] {
        thrust::device_vector<BVHNode<scalar_t>> leaves(n);
        thrust::device_vector<BVHNode<scalar_t>> internal(n-1);
        thrust::device_vector<int> ids(n);
        thrust::sequence(ids.begin(), ids.end());
        reconstruct_bvh<scalar_t>((AABB<scalar_t>*)boxes.data_ptr<scalar_t>(),
            (long4*)tree.data_ptr<long>(), internal.data().get(), leaves.data().get(),
            ids.data().get(), nodes, n);
        cudaCheckError();
        int block = NUM_THREADS;
        findPotentialCollisions<scalar_t><<<(n+block-1)/block, block>>>(
            (long2*)pairs.data_ptr<long>(), internal.data().get(), leaves.data().get(),
            ids.data().get(), n, capacity, count.data_ptr<int>(), (scalar_t)0.);
        cudaCheckError();
        cudaError_t err = cudaDeviceSynchronize();
        TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    }));
}
