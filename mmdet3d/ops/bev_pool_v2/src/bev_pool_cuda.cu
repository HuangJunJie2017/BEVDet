// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111

#include <stdio.h>
#include <stdlib.h>

/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b, d, h, w, c]
*/
__global__ void bev_pool_v2_kernel(int c, int n_intervals,
                                  const float *__restrict__ depth,
                                  const float *__restrict__ feat,
                                  const int *__restrict__ ranks_depth,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  float psum = 0;
  const float* cur_depth;
  const float* cur_feat;
  for(int i = 0; i < interval_length; i++){
    cur_depth = depth + ranks_depth[interval_start+i];
    cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
    psum += *cur_feat * *cur_depth;
  }

  const int* cur_rank = ranks_bev + interval_start;
  float* cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}


/*
  Function: pillar pooling backward
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    out_grad         : gradient of the BEV fmap from top, FloatTensor[b, d, h, w, c]
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    depth_grad       : gradient of the depth fmap, FloatTensor
    feat_grad        : gradient of the feature fmap, FloatTensor
*/
__global__ void bev_pool_grad_kernel(int c, int n_intervals,
                                  const float *__restrict__ out_grad,
                                  const float *__restrict__ depth,
                                  const float *__restrict__ feat,
                                  const int *__restrict__ ranks_depth,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ depth_grad,
                                  float* __restrict__ feat_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_intervals) return;
  int interval_start = interval_starts[idx];
  int interval_length = interval_lengths[idx];

  const int* cur_rank;
  const float* cur_out_grad;
  const float* cur_out_grad_start;

  const float* cur_feat;
  const float* cur_feat_start;
  float* cur_depth_grad;
  float grad_sum;
  for(int i = 0; i < interval_length; i++){
    cur_rank = ranks_bev + interval_start + i;
    cur_out_grad_start = out_grad +  * cur_rank * c;
    cur_feat_start = feat + ranks_feat[interval_start+i] * c;

    grad_sum = 0;
    for(int cur_c = 0; cur_c < c; cur_c++){
      cur_out_grad = cur_out_grad_start + cur_c;
      cur_feat = cur_feat_start + cur_c;
      grad_sum += *cur_out_grad * *cur_feat;
    }

    cur_depth_grad = depth_grad + ranks_depth[interval_start+i];
    *cur_depth_grad = grad_sum;
  }

  float* cur_feat_grad;
  const float* cur_depth;
  for(int cur_c = 0; cur_c < c; cur_c++){
    grad_sum = 0;
    for(int i = 0; i < interval_length; i++){
      cur_rank = ranks_bev + interval_start + i;
      cur_out_grad = out_grad + *cur_rank * c + cur_c;

      cur_depth = depth + ranks_depth[interval_start+i];
      grad_sum += *cur_out_grad * *cur_depth;
    }
    cur_feat_grad = feat_grad + ranks_feat[interval_start] * c + cur_c ;
    * cur_feat_grad = grad_sum;
  }
}

__global__ void bev_pool_grad_kernel_opt(
    int c, int n_intervals, const float *__restrict__ out_grad,
    const float *__restrict__ depth, const float *__restrict__ feat,
    const int *__restrict__ ranks_depth, const int *__restrict__ ranks_feat,
    const int *__restrict__ ranks_bev, const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, float *__restrict__ depth_grad,
    float *__restrict__ feat_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;

  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  for (int i = 0; i < interval_length; ++i) {
    const float *cur_out_grad_start =
        out_grad + ranks_bev[interval_start + i] * c;
    const float *cur_feat_start = feat + ranks_feat[interval_start + i] * c;
    float *cur_depth_grad = depth_grad + ranks_depth[interval_start + i];
    atomicAdd(cur_depth_grad,
              cur_out_grad_start[cur_c] * cur_feat_start[cur_c]);
    const int cur_rank = ranks_bev[interval_start + i];
    float *cur_feat_grad = feat_grad + ranks_feat[interval_start] * c + cur_c;
    *cur_feat_grad += out_grad[cur_rank * c + cur_c] *
                      depth[ranks_depth[interval_start + i]];
  }
}

void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat, const int* ranks_depth,
  const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* out) {
  bev_pool_v2_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}

void bev_pool_v2_grad(int c, int n_intervals, const float* out_grad,
  const float* depth, const float* feat, const int* ranks_depth, const int* ranks_feat,
  const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* depth_grad, float* feat_grad) {
  bev_pool_grad_kernel<<<(int)ceil(((double)n_intervals / 256)), 256>>>(
     c, n_intervals, out_grad, depth, feat, ranks_depth, ranks_feat,
     ranks_bev, interval_starts, interval_lengths, depth_grad, feat_grad
  );
}

void bev_pool_v2_grad_opt(int c, int n_intervals, const float *out_grad,
                          const float *depth, const float *feat,
                          const int *ranks_depth, const int *ranks_feat,
                          const int *ranks_bev, const int *interval_starts,
                          const int *interval_lengths, float *depth_grad,
                          float *feat_grad) {
  bev_pool_grad_kernel_opt<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
      c, n_intervals, out_grad, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, depth_grad, feat_grad);
}
