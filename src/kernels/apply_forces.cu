/*
    Apply forces to the points with momentum, exaggeration, etc.
*/

#include "include/kernels/apply_forces.h"


/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/
// Edited to add momentum, repulsive, attr forces, etc.
__global__
void IntegrationKernel(
                                 volatile float * __restrict__ points,
                                 volatile float * __restrict__ attr_forces,
                                 volatile float * __restrict__ rep_forces,
                                 volatile float * __restrict__ gains,
                                 volatile float * __restrict__ old_forces,
                                 const float eta,
                                 const float normalization,
                                 const float momentum,
                                 const float exaggeration,
                                 const int num_points)
{
  register int i, inc;
  register float dx, dy, ux, uy, gx, gy;

  // iterate over all bodies assigned to thread
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < num_points; i += inc) {
        ux = old_forces[i];
        uy = old_forces[num_points + i];
        gx = gains[i];
        gy = gains[num_points + i];
        dx = exaggeration*attr_forces[i] - (rep_forces[i] / normalization);
        dy = exaggeration*attr_forces[i + num_points] - (rep_forces[i + num_points] / normalization);

        gx = (signbit(dx) != signbit(ux)) ? gx + 0.2 : gx * 0.8;
        gy = (signbit(dy) != signbit(uy)) ? gy + 0.2 : gy * 0.8;
        gx = (gx < 0.01) ? 0.01 : gx;
        gy = (gy < 0.01) ? 0.01 : gy;

        // This is the core of the gradient descent step
        // 1. computation of the gradient for each coordinate 
        ux = momentum * ux - eta * gx * dx;
        uy = momentum * uy - eta * gy * dy;
        // 2. appying the gradient to the points 
        points[i] += ux;
        points[i + num_points] += uy;

        attr_forces[i] = 0.0f; // Resetting variables 
        attr_forces[num_points + i] = 0.0f; // Resetting variables 
        rep_forces[i] = 0.0f; // Resetting variables 
        rep_forces[num_points + i] = 0.0f; // Resetting variables 

        old_forces[i] = ux; // gradient of previous step is used for momentum calculation
        old_forces[num_points + i] = uy; // gradient of previous step is used for momentum calculation
        gains[i] = gx; // gains are also updated 
        gains[num_points + i] = gy; // gains are also updated 
   }
}

void tsnecuda::ApplyForces(tsnecuda::GpuOptions &gpu_opt,
                                thrust::device_vector<float> &points, // (y_1x, y_1y, y_2x, y_2y, ...)
                                thrust::device_vector<float> &attr_forces, // ((attr_1x, attr_1y), ..., (attr_ix, attr_iy))
                                thrust::device_vector<float> &rep_forces, // ((rep_1x, rep_1y), ..., (rep_ix, rep_iy))
                                thrust::device_vector<float> &gains,
                                thrust::device_vector<float> &old_forces,
                                const float eta,
                                const float normalization,
                                const float momentum,
                                const float exaggeration,
                                const int num_points,
                                const int num_blocks)
{
    IntegrationKernel<<<num_blocks * gpu_opt.integration_kernel_factor,
                                  gpu_opt.integration_kernel_threads>>>(
                    thrust::raw_pointer_cast(points.data()),
                    thrust::raw_pointer_cast(attr_forces.data()),
                    thrust::raw_pointer_cast(rep_forces.data()),
                    thrust::raw_pointer_cast(gains.data()),
                    thrust::raw_pointer_cast(old_forces.data()),
                    eta, normalization, momentum, exaggeration,
                    num_points);
    GpuErrorCheck(cudaDeviceSynchronize());
}
