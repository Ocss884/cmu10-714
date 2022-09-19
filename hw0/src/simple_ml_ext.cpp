#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float * mul(float *LM, float *RM, size_t M, size_t K, size_t N){
  float *res = new float[M*N];
  for (size_t m=0; m<M; m++) {
    for (size_t n=0; n<N; n++) {
      res[m*N + n] = 0;
      for (size_t k=0; k<K; k++) {
        res[m*N + n] += LM[m*K + k] * RM[k*N + n];
      }
    }
  }
  return res;
}
float * scal_prod(float scale, float *input, size_t M, size_t N) {
  float *res = new float[M*N];
  for (size_t i=0; i<M*N; i++) {
    res[i] = scale * input[i];
  }
  return res;
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t num_batch = ((m - 1) / batch) + 1;

    for (size_t bid=0; bid<num_batch; bid++) {
      // batch size in each loop
      size_t batch_size;
      if (bid+1==num_batch) {
        batch_size = m - bid*batch;
      } else {
        batch_size = batch;
      }
      float lrb = lr/batch_size;

      // slice X,y
      float b_x[batch_size*n] = {};
 
      for (size_t i=0; i<batch_size*n; i++){
        b_x[i] = X[bid*batch*n+i];
      }

      // Onehot y matrix
      size_t onehot_y[batch_size*k] = {};
      for (size_t i=0; i<batch_size; i++) {
        onehot_y[i*k + y[bid*batch+i]] = 1;
      }

      float *b_z = mul(b_x, theta, batch_size,n,k);
  //       for (size_t i=0; i<batch_size; i++) {
  //         for (size_t j=0; j<k; j++) {
  //           for (size_t o=0; o<n; o++) {
  //             b_z[i*k + j] += b_x[i*n + o] * theta[o*k+j];
  //     }
  //   }
  // }
      // exponential z
      for (size_t i=0; i<batch_size*k; i++){
        b_z[i] = exp(b_z[i]);
      }
      // normalize z
      float sum[batch_size] = {};
      for (size_t i=0; i<batch_size; i++){
        for (size_t j=0; j<k; j++) {
          sum[i] += b_z[i*k+j];
        }
      }
      for (size_t i=0; i<batch_size; i++) {
        for (size_t j=0; j<k; j++) {
          b_z[i*k+j] = b_z[i*k+j]/sum[i];
        }
      }
      // transpose b_x.T
      float b_xT[n*batch_size] = {};
      for (size_t i=0; i<batch_size; i++) {
        for (size_t j=0; j<n; j++) {
          b_xT[j*batch_size+i] = b_x[i*n+j];
        }
      }
      // b_z - onehot_y
      for (size_t i=0; i<batch_size*k; i++) {
        b_z[i] -= onehot_y[i];
      }
      float *update;
      float *prod = mul(b_xT, b_z, n, batch_size, k);
      update = scal_prod(lrb, prod, n, k);
      for (size_t i=0; i<n*k; i++) {
        theta[i] -= update[i];
      }
      delete[] b_z;
      delete[] prod;
      delete[] update;
    }
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
