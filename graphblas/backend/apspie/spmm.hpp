#ifndef GRB_BACKEND_APSPIE_SPMM_HPP
#define GRB_BACKEND_APSPIE_SPMM_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>
// #include <helper_math.h>

#include <moderngpu.cuh>

#include "graphblas/backend/apspie/SparseMatrix.hpp"
#include "graphblas/backend/apspie/DenseMatrix.hpp"
#include "graphblas/backend/apspie/kernels/spmm.hpp"
#include "graphblas/types.hpp"
#include "graphblas/util.hpp"
#include "graphblas/log.hpp"

// #define TA     32
// #define TB     32
// #define NT     64

namespace graphblas
{
  namespace backend
  {
    template <typename c, typename m, typename a, typename b>
    Info spmm(DenseMatrix<c> &C,
              const SparseMatrix<m> &mask,
              const BinaryOp &accum,
              const Semiring &op,
              const SparseMatrix<a> &A,
              const DenseMatrix<b> &B,
              const Descriptor &desc)
    {
      Index A_nrows, A_ncols, A_nvals;
      Index B_nrows, B_ncols;
      Index C_nrows, C_ncols;

      A.nrows(A_nrows);
      A.ncols(A_ncols);
      A.nvals(A_nvals);
      B.nrows(B_nrows);
      B.ncols(B_ncols);
      C.nrows(C_nrows);
      C.ncols(C_ncols);

      // Dimension compatibility check
      if ((A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows))
      {
        std::cout << "Dim mismatch" << std::endl;
        std::cout << A_ncols << " " << B_nrows << std::endl;
        std::cout << C_ncols << " " << B_ncols << std::endl;
        std::cout << C_nrows << " " << A_nrows << std::endl;
        return GrB_DIMENSION_MISMATCH;
      }

      // Domain compatibility check
      // TODO: add domain compatibility check

      // Read descriptor
      Desc_value mode, ta, tb, nt;
      desc.get(GrB_MODE, mode);
      desc.get(GrB_TA, ta);
      desc.get(GrB_TB, tb);
      desc.get(GrB_NT, nt);

      // Computation
      const int T = static_cast<int>(ta);
      const int TB = static_cast<int>(tb);
      const int NTHREADS = static_cast<int>(nt);
      const int NBLOCKS = (T * A_nrows + NTHREADS - 1) / NTHREADS;

      dim3 NT;
      dim3 NB;
      NT.x = NTHREADS;
      NT.y = 1;
      NT.z = 1;
      NB.x = NBLOCKS;
      NB.y = (B_ncols + 31) / 32;
      NB.z = 1;

      // CUDA( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
      if (mode == GrB_FIXEDROW)
        switch (TB)
        {
        case 1:
          spmmRowKernel3<c, 1, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 2:
          spmmRowKernel3<c, 2, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 4:
          spmmRowKernel3<c, 4, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 8:
          spmmRowKernel3<c, 8, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 16:
          spmmRowKernel3<c, 16, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                   A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                   B.d_denseVal_, C.d_denseVal_);
          break;
        case 32:
          spmmRowKernel3<c, 32, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                   A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                   B.d_denseVal_, C.d_denseVal_);
          break;
          break;
        }
      else if (mode == GrB_FIXEDROW2)
        switch (TB)
        {
        case 1:
          spmmRowKernel3<c, 1, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                 A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                 B.d_denseVal_, C.d_denseVal_);
          break;
        case 2:
          spmmRowKernel3<c, 2, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                 A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                 B.d_denseVal_, C.d_denseVal_);
          break;
        case 4:
          spmmRowKernel3<c, 4, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                 A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                 B.d_denseVal_, C.d_denseVal_);
          break;
        case 8:
          spmmRowKernel3<c, 8, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                 A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                 B.d_denseVal_, C.d_denseVal_);
          break;
        case 16:
          spmmRowKernel3<c, 16, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 32:
          spmmRowKernel3<c, 32, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
          break;
        }
      else if (mode == GrB_FIXEDROW3)
        switch (TB)
        {
        case 1:
          spmmRowKernel2<c, 1, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 2:
          spmmRowKernel2<c, 2, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 4:
          spmmRowKernel2<c, 4, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 8:
          spmmRowKernel2<c, 8, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 16:
          spmmRowKernel2<c, 16, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                   A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                   B.d_denseVal_, C.d_denseVal_);
          break;
        case 32:
          spmmRowKernel2<c, 32, false><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                   A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                   B.d_denseVal_, C.d_denseVal_);
          break;
          break;
        }
      else if (mode == GrB_FIXEDROW4)
        switch (TB)
        {
        case 1:
          spmmRowKernel2<c, 1, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                 A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                 B.d_denseVal_, C.d_denseVal_);
          break;
        case 2:
          spmmRowKernel2<c, 2, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                 A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                 B.d_denseVal_, C.d_denseVal_);
          break;
        case 4:
          spmmRowKernel2<c, 4, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                 A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                 B.d_denseVal_, C.d_denseVal_);
          break;
        case 8:
          spmmRowKernel2<c, 8, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                 A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                 B.d_denseVal_, C.d_denseVal_);
          break;
        case 16:
          spmmRowKernel2<c, 16, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
        case 32:
          spmmRowKernel2<c, 32, true><<<NB, NT>>>(A_nrows, B_ncols, A_ncols,
                                                  A_nvals, A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                                  B.d_denseVal_, C.d_denseVal_);
          break;
          break;
        }

      CUDA(cudaDeviceSynchronize());
      // spmmColKernel<<<NBLOCKS,NTHREADS>>>( A_nrows, B_ncols, A_ncols, A_nvals,
      //   A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_, B.d_denseVal_, C.d_denseVal_ );

      C.need_update_ = true;
      return GrB_SUCCESS;
    }

    template <typename c, typename a, typename b>
    Info cusparse_spmm(DenseMatrix<c> &C,
                       const Semiring &op,
                       const SparseMatrix<a> &A,
                       const DenseMatrix<b> &B)
    {
      // alpha and beta
      float alpha = 1.0;
      float beta = 0.0;

      Index A_nrows, A_ncols, A_nvals;
      Index B_nrows, B_ncols;
      Index C_nrows, C_ncols;

      A.nrows(A_nrows);
      A.ncols(A_ncols);
      A.nvals(A_nvals);
      B.nrows(B_nrows);
      B.ncols(B_ncols);
      C.nrows(C_nrows);
      C.ncols(C_ncols);

      // Dimension compatibility check
      if ((A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows))
      {
        std::cout << "Dim mismatch" << std::endl;
        std::cout << A_ncols << " " << B_nrows << std::endl;
        std::cout << C_ncols << " " << B_ncols << std::endl;
        std::cout << C_nrows << " " << A_nrows << std::endl;
        return GrB_DIMENSION_MISMATCH;
      }

      // Domain compatibility check
      // TODO: add domain compatibility check

      // Computation cuda-11.6 impl
      cusparseHandle_t handle;
      cusparseCreate(&handle);
      cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

      cusparseSpMatDescr_t matA;
      cusparseDnMatDescr_t matB, matC;
      void *dBuffer = NULL;
      size_t bufferSize = 0;
      // Create sparse matrix A in CSR format
      CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_nrows, A_ncols, A_nvals,
                                       A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

      // Create dense matrix B
      CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B.nrows_, B.ncols_, A.ncols_, B.d_denseVal_,
                                         CUDA_R_32F, CUSPARSE_ORDER_COL));
      // Create dense matrix C
      CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C.nrows_, C.ncols_, A.nrows_, C.d_denseVal_,
                                         CUDA_R_32F, CUSPARSE_ORDER_COL));
      // allocate an external buffer if need
      CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                             CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
      CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
      // execute SpMM
      CHECK_CUSPARSE(cusparseSpMM(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
      C.need_update_ = true; // Set flag that we need to copy data from GPU
      // destroy matrix/vector descriptors
      CHECK_CUSPARSE(cusparseDestroySpMat(matA));
      CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
      CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
      CHECK_CUSPARSE(cusparseDestroy(handle));
      // device memory deallocation
      CHECK_CUDA(cudaFree(dBuffer));
      return GrB_SUCCESS;
    }

    template <typename c, typename a, typename b>
    Info cusparse_spmm2(DenseMatrix<c> &C,
                        const Semiring &op,
                        const SparseMatrix<a> &A,
                        const DenseMatrix<b> &B)
    {
      // alpha and beta
      float alpha = 1.0;
      float beta = 0.0;

      Index A_nrows, A_ncols, A_nvals;
      Index B_nrows, B_ncols;
      Index C_nrows, C_ncols;

      A.nrows(A_nrows);
      A.ncols(A_ncols);
      A.nvals(A_nvals);
      B.nrows(B_nrows);
      B.ncols(B_ncols);
      C.nrows(C_nrows);
      C.ncols(C_ncols);

      // Dimension compatibility check
      if ((A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows))
      {
        std::cout << "Dim mismatch" << std::endl;
        std::cout << A_ncols << " " << B_nrows << std::endl;
        std::cout << C_ncols << " " << B_ncols << std::endl;
        std::cout << C_nrows << " " << A_nrows << std::endl;
        return GrB_DIMENSION_MISMATCH;
      }

      // Domain compatibility check
      // TODO: add domain compatibility check

      // Computation cuda-11.6 impl
      cusparseHandle_t handle;
      cusparseCreate(&handle);
      cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

      cusparseSpMatDescr_t matA;
      cusparseDnMatDescr_t matB, matC;
      void *dBuffer = NULL;
      size_t bufferSize = 0;
      // Create sparse matrix A in CSR format
      CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_nrows, A_ncols, A_nvals,
                                       A.d_csrRowPtr_, A.d_csrColInd_, A.d_csrVal_,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

      // Create dense matrix B
      CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B.nrows_, B.ncols_, A.ncols_, B.d_denseVal_,
                                         CUDA_R_32F, CUSPARSE_ORDER_COL));
      // Create dense matrix C
      CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C.nrows_, C.ncols_, A.nrows_, C.d_denseVal_,
                                         CUDA_R_32F, CUSPARSE_ORDER_COL));
      // allocate an external buffer if need
      CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                             CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
      CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
      // execute SpMM
      CHECK_CUSPARSE(cusparseSpMM(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
      C.need_update_ = true; // Set flag that we need to copy data from GPU
      // destroy matrix/vector descriptors
      CHECK_CUSPARSE(cusparseDestroySpMat(matA));
      CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
      CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
      CHECK_CUSPARSE(cusparseDestroy(handle));
      // device memory deallocation
      CHECK_CUDA(cudaFree(dBuffer));
      return GrB_SUCCESS;
    }

    template <typename c, typename a, typename b>
    Info mergepath_spmm(DenseMatrix<c> &C,
                        const Semiring &op,
                        const SparseMatrix<a> &A,
                        const DenseMatrix<b> &B,
                        Descriptor &desc)
    {
      Index A_nrows, A_ncols, A_nvals;
      Index B_nrows, B_ncols;
      Index C_nrows, C_ncols;

      A.nrows(A_nrows);
      A.ncols(A_ncols);
      A.nvals(A_nvals);
      B.nrows(B_nrows);
      B.ncols(B_ncols);
      C.nrows(C_nrows);
      C.ncols(C_ncols);

      // std::cout << "A: " << A_nrows << " " << A_ncols << std::endl;
      // std::cout << "B: " << B_nrows << " " << B_ncols << std::endl;
      // std::cout << "C: " << C_nrows << " " << C_ncols << std::endl;

      // Dimension compatibility check
      if ((A_ncols != B_nrows) || (C_ncols != B_ncols) || (C_nrows != A_nrows))
      {
        std::cout << "Dim mismatch" << std::endl;
        std::cout << A_ncols << " " << B_nrows << std::endl;
        std::cout << C_ncols << " " << B_ncols << std::endl;
        std::cout << C_nrows << " " << A_nrows << std::endl;
        return GrB_DIMENSION_MISMATCH;
      }

      // Domain compatibility check
      // TODO: add domain compatibility check

      // Temporarily for testing purposes
      // C.allocate();
      Desc_value tb, nt;
      desc.get(GrB_TB, tb);
      desc.get(GrB_NT, nt);

      // Computation
      const int TB = static_cast<int>(tb);
      const int NT = static_cast<int>(nt);

      // Computation
      // std::cout << "Success creating mgpu context\n";
      mgpu::SpmmCsrBinary(A.d_csrVal_, A.d_csrColInd_, A_nvals, A.d_csrRowPtr_,
                          A_nrows, B.d_denseVal_, false, C.d_denseVal_, (c)0,
                          mgpu::multiplies<c>(), mgpu::plus<c>(), B_ncols, desc.d_limits_,
                          desc.d_carryin_, desc.d_carryout_, TB, NT, *desc.d_context_);
      // std::cout << "Finished SpmmCsrBinary\n";

      C.need_update_ = true; // Set flag that we need to copy data from GPU
      return GrB_SUCCESS;
    }

  } // backend
} // graphblas

#endif // GRB_BACKEND_APSPIE_SPMM_HPP
