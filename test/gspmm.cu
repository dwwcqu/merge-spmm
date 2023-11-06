#define GRB_USE_APSPIE
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>

#include "graphblas/mmio.hpp"
#include "graphblas/util.hpp"
#include "graphblas/graphblas.hpp"

#include <boost/program_options.hpp>
#include <test/test.hpp>

int main(int argc, char **argv)
{
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;

  // Parse arguments
  namespace po = boost::program_options;
  po::variables_map vm;
  parseArgs(argc, argv, vm);
  int TA, TB, NT, NUM_ITER, MAX_NCOLS;
  bool ROW_MAJOR, DEBUG;
  std::string mode;
  if (vm.count("ta"))
    TA = vm["ta"].as<int>();
  if (vm.count("tb"))
    TB = vm["tb"].as<int>();
  if (vm.count("nt"))
    NT = vm["nt"].as<int>();
  if (vm.count("max_ncols"))
    MAX_NCOLS = vm["max_ncols"].as<int>();

  // default values of TA, TB, NT will be used
  graphblas::Descriptor desc;
  // default mergepath algo
  desc.set(graphblas::GrB_MODE, graphblas::GrB_MERGEPATH);
  desc.set(graphblas::GrB_NT, NT);
  desc.set(graphblas::GrB_TA, TA);
  desc.set(graphblas::GrB_TB, TB);

  if (vm.count("debug"))
    DEBUG = vm["debug"].as<bool>();
  if (vm.count("iter"))
    NUM_ITER = vm["iter"].as<int>();
  if (vm.count("mode"))
  {
    mode = vm["mode"].as<std::string>();
  }

  // cuSPARSE (column major)
  if (mode == "cusparse")
  {
    ROW_MAJOR = false;
    desc.set(graphblas::GrB_MODE, graphblas::GrB_CUSPARSE);
    // fixed # of threads per row (row major)
  }
  else if (mode == "fixedrow")
  {
    ROW_MAJOR = true;
    desc.set(graphblas::GrB_MODE, graphblas::GrB_FIXEDROW);
    // fixed # of threads per column (col major)
  }
  else if (mode == "fixedcol")
  {
    ROW_MAJOR = false;
    desc.set(graphblas::GrB_MODE, graphblas::GrB_FIXEDCOL);
    // variable # of threads per row (row major)
  }
  else if (mode == "mergepath")
  {
    ROW_MAJOR = true;
    desc.set(graphblas::GrB_MODE, graphblas::GrB_MERGEPATH);
  }

  // Info
  if (DEBUG)
  {
    std::cout << "ta:    " << TA << "\n";
    std::cout << "tb:    " << TB << "\n";
    std::cout << "nt:    " << NT << "\n";
    std::cout << "row:   " << ROW_MAJOR << "\n";
    std::cout << "debug: " << DEBUG << "\n";
  }

  // Read in sparse matrix
  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  }
  else
  {
    readMtx(argv[argc - 1], row_indices, col_indices, values, nrows, ncols,
            nvals, DEBUG);
  }

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build(row_indices, col_indices, values, nvals);
  a.nrows(nrows);
  a.ncols(ncols);
  a.nvals(nvals);
  if (DEBUG)
    a.print();

  // Matrix B
  graphblas::Index MEM_SIZE = 1000000000; // 2x4=8GB GPU memory for dense
  graphblas::Index max_ncols = std::min(MEM_SIZE / nrows / 32 * 32, MAX_NCOLS);
  if (DEBUG)
    std::cout << "Restricting col to: " << max_ncols << std::endl;

  graphblas::Matrix<float> b(ncols, max_ncols);
  std::vector<float> denseVal;

  graphblas::Index a_nvals;
  a.nvals(a_nvals);
  int num_blocks = (a_nvals + NT - 1) / NT;
  int num_segreduce = num_blocks * max_ncols;
  CUDA(cudaMalloc(&desc.descriptor_.d_limits_,
                  (num_blocks + 1) * sizeof(graphblas::Index)));
  CUDA(cudaMalloc(&desc.descriptor_.d_carryin_,
                  num_blocks * max_ncols * sizeof(float)));
  CUDA(cudaMalloc(&desc.descriptor_.d_carryout_,
                  num_segreduce * sizeof(float)));

  // Row major order
  if (ROW_MAJOR)
    for (int i = 0; i < ncols; i++)
      for (int j = 0; j < max_ncols; j++)
      {
        if (i == j)
          denseVal.push_back(1.0);
        else
          denseVal.push_back(0.0);
      }
  else
    // Column major order
    for (int i = 0; i < max_ncols; i++)
      for (int j = 0; j < ncols; j++)
      {
        // denseVal.push_back(1.0);
        if (i == j)
          denseVal.push_back(1.0);
        else
          denseVal.push_back(0.0);
      }
  b.build(denseVal);
  if (DEBUG)
    b.print();
  graphblas::Matrix<float> c(nrows, max_ncols);
  graphblas::Semiring op;
  // spmm compute summary print
  std::cout << "A: " << nrows << ", " << ncols << ", " << nvals << std::endl;
  std::cout << "B: " << ncols << ", " << max_ncols << std::endl;

  // sparse matrix sparse degree print
  double degree = ((double )nvals) / ((double) nrows * ncols);
  std::cout << std::fixed << "Sparse Degree: " <<  degree * 100 << "%\n";
  graphblas::GpuTimer gpu_mxm;
  cudaProfilerStart();
  gpu_mxm.Start();
  graphblas::mxm<float, float, float>(c, graphblas::GrB_NULL, graphblas::GrB_NULL, op, a, b, desc);
  gpu_mxm.Stop();
  cudaProfilerStop();
  float elapsed_mxm = gpu_mxm.ElapsedMillis();
  std::cout << std::fixed << "mxm: " << elapsed_mxm << " ms\n";
  // print GFLOPS
  double gflops = 2.0 * ((double) max_ncols * nvals) / (elapsed_mxm / 1000.f);
  std::cout << std::fixed << "GFLOPS: " << gflops / 1000000000 << " GFLOPs" << std::endl;

  std::vector<float> out_denseVal;
  if (DEBUG)
    c.print();
  c.extractTuples(out_denseVal);
  int count = 0, correct = 0;
  for (int i = 0; i < nvals; i++)
  {
    graphblas::Index row = row_indices[i];
    graphblas::Index col = col_indices[i];
    float val = values[i];
    if (col < max_ncols)
    {
      count++;
      // Row major order
      if (ROW_MAJOR)
      {
        if (val != out_denseVal[row * max_ncols + col])
        {
          std::cout << row << " " << col << " " << val << " " << out_denseVal[row * max_ncols + col] << std::endl;
          correct++;
        }
        // BOOST_ASSERT( val==out_denseVal[row*max_ncols+col] );
      }
      else
        // Column major order
        // std::cout << row << " " << col << " " << val << " " << out_denseVal[col*nrows+row] << std::endl;
        BOOST_ASSERT(val == out_denseVal[col * nrows + row]);
    }
  }
  std::cout << "There were " << correct << " errors out of " << count << ".\n";
  CUDA(cudaFree(desc.descriptor_.d_limits_));
  CUDA(cudaFree(desc.descriptor_.d_carryin_));
  CUDA(cudaFree(desc.descriptor_.d_carryout_));
  return (correct == 0 ? 0 : 1);
}
