#include "operators/matmul.h"

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA),
      transB(transB) {
  IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
  std::ostringstream os;
  os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
     << ",A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
     << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << "," << k
     << "])";
  return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
  // =================================== 作业
  // ===================================
  // TODO：返回经过 matmul 操作后的 shape
  // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
  // =================================== 作业
  // ===================================
  Shape A_dims = inputs[0]->getDims();
  Shape B_dims = inputs[1]->getDims();
  int n = inputs[0]->getRank(), m = inputs[1]->getRank();
  int k = std::max(n, m);
  Shape C_dims(k);
  for (int i = n - 1, j = m - 1; i >= 0 || j >= 0; i--, j--) {
    int a = (i >= 0 ? A_dims[i] : 1);
    int b = (j >= 0 ? B_dims[j] : 1);
    if (a == b)
      C_dims[std::max(i, j)] = a;
    else if (a == 1 || b == 1)
      C_dims[std::max(i, j)] = std::max(a, b);
  }
  if (this->getTransA())
    C_dims[k - 2] = A_dims[n - 1];
  else
    C_dims[k - 2] = (n - 2 >= 0 ? A_dims[n - 2] : 1);
  if (this->getTransB())
    C_dims[k - 1] = (m - 2 >= 0 ? B_dims[m - 2] : 1);
  else
    C_dims[k - 1] = B_dims[m - 1];

  return {{C_dims}};
}

} // namespace infini