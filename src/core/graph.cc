#include "core/graph.h"
#include "core/op_type.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
  sorted = false;
  ops.push_back(op);
  for (auto &input : op->getInputs()) {
    if (input) {
      input->addTarget(op);
      if (auto pred = input->getSource()) {
        pred->addSuccessors(op);
        op->addPredecessors(pred);
      }
    }
  }
  for (auto &output : op->getOutputs()) {
    if (output) {
      output->setSource(op);
      for (auto &succ : output->getTargets()) {
        succ->addPredecessors(op);
        op->addSuccessors(succ);
      }
    }
  }
}

string GraphObj::toString() const {
  std::ostringstream oss;
  oss << "Graph Tensors:\n";
  for (const auto &tensor : tensors)
    oss << tensor << "\n";

  oss << "Graph operators:\n";
  for (const auto &op : ops) {
    vector<UidBaseType> preds, succs;
    for (auto &o : op->getPredecessors())
      preds.emplace_back(o->getGuid());
    for (auto &o : op->getSuccessors())
      succs.emplace_back(o->getGuid());
    oss << "OP " << op->getGuid();
    oss << ", pred " << vecToString(preds);
    oss << ", succ " << vecToString(succs);
    oss << ", " << op << "\n";
  }
  return oss.str();
}

bool GraphObj::topo_sort() {
  if (this->sorted) {
    return true;
  }
  std::vector<Operator> sorted;
  std::unordered_set<OperatorObj *> flags;
  sorted.reserve(ops.size());
  flags.reserve(ops.size());
  while (sorted.size() < ops.size()) {
    // Any node is move to sorted in this loop.
    auto modified = false;
    for (auto const &op : ops) {
      if (auto const &inputs = op->getInputs();
          flags.find(op.get()) == flags.end() &&
          std::all_of(inputs.begin(), inputs.end(),
                      [&flags](auto const &input) {
                        auto ptr = input->getSource().get();
                        return !ptr || flags.find(ptr) != flags.end();
                      })) {
        modified = true;
        sorted.emplace_back(op);
        flags.insert(op.get());
      }
    }
    if (!modified) {
      return false;
    }
  }
  this->ops = std::move(sorted);
  return this->sorted = true;
}

void GraphObj::optimize() {
  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来实现指定的图优化规则
  // 图优化规则如下：
  // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose
  // 算子，且做的是相反的操作，可以将其全部删除）
  // 2.
  // 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
  // =================================== 作业
  // ===================================
  if (!this->topo_sort()) {
    return;
  }

  int n_op = this->ops.size();
  for (int i = 0; i < n_op; ++i) {
    auto op = ops[i];
    // 处理 transpose
    if (op->getOpType() == OpType::Transpose) {
      auto op_trans = std::dynamic_pointer_cast<TransposeObj>(op);
      auto input = op->getInputs(0);
      auto pre_op = input->getSource();
      if (pre_op && pre_op->getOpType() == OpType::Transpose &&
          input->getTargets().size() == 1) {
        auto pre_op_trans = std::dynamic_pointer_cast<TransposeObj>(pre_op);
        auto pre_input = pre_op->getInputs(0);
        auto perm = op_trans->getPermute();
        bool flag = true;
        for (size_t j = 0; j < perm.size(); ++j) {
          perm[j] = pre_op_trans->getPermute()[perm[j]];
          if (perm[j] != int(j)) {
            flag = false;
          }
        }
        pre_input->removeTarget(pre_op);

        if (flag) {
          for (auto succ : op->getSuccessors()) {
            succ->replaceInput(op->getOutput(), pre_input);
            pre_input->addTarget(succ);
          }
          this->removeTensor(op->getOutput());
        } else {
          auto new_op =
              make_ref<TransposeObj>(this, pre_input, op->getOutput(), perm);
          this->addOperatorAndConnect(new_op);
        }

        for (auto pred : pre_op->getPredecessors()) {
          pred->removeSuccessors(pre_op);
        }

        for (auto succ : op->getSuccessors()) {
          succ->removePredecessors(op);
        }

        this->removeTensor(input);
        this->removeOperator(op);
        this->removeOperator(pre_op);

        i -= 2;
        n_op -= 2;
        continue;
      }
    }

    if (op->getOpType() == OpType::MatMul) {
      auto op_matmul = std::dynamic_pointer_cast<MatmulObj>(op);
      auto A = op->getInputs(0), B = op->getInputs(1);
      auto pre_op_A = A->getSource(), pre_op_B = B->getSource();
      if (pre_op_A && pre_op_A->getOpType() == OpType::Transpose &&
          A->getTargets().size() == 1) {
        auto pre_op_trans_A = std::dynamic_pointer_cast<TransposeObj>(pre_op_A);
        auto perm = pre_op_trans_A->getPermute();

        bool flag = true;
        for (size_t j = 0; j < perm.size() - 2; ++j) {
          if (perm[j] != int(j)) {
            flag = false;
            break;
          }
        }
        if (!flag || perm[perm.size() - 2] != int(perm.size() - 1) ||
            perm[perm.size() - 1] != int(perm.size() - 2)) {
          continue;
        }

        auto pre_input = pre_op_A->getInputs(0);
        op_matmul->setTransA(!op_matmul->getTransA());
        op_matmul->removePredecessors(pre_op_A);
        for (auto pre_pre : pre_op_A->getPredecessors()) {
          pre_pre->removeSuccessors(pre_op_A);
          pre_pre->addSuccessors(op);
          op->addPredecessors(pre_pre);
        }
        pre_input->removeTarget(pre_op_A);
        pre_input->addTarget(op);
        op_matmul->inputs[0] = pre_input;
        this->removeOperator(pre_op_A);
        this->removeTensor(A);
        i--;
        n_op--;
      }

      if (pre_op_B && pre_op_B->getOpType() == OpType::Transpose &&
          B->targets.size() == 1) {
        auto pre_op_trans_B = std::dynamic_pointer_cast<TransposeObj>(pre_op_B);
        auto perm = pre_op_trans_B->getPermute();

        // 检查是否只有最后两个维度有交换
        bool flag = true;
        for (size_t j = 0; j < perm.size() - 2; ++j) {
          if (perm[j] != int(j)) {
            flag = false;
            break;
          }
        }
        // 不满足条件
        if (!flag || perm[perm.size() - 2] != int(perm.size() - 1) ||
            perm[perm.size() - 1] != int(perm.size() - 2)) {
          continue;
        }

        // 满足条件，删除当前 trans_op 并融合到 matmul
        auto pre_input = pre_op_B->getInputs(0);
        op_matmul->setTransB(!op_matmul->getTransB());
        op_matmul->removePredecessors(pre_op_B);
        for (auto pre_pre : pre_op_B->getPredecessors()) {
          pre_pre->removeSuccessors(pre_op_B);
          pre_pre->addSuccessors(op);
          op->addPredecessors(pre_pre);
        }
        pre_input->removeTarget(pre_op_B);
        pre_input->addTarget(op);
        op_matmul->inputs[1] = pre_input;
        this->removeOperator(pre_op_B);
        this->removeTensor(B);
        i--;
        n_op--;
      }
    }
  }
}

Tensor GraphObj::getTensor(int fuid) const {
  for (auto tensor : tensors) {
    if (tensor->getFuid() == fuid) {
      return tensor;
    }
  }
  return nullptr;
}

void GraphObj::shape_infer() {
  for (auto &op : ops) {
    auto ans = op->inferShape();
    IT_ASSERT(ans.has_value());
    auto oldOutputs = op->getOutputs();
    IT_ASSERT(ans.value().size() == oldOutputs.size());
    // replace the old outputshape and size with new one
    for (int i = 0; i < (int)ans.value().size(); ++i) {
      auto newShape = ans.value()[i];
      auto oldShape = oldOutputs[i]->getDims();
      auto fuid = oldOutputs[i]->getFuid();
      if (newShape != oldShape) {
        auto tensor = this->getTensor(fuid);
        tensor->setShape(newShape);
      }
    }
  }
}

void GraphObj::dataMalloc() {
  // topological sorting first
  IT_ASSERT(topo_sort() == true);

  // =================================== 作业
  // ===================================
  // TODO：利用 allocator 给计算图分配内存
  // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor
  // 绑定内存
  // =================================== 作业
  // ===================================
  int n = this->tensors.size();
  vector<size_t> offsets(n);
  for (int i = 0; i < n; i++) {
    size_t size = this->tensors[i]->getBytes();
    offsets[i] = this->allocator.alloc(size);
  }
  auto hptr = this->allocator.getPtr();
  for (int i = 0; i < n; i++) {
    Blob blob = make_ref<BlobObj>(this->runtime, hptr + offsets[i]);
    this->tensors[i]->setDataBlob(blob);
  }
  allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
  return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
  IT_ASSERT(tensor->getRuntime() == runtime,
            std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                tensor->getRuntime()->toString() + " to " +
                runtime->toString());
  tensors.emplace_back(tensor);
  return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
  for (auto &t : tensors)
    addTensor(t);
  return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
  for (auto tensor : tensors) {
    IT_ASSERT(
        !(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));
    for (auto op : tensor->getTargets()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
    }
    auto op = tensor->getSource();
    IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
  }
  for (auto op : ops) {
    for (auto tensor : op->getInputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto tensor : op->getOutputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto pre : op->getPredecessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
    }
    for (auto suc : op->getSuccessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
    }
  }
  std::set<UidBaseType> s;
  // check whether two tensors with the same FUID exist
  for (auto tensor : tensors) {
    int cnt = s.count(tensor->getFuid());
    IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
    s.insert(tensor->getFuid());
  }
  return true;
}

} // namespace infini