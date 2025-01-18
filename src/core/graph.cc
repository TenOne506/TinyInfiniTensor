#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <iostream>
namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
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

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

        void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        
        // 优化规则1：去除冗余的transpose算子
        for (auto opIter = ops.begin(); opIter != ops.end(); ) {
            auto op = *opIter;
            if (op->getOpType() == OpType::Transpose) {
                auto successors = op->getSuccessors();
                if (successors.size() == 1) {
                    auto nextOp = successors[0];
                    if (nextOp->getOpType() == OpType::Transpose) {
                        // 检查是否可以安全删除
                        bool canRemove = true;
                        for (auto afterOp = opIter + 1; afterOp != ops.end(); ++afterOp) {
                            for (auto pred : (*afterOp)->getPredecessors()) {
                                if (afterOp->get()->getGuid() != nextOp->getGuid()
                                &&pred->getGuid() == op->getGuid()) {
                                    canRemove = false;
                                    break;
                                }
                            }
                            if (!canRemove) break;
                        }

                        if (canRemove) {
                            // 更新连接关系
                            auto inputTensor = op->getInputs()[0];
                            auto outputTensor = nextOp->getOutputs()[0];
                            
                            inputTensor->removeTarget(op);
                            // 将input直接连接到nextOp的输出
                            for (auto target : outputTensor->getTargets()) {
                                inputTensor->addTarget(target);
                                target->removePredecessors(nextOp);
                                target->replaceInput(target->getInputs().front(), inputTensor);
                                if(inputTensor->getSource()){
                                    target->addPredecessors(inputTensor->getSource());
                                }
                                
                            }

                            // 删除中间tensor和算子
                            removeTensor(op->getOutputs()[0]);
                            removeTensor(outputTensor);
                            removeOperator(op);
                            removeOperator(nextOp);
                            
                            // 更新迭代器
                            //opIter = ops.begin();
                            continue;
                        }
                    }
                }
            }
            ++opIter;
        }

        //优化规则2：合并transpose到matmul
        for (auto opIter = ops.begin(); opIter != ops.end(); ) {
            auto op = *opIter;
            if (op->getOpType() == OpType::MatMul) {
                auto matmulOp = dynamic_cast<infini::MatmulObj*>(op.get());
                if (!matmulOp) continue;
                
                // // 检查输入A
                auto inputA = op->getInputs()[0];
                if (inputA->getSource() && inputA->getSource()->getOpType() == OpType::Transpose) {
                    auto transposeOp = inputA->getSource();
                    if(!transposeOp) continue;
                    auto transpose_input = transposeOp->getInputs()[0];
                    if(!transpose_input) continue;
                    // 检查transpose是否只交换最后两个维度
                    auto permute = dynamic_cast<infini::TransposeObj*>(transposeOp.get())->getPermute();
                    if (permute.size() >= 2 && 
                        permute[permute.size()-2] == (int)permute.size()-1 &&
                        permute[permute.size()-1] == (int)permute.size()-2) {
                       //更新A转之前的目标
                        transpose_input->removeTarget(transposeOp);
                        transpose_input->addTarget(op);
                        // 更新连接关系
                        matmulOp->removePredecessors(transposeOp);
                        matmulOp->replaceInput(inputA, transpose_input);
                        if(transpose_input->getSource()){
                            matmulOp->addPredecessors(transpose_input->getSource());
                        }
                        // 更新matmul的transB属性
                        matmulOp->setTransB(!matmulOp->getTransB());
                        //删除transpose张量
                        removeTensor(transposeOp->getOutput());
                        // 删除transpose算子
                        removeOperator(transposeOp);

                        opIter = ops.begin();
                        continue;
                    }
                }

                // 检查输入B
                auto inputB = op->getInputs()[1];
                if (inputB->getSource() && inputB->getSource()->getOpType() == OpType::Transpose) {
                    auto transposeOp = inputB->getSource();
                    if(!transposeOp) continue;
                    auto transpose_input = transposeOp->getInputs()[0];
                    if(!transpose_input) continue;
                    // 检查transpose是否只交换最后两个维度
                    auto permute = dynamic_cast<infini::TransposeObj*>(transposeOp.get())->getPermute();
                    if (permute.size() >= 2 && 
                        permute[permute.size()-2] == (int)permute.size()-1 &&
                        permute[permute.size()-1] == (int)permute.size()-2) {
                        //更新B转之前的目标
                        transpose_input->removeTarget(transposeOp);
                        transpose_input->addTarget(op);
                        // 更新连接关系
                        matmulOp->removePredecessors(transposeOp);
                        matmulOp->replaceInput(inputB, transpose_input);
                        if(transpose_input->getSource()){
                            matmulOp->addPredecessors(transpose_input->getSource());
                        }
                        // 更新matmul的transB属性
                        matmulOp->setTransB(!matmulOp->getTransB());
                        //删除transpose张量
                        removeTensor(transposeOp->getOutput());
                        // 删除transpose算子
                        removeOperator(transposeOp);

                        opIter = ops.begin();
                        continue;
                    }
                }
            }
            ++opIter;
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        auto n = this->tensors.size();
        vector<size_t> heads(n);
        for (size_t i = 0; i < n; i++) {
            heads[i] = this->allocator.alloc(this->tensors[i]->getBytes());
        }
        auto baseptr = this->allocator.getPtr();
        for (size_t i = 0; i < n; i++) {
            auto beginptr = baseptr + heads[i];
            auto blob = make_ref<BlobObj>(this->runtime, beginptr);
            this->tensors[i]->setDataBlob(blob);
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini