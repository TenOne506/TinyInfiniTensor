#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto shapeA = inputs[0]->getDims();
        auto shapeB = inputs[1]->getDims();
        int rankA = inputs[0]->getRank();
        int rankB = inputs[1]->getRank();
        assert(rankA == rankB);
        if(this->transA) std::swap(shapeA[rankA-1],shapeA[rankA-2]);
        if(this->transB)  std::swap(shapeB[rankB-1],shapeB[rankB-2]);
        shapeA[rankA-1]=shapeB[rankB-1];
        return {{shapeA}};
    }

} // namespace infini