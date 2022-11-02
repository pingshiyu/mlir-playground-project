#ifndef TEST_DIALECT_OPTIMISATIONS
#define TEST_DIALECT_OPTIMISATIONS

#include <iostream>
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "TestOps.h"

using namespace mlir::test;

struct RemoveRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    RemoveRedundantTranspose(mlir::MLIRContext *context)
     : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

    mlir::LogicalResult matchAndRewrite(
        TransposeOp op,
        mlir::PatternRewriter &rewriter) const override {
            mlir::Value operand = op.getOperand();

            // see if the operation in the operand is Transpose as well
            TransposeOp transposeOpCastOperand = operand.getDefiningOp<TransposeOp>();
            if (!transposeOpCastOperand) return mlir::LogicalResult::failure();

            // it is also a transpose: replace with whatever is in the inner transpose's operand
            rewriter.replaceOp(op, {transposeOpCastOperand.getOperand()});
            return mlir::LogicalResult::success();
    }
};

#endif // TEST_DIALECT_OPTIMISATIONS