#ifndef TEST_SHAPE_INFERENCE_PASS
#define TEST_SHAPE_INFERENCE_PASS

#include "mlir/Pass/Pass.h"

namespace mlir { namespace test {

// lower Test dialect to Affine dialect
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

// Inline functions and specialise to call site's shapes
std::unique_ptr<mlir::Pass> createShapeInferencePass();

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

}}

#endif // TEST_SHAPE_INFERENCE_PASS