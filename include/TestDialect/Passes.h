#ifndef TEST_SHAPE_INFERENCE_PASS
#define TEST_SHAPE_INFERENCE_PASS

#include "mlir/Pass/Pass.h"

namespace mlir { 

std::unique_ptr<mlir::Pass> createShapeInferencePass();

}

#endif // TEST_SHAPE_INFERENCE_PASS