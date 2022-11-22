#ifndef TEST_INTERFACES
#define TEST_INTERFACES

include "mlir/IR/OpBase.td"

/**
 * We do shape inference by: 
 * 1. inlining all of the functions (which takes generic shapes)
 * 2. this will reveal a bunch of places we can fold constants, and materialise the function to particular argument shapes
 * 3. function code now is applied to particular shapes, and we applied optimisations to particular shapes
 */

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}


#endif // TEST_INTERFACES