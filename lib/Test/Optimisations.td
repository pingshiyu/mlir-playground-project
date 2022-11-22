#ifndef TEST_OPTIMISATIONS
#define TEST_OPTIMISATIONS

include "mlir/IR/PatternBase.td"
include "TestDialect/TestOps.td"

// Transpose(Transpose(x)) = x
def TransposeTransposeOptPattern : Pat<(TransposeOp (TransposeOp $arg)), (replaceWithValue $arg)>;

// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;

def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<mlir::ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (Test_ConstantOp $arg)),
  (Test_ConstantOp (ReshapeConstant $arg, $res))>;

#endif // TEST_OPTIMISATIONS