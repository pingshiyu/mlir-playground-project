#include "mlir/IR/Builders.h"

#include "TestDialect/TestOps.h"
#include "TestDialect/Interfaces.h"

namespace mlir { namespace test {

#include "TestDialect/Interfaces.cpp.inc"

Operation* TestInlinerInterface::materializeCallConversion(
    OpBuilder &builder, Value input,
    Type resultType,
    Location conversionLoc) const 
    {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }

void TestInlinerInterface::handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const 
    {
        // Only "toy.return" needs to be handled here.
        auto returnOp = cast<ReturnOp>(op);

        // Replace the values directly with the return operands.
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }

}}

