//===------------- TestDialect.cpp - Test dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"

#include "TestDialect/TestDialect.h"
#include "TestDialect/TestOps.h"
#include "TestDialect/TestOpsDialect.cpp.inc"

#include "TestDialect/Interfaces.h"

using namespace mlir;
using namespace mlir::test;

//===----------------------------------------------------------------------===//
// Test dialect.
//===----------------------------------------------------------------------===//

/// Parse an instance of a type registered to the test dialect.
mlir::Type TestDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!elementType.isa<mlir::TensorType, StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}

/// Print an instance of a type registered to the test dialect.
void TestDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // Currently the only test type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}

void TestDialect::initialize() {
  addOperations<
#define GET_OP_LIST // Check the file *Ops.cpp.inc. Setting this variable emits a list of variables
#include "TestDialect/TestOps.cpp.inc"
      >();
  addTypes<StructType>();
    
  addInterfaces<TestInlinerInterface>();
}

mlir::Operation *TestDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  if (type.isa<StructType>())
    return builder.create<StructConstantOp>(loc, type,
                                            value.cast<mlir::ArrayAttr>());
  return builder.create<ConstantOp>(loc, type,
                                    value.cast<mlir::DenseElementsAttr>());
}