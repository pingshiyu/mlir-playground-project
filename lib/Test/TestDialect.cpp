//===------------- TestDialect.cpp - Test dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect/TestDialect.h"
#include "TestDialect/TestOps.h"

using namespace mlir;
using namespace mlir::test;

//===----------------------------------------------------------------------===//
// Test dialect.
//===----------------------------------------------------------------------===//

void TestDialect::initialize() {
  addOperations<
#define GET_OP_LIST // Check the file *Ops.cpp.inc. Setting this variable emits a list of variables
#include "TestDialect/TestOps.cpp.inc"
      >();
}

