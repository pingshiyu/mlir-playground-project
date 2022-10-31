#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

#include "ast/Parser.h"

#include "TestDialect/MLIRGen.h"
#include "TestDialect/TestDialect.h"
#include "TestDialect/TestOpsDialect.cpp.inc"

using namespace test;
namespace cl = llvm::cl;

// ===========================================================================
// =========== Grabbing CMD args =============================================
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input test file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
namespace {
  enum InputType { Test, MLIR };
  enum Action { None, DumpAST, DumpMLIR, DumpTest };
} // namespace

static cl::opt<enum InputType> inputType(
    "x", cl::init(Test), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Test, "test", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
               cl::values(clEnumValN(DumpMLIR, "mlir", "output MLIR")),
               cl::values(clEnumValN(DumpTest, "test", "output Test dialect")));

// ===========================================================================

/// Returns a Test AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<test::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int dumpAST() {
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  // just prints out to stderr
  dump(*moduleAST);

  return 0;
}

int dumpTestDialect() {
  // Get the context and load our dialect in
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::test::TestDialect>();

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 6;
  
  // generate a MLIR Module object from AST. Convert everything into Ops
  mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
  if (!module)
    return 1;

  // print out to stderr
  module->dump();
  return 0;
}

int dumpMLIR() {
  // Get the context and load our dialect in
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::test::TestDialect>();

  // File type needs to be .mlir!
  if (llvm::StringRef(inputFilename).endswith(".mlir")) {
    llvm::errs() << "Gotta be a .MLIR file! \n";
    return -1;
  }
  
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  // this just calls the built-in MLIR parser into `module`.
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  module->dump();
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "test compiler\n");

  switch (emitAction) {
    case Action::DumpAST:
      return dumpAST();
    case Action::DumpMLIR:
      return dumpMLIR();
    case Action::DumpTest:
      return dumpTestDialect();
    default:
      llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}