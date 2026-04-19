# Plan: Add GGUF to MLX Adapter Infrastructure

## Goal
Create basic scaffolding files for a GGUF to MLX model adapter, establishing the infrastructure flow without implementing core translation logic. This sets up the interface and structure for future MLX model support in Ollama.

- see (Mlx Adapter)[https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama_mlx.py]
- see (MLX adotion in ollama)[https://ollama.com/blog/mlx]

## Constraints
- Follow Windsurf global rules
- Avoid over-engineering
- Use trusted libraries
- Apply TDD for every chunk
- Focus on minimal viable infrastructure
- Follow llama.cpp adapter patterns

## Assumptions
- MLX framework is available and compatible with the build system
- GGUF loading interfaces are already established in llama.cpp
- Adapter pattern from llama-adapter.h/.cpp is the correct approach
- User will implement the actual MLX translation logic after infrastructure is in place

## Breakdown (Small, Testable Chunks)
### Chunk 1: Create MLX Adapter Header File
- Description: Define the interface for GGUF to MLX adapter following llama_adapter pattern
- Inputs: llama-adapter.h as reference
- Outputs: llama-adapter-mlx.h with basic class declaration
- Tests to Add/Update: None (interface only)
- Definition of Done: Header compiles without errors, includes basic MLX includes and forward declarations

### Chunk 2: Create MLX Adapter Implementation Stub
- Description: Implement basic adapter class with empty methods for GGUF loading and MLX translation
- Inputs: llama-adapter.cpp as reference
- Outputs: llama-adapter-mlx.cpp with skeleton implementation
- Tests to Add/Update: Basic compilation test
- Definition of Done: File compiles, all methods return appropriate stub values, integrates with build system

### Chunk 3: Update CMakeLists for MLX Support
- Description: Add conditional compilation flags and MLX library linking to CMakeLists.txt
- Inputs: Existing CMakeLists.txt
- Outputs: Modified CMakeLists.txt with MLX detection and compilation options
- Tests to Add/Update: Build test with MLX enabled
- Definition of Done: CMake configuration succeeds with MLX support enabled, basic build works

### Chunk 4: Create Basic MLX Model Loader Stub
- Description: Add minimal MLX model loading interface that can be extended
- Inputs: llama-model-loader.cpp as reference
- Outputs: llama-mlx-loader.h/.cpp with stub functions
- Tests to Add/Update: Loader interface test
- Definition of Done: Loader integrates with existing model loading flow, compiles successfully

## Risks / Open Questions
- MLX library availability and version compatibility
- How MLX tensors map to GGUF quantization schemes
- Performance implications of translation layer
- Whether this should be a separate backend or adapter pattern

## Ledger
Link to: /AGENT_WHAT_I_DID.md
