# Hyperion: Development Philosophy and Learning Goals

Hyperion is not a production deep learning framework.
It is a **learning engine** – a project built to understand, from first principles, how modern machine learning systems work internally.

The goal is to reconstruct the full stack:

* tensor operations
* memory layouts
* autograd and computation graphs
* neural network layers
* optimizers
* attention mechanisms
* a complete Transformer
* custom CUDA kernels
* FlashAttention-like optimizations

The project takes inspiration from PyTorch, tinygrad, minGPT, Triton, and classical HPC design, but everything is rebuilt deliberately, step by step, to achieve true systems-level understanding.

Hyperion is designed primarily for **technical depth**, not convenience or speed.

## Guiding Principles

### 1. Rebuild the Essential Parts Yourself

Hyperion deliberately avoids using high-level abstractions or pre-existing deep learning libraries.
The goal is to implement the core mechanisms *explicitly*:

* Tensor representation (shape, strides, dtype, device)
* Manual storage and memory management
* Dynamic autograd
* Operators with well-defined forward/backward passes
* CPU and CUDA kernels
* Transformer blocks and attention internals

These components form the "sacred core" of the framework and are written from scratch to gain complete control and understanding.

### 2. Use Modern C++ When It’s Helpful, Not When It’s Distracting

Hyperion does not aim to re-implement the entire C++ standard library.
The project focuses on ML-specific mechanisms; therefore:

**Use STL for:**

* `std::vector`, `std::array`
* `std::string`, `std::string_view`
* `std::unordered_map`, `std::map`
* smart pointers (`unique_ptr`, `shared_ptr`)
* `std::optional`, `std::variant`, `std::expected`
* `std::chrono`, `std::filesystem`, `std::thread`

These tools do not interfere with the learning goals and make the project more maintainable.

**Reimplement yourself:**

* Tensor storage and memory allocator
* Shape/stride logic
* Broadcasting rules
* Autograd graph internals
* Internal buffers and lightweight containers
* Execution and kernel dispatch
* CUDA kernels (matmul, softmax, layernorm, attention)
* GPU memory management

These pieces are *fundamental* to understanding ML systems, and rewriting them is essential.

### 3. Embrace Experiments

Hyperion includes a `playground/` directory where experimental `.cpp` files can be created without restriction.
This is where ideas are tested, debugged, benchmarked, or simply explored.
Anything in `playground/` does **not** need to be clean or final.

It is the space where learning happens.

### 4. Code Clarity Over Performance

Although performance matters (especially in the CUDA backend), Hyperion prioritizes:

* correctness
* readability
* explicit logic
* well-documented internals
* reproducibility of experiments

Optimizations are added **after** correctness is achieved, not before.

### 5. Keep Architecture Modular

The project is structured into:

* `core/` (tensor, dtype, device, storage)
* `autograd/` (graph, nodes, backward engine)
* `nn/` (layers, attention, transformer blocks)
* `optim/` (optimizers)
* `cuda/` (GPU kernels)
* `utils/` (profiling, timers, logs)
* `tests/` (unit tests)
* `playground/` (exploratory development)

This architecture allows Hyperion to scale as the project grows, and to serve as a readable educational reference.

## What Hyperion Is Trying to Achieve

Hyperion serves three purposes:

### 1. A Deep Learning Framework You Fully Understand

Every part of Hyperion is built by hand.
There is no “magic box.”
All layers, operators, and kernels are traceable and hackable.

### 2. A Vehicle for Becoming a Machine Learning Systems Expert

Building Hyperion means learning:

* Autograd internals
* Tensor memory layouts
* GPU programming
* High-performance kernels
* Attention implementations
* Full Transformer architectures
* ML runtime design

This combination of knowledge places a developer into a rare category of ML systems engineers.

### 3. An Educational Tool for Others

Hyperion aims to become a clean, well-documented reference for anyone interested in understanding deep learning at a low level.
The project will include:

* clear examples
* diagrams
* explanations
* notes on architecture and design
* reproducible experiments

## What Hyperion Is Not

* It is **not** a new PyTorch.
* It is **not** intended for production workloads.
* It is **not** focused on implementing every ML operation.
* It does **not** aim to achieve state-of-the-art performance.

Hyperion’s value is conceptual, educational, architectural, and experimental.

## Long-Term Vision

Once the core system is complete, Hyperion may expand to:

* a differentiable physics backend
* PDE solvers
* custom operators for simulation-based inference
* deep profiling and visualization tools
* custom compiler passes or kernel fusion
* an explanatory website or book

The ambition is to create one of the clearest machine learning systems implementations available publicly.
