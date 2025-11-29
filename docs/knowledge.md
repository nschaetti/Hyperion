# Hyperion – Knowledge Requirements

## 1. Mathematics

### 1.1 Linear Algebra

* Matrix multiplication, linear maps, vector spaces
* Eigendecomposition, SVD
* Kronecker products
* Norms, conditioning, stability
  **References**
* Gilbert Strang — *Linear Algebra and Learning from Data*
* Trefethen & Bau — *Numerical Linear Algebra*

### 1.2 Calculus for Machine Learning

* Jacobians, Hessians, vector-Jacobian products
* Chain rule in tensor form
* Directional derivatives
* Gradient flow
  **References**
* Stanford CS229 Notes: *Matrix Calculus Review*
* “The Matrix Calculus You Need For Deep Learning” (Petersen & Pedersen)

### 1.3 Optimization

* Gradient descent, momentum
* Convexity basics
* Lipschitz continuity
* Loss landscapes
* Learning rate schedules
* Adam, RMSProp, adaptive methods
  **References**
* Boyd & Vandenberghe — *Convex Optimization* (Ch. 4–5)
* Wilson et al. — “The Marginal Value of Adaptive Methods”

### 1.4 Probability & Information Theory

* Random variables, expectations
* KL divergence, entropy
* Gaussian families
* Maximum likelihood
* Sampling methods
  **References**
* Cover & Thomas — *Elements of Information Theory*
* Murphy — *Probabilistic Machine Learning* (Vol. 1)

### 1.5 Deep Learning Theory

* Initialization theory (Xavier, Kaiming)
* Normalization theory
* Attention mechanisms
* Scaling laws
  **References**
* “Attention is All You Need”
* “Scaling Laws for Neural Language Models” (Kaplan et al.)
* “Chinchilla: Compute-Optimal Large Models” (Hoffmann et al.)

### 1.6 Numerical Methods (Optional but Valuable)

* Floating point arithmetic (IEEE 754)
* Stability vs precision
* Error propagation
  **References**
* Higham — *Accuracy and Stability of Numerical Algorithms*

# 2. Computer Science / Computing

## 2.1 Algorithms & Data Structures

* Sorting, hashing, maps
* Computational graphs
* Graph traversal (topological sort)
* Backtracking + recursion
  **References**
* CLRS — *Introduction to Algorithms* (Graph chapters)

## 2.2 Compiler & Interpreter Basics

Essentiel pour comprendre l’architecture interne d’un framework ML.

* AST
* Bytecode interpretation
* IR (Intermediate Representation)
  **References**
* Crafting Interpreters (Nystrom)

## 2.3 GPU Architecture

* SMs, warps, wavefronts
* Streaming multiprocessors
* Occupancy
* Memory hierarchy: global, shared, registers, constant
* Warp divergence
* Coalescing
  **References**
* NVIDIA CUDA Programming Guide
* “CUDA Best Practices Guide”

## 2.4 Parallel Programming

* CUDA kernels
* Thread hierarchy
* Blocks, grids
* Synchronization
* atomics, reductions
  **References**
* Udacity CUDA Course (free)
* Stanford CME343: GPU Computing

## 2.5 HPC / Performance Engineering

* Cache locality
* SIMD
* Tiling
* Prefetching
* Kernel fusion
* Profiling
  **References**
* *High Performance Computing* (Chapman)
* *Computer Systems: A Programmer’s Perspective* (Bryant & O’Hallaron)

## 2.6 Low-Level ML Systems

* Memory formats (contiguous, strides)
* Broadcasting
* Autograd internals
* Layout: NCHW, NHWC
  **References**
* PyTorch Documentation: *Autograd Mechanics*
* JAX documentation: *PyTrees, XLA*
* tinygrad (George Hotz) source code

# 3. Technology / Engineering

## 3.1 C++ (Modern)

* RAII
* Move semantics
* Smart pointers
* Templates
* constexpr
* Namespaces
* Build systems
  **References**
* Jason Turner — C++ Weekly
* *Effective Modern C++* (Scott Meyers)
* *A Tour of C++* (Bjarne Stroustrup)

## 3.2 CMake / Build Systems

* Targets
* Linking static and shared libs
* Multi-target repositories
* External libraries
  **References**
* *Professional CMake* (Craig Scott)
* CMake official docs

## 3.3 CUDA / GPU Programming

* Kernel launch configurations
* Memory transfers
* Shared memory tiling
* Tensor core usage (optional)
  **References**
* CUDA by Example
* *Programming Massively Parallel Processors* (Kirk & Hwu)

## 3.4 PyTorch Internals

* Tensor implementation
* Storage + views
* Autograd engine
* Dispatch keys
  **References**
* PyTorch internals tutorial (Zalando blog)
* PyTorch source code (`aten/`)

## 3.5 Neural Network Engineering

* Layer implementations
* Weight initialization
* Normalization techniques
* Masking logic
* Model parallelism basics
  **References**
* Karpathy: *makemore*, *minGPT*, *nanoGPT*
* *Dive into Deep Learning* (Aston Zhang)

## 3.6 Transformer Deep Knowledge

* Multi-head attention internals
* Scaling factors
* FlashAttention approximations
* Feed-forward networks
* RoPE embeddings
  **References**
* FlashAttention (Tri Dao)
* GPT-2 technical report
* *xformers* source code

## 3.7 FlashAttention Engineering

* IO-awareness
* Block-sparse memory access
* Streaming Q/K/V tiles
* Custom GPU kernels
  **References**
* FlashAttention-1 paper
* FlashAttention-2 paper
* Tri Dao’s GitHub implementations

## 3.8 Testing & Benchmarking

* GoogleTest / Catch2
* microbenchmarks
* profiling with nvprof
* nsight compute
  **References**
* GoogleTest documentation
* NVIDIA Nsight tutorials

## 3.9 Optional: Differentiable Simulation / PDE

* FDTD
* PML
* Finite differences
* Fourier Neural Operators
  **References**
* Raissi et al. — PINNs
* Li et al. — FNO
* Virieux — seismic wave propagation

# 4. The Mandatory Papers (Core of the Project)

### Core Deep Learning

* “Attention is All You Need”
* “Adam: A Method for Stochastic Optimization”
* “Batch Normalization”
* “Layer Normalization”

### Transformers

* “GPT-2”
* “GPT-3”
* “RoPE Embeddings”
* “FlashAttention 1 & 2”

### Systems ML

* “One weird trick to accelerate convolution”
* “XLA: Optimizing ML Compilers”
* “Triton: A Deep Learning Compiler”

### Autograd

* “Automatic differentiation in machine learning: a survey”

# 5. The Mandatory Resources (Videos / Repos)

### Karpathy

* *micrograd*
* *makemore*
* *minGPT*
* *nanoGPT*

### Tri Dao

* FlashAttention repo
* Talks on efficient attention

### George Hotz

* tinygrad
* YouTube livestreams on GPU kernels

### Chris Olah

* Distill.pub articles: circuits, attention

### NVIDIA

* Nsight tutorials
* CUDA samples
