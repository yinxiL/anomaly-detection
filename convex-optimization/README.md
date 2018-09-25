# Convex Optimization

Notes on reading _Convex Optimization_ by Stephen Boyd.

## 1. Introduction
### 1.1 Mathematical optimization
Include least-squares and linear programming problems.

New interests in the topic: 

1. Interior-point methods allow us to solve semi-definite programs and second-order
cone programs almost as easily as linear programs.

2. Beyond least-squares and linear programs are more prevalent in practice than was previously thought. (automatic control systems, estimation and signal processing, communications and networks, electronic circuit design, data analysis and modeling,
statistics, and finance)

### 1.2. Least-squares and linear programming

A least-squares problem is an optimization problem with no constraints (i.e., m = 0) and an objective which is a sum of squares of terms of the form a<sup>T</sup><sub>i</sub>x − b<sub>i</sub>

The least-squares problem can be solved in a time approximately proportional to n<sup>2</sup>k, with a known constant.

A current desktop computer can solve a least-squares problem with hundreds of variables, and thousands of terms, in a few seconds.

In linear programming, the objective and all constraint functions are linear: minimize c<sup>T</sup>x, subject to a<sup>T</sup><sub>i</sub>x ≤ b<sub>i</sub>, i = 1,...,m.

### 1.3 Convex optimization

A convex optimization problem: f<sub>i</sub> (αx + βy) ≤ αf<sub>i</sub> (x) + βf<sub>i</sub> (y) , with α+β = 1, α ≥ 0, β ≥ 0. 

The challenge, and art, in using convex optimization is in recognizing and formulating the problem. Once this formulation is done, solving the problem is, like
least-squares or linear programming, (almost) technology.

### 1.4 Nonlinear optimization

There are no effective methods for solving
the general nonlinear programming problem. Even simple looking problems with as few as ten variables can be extremely challenging, while problems with a few hundreds of variables can be intractable.

Methods for the general nonlinear
programming problem therefore take several different approaches, each of which
involves some compromise.

- Local optimization: Can be fast, can handle large-scale problems, and are widely applicable. But not finding the true, globally optimal solution, and require an initial
guess for the optimization variable.
- Global optimization: The compromise is efficiency

Convex optimization also plays an important role in problems that are not convex:
- Initialization for local optimization
- Convex heuristics for non-convex optimization
- Bounds for global optimization: Simple heuristics to sparse solutions (Chapter 6), Randomized algorithms
- Bounds for global optimization: Lagrangian relaxation (Chapter 5)

### 1.5 Outline
- Theory
	- Convex sets and convex functions (Chapter 2 & 3)
	- Convex optimization problems (Chapter 4)
	- Lagrangian duality (Chapter 5)

- Applications
- Algorithms
	-  Unconstrained optimization
	-  Equality constrained optimization
	-  Inequality constrained optimization




