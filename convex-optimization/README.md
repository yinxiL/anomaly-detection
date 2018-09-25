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

## 2. Convex sets
### 2.1 Definition
θx1 + (1−θ)x2 ∈ C for any x1 , x2 ∈ C and 0 ≤ θ ≤ 1

### 2.2 Examples
- __Hyperplanes and halfspaces__: {x|aTx = b}, {x|aTx ≤ b}
- __Euclidean balls and ellipsoids__: {x|||x−xc||2≤r} = {x│(x−xc)T(x−xc)≤r2}, {x│(x−xc)TP−1(x−xc)≤1}in which P is symmetric and positive definite.
- __Norm balls and norm cones__: {x|||x−xc||≤r}, {(x,t)|||x||≤t}
- __Polyhedra__: {x|aTjx≤bj,j=1,…,m,cTjx≤dj,j=1,…,p}
- __The positive semidefinite cone__: The set Sn+ is a convex cone: if θ1 ,θ2 ≥ 0 and A, B ∈ Sn+ , then θ1A + θ2B ∈ Sn+.

### 2.3 Operations that preserve convexity
- __Intersection__
- __Affine functions__: Has the form f(x) = Ax + b, where A ∈ Rm×n and b ∈ Rm 
- __Linear-fractional and perspective functions__: f(x)=(Ax+b)/(cTx+d) dom f = {x | cTx + d > 0} can be seen as stretching in the original set and Its inverse function also preserve convex, P(z,t)=z/t Similar to small hole imaging, it is a mapping from high dimension to low dimension.

### 2.4 Generalized inequalities
Defines a comparison between variables with multiple components:

x≥<sub>K</sub>y <=> x−y∈K
 
x><sub>K</sub>y <=> x−y∈int K


int K is the internal point of K, notice that K must be a closed, solid, pointed convex.

__Minimum__: Can be compared to all points in the collection and is minimal. 

__Minimal__: The smallest of all points that can be compared within a collection.

### 2.5 Separating and supporting hyperplanes

__Separating__: The hyperplane that can separate the two sets, and strictly, that is, the two sets have no intersection. There must be one split plane for the two convex sets.

__Supporting hyperplanes__: There is a point x0 at the edge of the set that makes aTx ≤ aTx0, Where x0 is a point within the collection, and a≠0.

### 2.6 Dual cones and generalized inequalities
Let K be a cone. The set K∗ = {y | xTy ≥ 0 for all x ∈ K} is called the dual cone of K. As the name suggests, K∗ is a cone, and is always convex, even when the original cone K is not.

Examples of dual cones:
- Subspace
- Non-negative orthant
- Positive semidefinite cone
- Dual of a norm cone

Properties of dual cones:
- If K has nonempty interior, then K∗ is pointed.
- If the closure of K is pointed then K∗ has nonempty interior.
- K∗∗ is the closure of the convex hull of K. (Hence if K is convex and closed, K ∗∗ = K.)

Important properties relating a generalized inequality and its dual:
- x ≤ Ky if and only if λTx ≤ λTy for all λ ≥ K ∗ 0.
- x ≺ Ky if and only if λTx < λTy for all λ ≥ K ∗ 0, λ ≠ 0.

__Minimum element__: 
- __Properties__: x is the minimum element of S, with respect to the generalized inequality ≤ K , if and only if for all λ ≻ K∗ 0, x is the unique minimizer of λTz over z ∈ S. Geometrically, this means that for any λ ≻ K ∗ 0, {z | λT (z − x) = 0} is a strict supporting hyperplane to S at x.
- The point x is the minimum element of the set S with respect to R<sup>2</sup>+. This is __Equivalent to__
For every λ ≻ 0, the hyperplane {z | λ T (z − x) = 0} strictly supports S at x, i.e., contains S on one side, and touches it only at x.

__Minimal elements__:

If λ ≻ K∗0 and x minimizes λ<sup>T</sup>z over z ∈ S, then x is minimal.

Geometrically, it is very similar to the theorem of the smallest element, but this time there may be more than one supporting hyperplane.

The inverse proposition of this proposition is only established when S is convex. If S is not a convex set, then the minimal element x on S may not be the solution of minimizing λTz on z∈S for any λ.

A point x2 ∈ S2 can be not minimal, but does minimize
λTz over z ∈ S2 for λ = (0,1) ≥ 0 at the same time.


## 3. Convex functions
### 3.1 Basic properties and examples

