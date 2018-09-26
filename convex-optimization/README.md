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
	- [Convex sets](#2-convex-sets) and [convex functions](#3-convex-functions) (Chapter 2 & 3)
	- [Convex optimization problems](#4-convex-optimization-problems) (Chapter 4)
	- [Lagrangian duality](#5-lagrangian-duality) (Chapter 5)

- Applications
- Algorithms
	-  Unconstrained optimization
	-  Equality constrained optimization
	-  Inequality constrained optimization

## 2. Convex sets
### 2.1 Definition
θx1 + (1−θ)x2 ∈ C for any x1 , x2 ∈ C and 0 ≤ θ ≤ 1

### 2.2 Examples
- __Hyperplanes and halfspaces__: {x|a<sup>T</sup>x = b}, {x|a<sup>T</sup>x ≤ b}
- __Euclidean balls and ellipsoids__: {x|||x−xc||2≤r} = {x│(x−xc)T(x−xc)≤r2}, {x│(x−xc)TP−1(x−xc)≤1}in which P is symmetric and positive definite.
- __Norm balls and norm cones__: {x|||x−xc||≤r}, {(x,t)|||x||≤t}
- __Polyhedra__: {x|aTjx≤bj,j=1,…,m,cTjx≤dj,j=1,…,p}
- __The positive semidefinite cone__: The set Sn+ is a convex cone: if θ1 ,θ2 ≥ 0 and A, B ∈ Sn+ , then θ1A + θ2B ∈ Sn+.

### 2.3 Operations that preserve convexity
- __Intersection__
- __Affine functions__: Has the form f(x) = Ax + b, where A ∈ Rm×n and b ∈ Rm 
- __Linear-fractional and perspective functions__: f(x)=(Ax+b)/(c<sup>T</sup>x+d) dom f = {x | c<sup>T</sup>x + d > 0} can be seen as stretching in the original set and Its inverse function also preserve convex, P(z,t)=z/t Similar to small hole imaging, it is a mapping from high dimension to low dimension.

### 2.4 Generalized inequalities
Defines a comparison between variables with multiple components:

x≥<sub>K</sub>y <=> x−y∈K
 
x><sub>K</sub>y <=> x−y∈int K


int K is the internal point of K, notice that K must be a closed, solid, pointed convex.

__Minimum__: Can be compared to all points in the collection and is minimal. 

__Minimal__: The smallest of all points that can be compared within a collection.

### 2.5 Separating and supporting hyperplanes

__Separating__: The hyperplane that can separate the two sets, and strictly, that is, the two sets have no intersection. There must be one split plane for the two convex sets.

__Supporting hyperplanes__: There is a point x0 at the edge of the set that makes a<sup>T</sup>x ≤ a<sup>T</sup>x0, Where x0 is a point within the collection, and a≠0.

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
- x ≤ Ky if and only if λ<sup>T</sup>x ≤ λTy for all λ ≥ K ∗ 0.
- x ≺ Ky if and only if λ<sup>T</sup>x < λTy for all λ ≥ K ∗ 0, λ ≠ 0.

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

A functions from n-dimensional to one-dimensional functions and meet the following conditions is called convex functions:

f(θx + (1 − θ)y) ≤ θf(x) + (1 − θ)f(y). Where 0 ≤ θ ≤ 1.

The one-dimensional derivative of a convex function has the following properties:

f(y) ≥ f(x) + ∇f(x)T(y−x)

It can be noticed that the right side of the equal sign is the first-order Taylor expansion of f on x. Besides, the second derivative of the convex function is greater than or equal to zero.

Examples of convex functions:
- Exponential e<sup>ax</sup>
- Powers x<sup>a</sup>, with x≥0, a<0 or a>1
- Powers of absolute value. |x|<sup>a</sup>, with a>1
- Logarithm. logx 
- Negative entropy. xlogx
- Norms.
- Max function. 
- Quadratic-over-linear function. f(x,y)=x2/y ，y>0
- Log-sum-exp. f(x)=log(e<sup>x<sub>1</sub></sup>+e<sup>x<sub>2</sub></sup>+,…,+e<sup>x<sub>n</sub></sup>), 
This is actually a soft-max
- Geometric mean. f(x)=(∏<sup>n</sup><sub>i=1</sub> x<sub>i</sub>)<sup>1/n</sup>
- Log-determinant. log|A| where the A matrix is ​​positive

Two other concepts:
- Sublevel sets: The corresponding domain when the function value is less than a certain threshold.
- Epigraph: {(x,t)|t≥f(x)}, This is the area above a piece of function.

Convex function satisfies jensen's inequality: f(θx + (1 − θ)y) ≤ θf(x) + (1 − θ)f(y)

### 3.2 Operations that preserve convexity
#### Nonnegative weighted sums
That is, multiply multiple functions by a non-negative weight: f = w<sub>1</sub>f<sub>1</sub> + ··· + w<sub>m</sub>f<sub>m</sub> 

#### Composition with an affine mapping
g(x)=f(Ax+b), If f is convex, then g is convex. If f is concave, g is also concave. That is, the affine after a affine is preserved.

#### Pointwise maximum and supremum
An operation is defined as the largest of the multiple functions with the same argument. The operation is convex.

#### Composition
f(x) = h(g(x)), domf = {x ∈ domg | g(x) ∈ domh}.
- If h is a convex function and is not subtracted, g is a convex function, then f is a convex function.
- If h is a convex function and non-increasing, and g is a concave function, then f is a convex function.
- If h is a concave function and is not subtracted, g is a concave function, then f is a concave function.
- If h is a concave function and is not increasing, and g is a convex function, then f is a concave function.


The expansion function of f is the same as above.

#### Minimization
g(x) = s`inf`<sub>(y∈C)</sub>f(x,y)

If f is convex in (x, y), then g is also convex.

#### Perspective of a function
g(x,t) = tf(x/t) where t>0

If f is convex, then g is convex, and if f is concave, g is also concave.

### 3.3 The conjugate function
f<sup>∗</sup>(y) = sup<sub>x</sub>(y<sup>T</sup>x−f(x)) -> The maximum distance from the plane y<sup>T</sup>x to f(x) given y

The maximum distance from y<sup>T</sup>x to f(x) occurs on the tangent of the slope of f(x) y.

The conjugate function of any function is convex because f∗(y) is the affine of y.

The conjugate function is obtained by bringing xy−f(x)into the expression of f(x)and deriving x. Make the derivation result equal to 0 (that is, find the maximum value of f(x)and xy distance), and substitute the result of x into f∗(y)=xy−f(x).

### 3.4 Quasi-convex functions

If the sublevel sets of a function are convex sets, then the function is a quasi-convex function.

Examples:
- Logarithm. logx
- Ceiling function. ceil(x) = inf{z ∈ Z | z ≥ x}
- Length of a vector.
- f(x1,x2) = x1x2 The Hessian matrix of the matrix is ​​neither positive nor negative, its lower level set is a convex set.
- Linear-fractional function. f(x)=(a<sup>T</sup>x+b)/(c<sup>T</sup>x+d) 
Non-minus or non-increasing, or if there is a point function ...

### 3.5 Log-concave and log-convex functions
If the value range of the function is greater than 0 and logf(x) is convex, then the function becomes a logarithmic convex function. The logarithmic concave function is the same.

- Affine function.
- Powers. f(x) = x<sup>a</sup> 
- Exponentials.
- The cumulative distribution function of a Gaussian density
- Gamma function.
- Determinant.
- Determinant over trace.


It should be noted that the multiplication of the two logarithmic convex functions is still a logarithmic convex function, but it is not necessarily added. The convolution operation of two logarithmic convex functions is a logarithmic convex function. The integral operation of a logarithmic convex function is a logarithmic convex function.

### 3.6 Convexity with respect to generalized inequalities
Matrix monotone functions:
- tr(WX), where W ∈ Sn. non-decreasing if W≥0, and 
- tr(X<sup>−1</sup>). decreasing on Sn++.
- detX. Increasing on Sn++, and non-decreasing on Sn+.

Convexity with respect to componentwise inequality: A function f :
R n → R m is convex with respect to componentwise inequality if and only if for all x, y and 0 ≤ θ ≤ 1, f(θx + (1 − θ)y) ≤ θf(x) + (1 − θ)f(y).

## 4. Convex optimization problems
### 4.1 Optimization problems
> _minimize　f0_(_x_)
> _subject to fi_(_x_)≤0,　i=1,…,m
> _hi_(_x_)=0,　i=1,…,p

Note that in addition to explicit constraints, each function has implicit domain constraints. The domain of the entire problem is the intersection of the domains of all functions. For each such problem, the optimal solution is defined as: p<sup>∗</sup> = inf{f0(x)|fi(x)≤0,i=1,…,m,hi(x)=0,i=1,…,p}

The local optimal solution is defined as the minimum value within the domain of the radius R. If this is the smallest problem, f0(x) is called the loss function, and if it is the biggest problem, it is called the utility function.

### 4.2 Convex optimization
A convex optimization problem is one of the form:
> minimize f0 (x)
> subject to fi (x) ≤ 0, i = 1,...,m
> a<sup>T</sup><sub>i</sub>x = b i , i = 1,...,p, where f 0 ,...,f m are convex functions

Comparison:
- the objective function must be convex,
- the inequality constraint functions must be convex,
- the equality constraint functions h<sub>i</sub>(x) = a<sup>T</sup><sub>i</sub>x − b<sub>i</sub> must be affine.

The feasible set of a convex optimization problem is convex, since it is the intersection of the domain of the problem which is a convex set.

If f0 is quasiconvex instead of convex, we say the problem is a quasiconvex optimization problem.

x is optimal if and only if x ∈ X and ∇f 0 (x)<sup>T</sup> (y − x) ≥ 0 for all y ∈ X. 

__Epigraph problem form__:
Sometimes in order to simplify theoretical analysis, the problem can be transformed into a linear objective function.
> minimize t
> subject to f0(x) − t ≤ 0
> fi(x) ≤ 0, i = 1,...,m
> a<sup>T</sup><sub>i</sub>x = bi, i = 1,...,p.

At the same time, the quasi-convex problem can be solved by:
> f0(x)≤t <=> ϕt(x)≤0
>
> find x
> subject to φt(x) ≤ 0
> fi(x)≤0, i=1,...,m
> Ax = b,

This method finds a t which has a feasible domain by Binary search on the domain, and the solved x is the suboptimal solution.

### 4.3 Linear optimization problems (LP)
A general linear program has the form:
> minimize c<sup>T</sup>x + d
> subject to Gx ≤ h
> Ax = b

Examples:
- Diet problem
- Chebyshev center of a polyhedron
- Dynamic activity planning
- Chebyshev inequalities
- Piecewise-linear minimization

### 4.4 Quadratic optimization problems (QP)
A quadratic program can be expressed in the form:
> minimize (1/2)x<sup>T</sup>Px+q<sup>T</sup>x+r
> subject to Gx ≤ h
> Ax = b,


If its inequality is constrained to a quadratic constraint, then the problem is __QCQP__: 
> minimize (1/2)x<sup>T</sup>P0x+q0<sup>T</sup>x+r0
> subject to (1/2)x<sup>T</sup>Pix+q<sup>T</sup><sub>i</sub>x+ri,　i=1,…,m
> Ax = b

Examples:
- Least-squares and regression
- Distance between polyhedra
- Bounding variance
- Linear program with random cost
- Markowitz portfolio optimization

Second-order cone programming (__SOCP__):
> minimize　f<sup>T</sup>x
> subject　to　||A<sub>i</sub>x+b||<sub>2</sub> ≤ c<sup>T</sup><sub>i</sub>x + d
> Fx=g

This plan can be used for linear programming with unknown constants

solutions: 
1. Set the covariance matrix P to indicate the degree of randomness of the parameters, and then set the constraint to tolerate its great loss.
2. Set the variance to follow the normal distribution and determine the constraint range by a confidence level of 0.95 or 0.99.

### 4.5 Geometric programming

Monomial function:
> f(x) = cx<sub>1</sub><sup>a<sub>1</sub></sup>x<sub>2</sub><sup>a<sub>2</sub></sup>...x<sub>n</sub><sup>a<sub>n</sub></sup>

Polynomials are defined as the sum of multiple monomials:
> f(x)=∑<sup>K</sup><sub>k=1</sub>c<sub>k</sub>x<sub>1</sub><sup>a<sub>1</sub>k</sup>x<sub>2</sub><sup>a<sub>2</sub>k</sup>...x<sub>n</sub><sup>a<sub>n</sub>k</sup>, where c<sub>k</sub> > 0

Geometric programming:
> minimize f0(x)
> subject to fi(x) ≤ 1, i = 1,...,m
> hi(x) = 1, i = 1,...,p
where f 0 ,...,f m are polynomials and h 1 ,...,h p are monomials.

Geometric programming is not a convex function, but can be converted into a convex function by  letting yi = logxi, and coat a log out of fi, the exponent of the problem is then converted into an affine function.

### 4.6 Generalized inequality constraints
Extending the constraint to the generalized inequality, that is, the mapping result of fi is a vector, and the inequality of fi is a generalized inequality

Conic form problems => Semidefinite programming (__SDP__)

> minimize c<sup>T</sup>x
> subject to x1F1 + ··· + xnFn + G ≤ 0
> Ax = b,

__LP => QCQP => SOCP => this__

### 4.7 Vector optimization
The confusing aspect of vector optimization is that the two objective values f0 (x) and f0 (y) need not be
comparable

Usually, the set of achievable objective values does not have a minimum
element. In these cases minimal elements of the set of achievable values play an important role.

Scalarization: 
> minimize λ<sup>T</sup>f0 (x)
> subject to fi (x) ≤ 0, i = 1,...,m
> hi (x) = 0, i = 1,...,p,

## 5. Duality
### 5.1 The Lagrange dual function