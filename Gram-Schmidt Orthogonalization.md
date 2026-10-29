# Gram-Schmidt Orthonormalization

## Core idea (one line)
Gram-Schmidt takes any set of independent vectors and turns them into an
**orthonormal** set (mutually perpendicular, each length 1) that spans the
same space.

## The two operations
Repeated for each vector:
1. **Subtract projections** → make it perpendicular to the previous vectors (orthogonal)
2. **Divide by its length** → make it unit length (normal)

orthogonal + normal = orthonormal.

## The procedure
Given independent vectors v₁, v₂, ..., build orthonormal u₁, u₂, ...

**Step 1 — normalize the first vector:**
$$u_1 = \frac{v_1}{\|v_1\|}$$

**Step 2 — remove v₂'s component along u₁, then normalize:**
$$w_2 = v_2 - (v_2^T u_1)\,u_1 \qquad u_2 = \frac{w_2}{\|w_2\|}$$

**Step k (general) — remove components along ALL previous u's, then normalize:**
$$w_k = v_k - \sum_{j<k}(v_k^T u_j)\,u_j \qquad u_k = \frac{w_k}{\|w_k\|}$$

The sum subtracts the part of vₖ lying along every u you've already built, so
what's left is perpendicular to all of them. Then normalize.

## Why the subtraction makes it perpendicular
(vₖᵀuⱼ)·uⱼ is the projection of vₖ onto uⱼ — "how much of vₖ points along uⱼ."
Subtracting it removes that overlap, leaving a vector perpendicular to uⱼ.
This is the same "drop a perpendicular" idea as least squares projection.

## Worked example
v₁ = [3, 0],  v₂ = [2, 2]

Step 1:  ‖v₁‖ = 3  →  u₁ = [1, 0]
Step 2:  v₂ᵀu₁ = 2
         w₂ = [2,2] − 2·[1,0] = [0,2]   (check ⊥ u₁: [0,2]·[1,0]=0 ✓)
         ‖w₂‖ = 2  →  u₂ = [0, 1]

Result: {[1,0], [0,1]} — orthonormal ✓

## Verify orthonormality
u₁ᵀu₁ = 1,   u₂ᵀu₂ = 1   (unit length)
u₁ᵀu₂ = 0               (perpendicular)
Equivalently, stacking as columns of U gives UᵀU = I.

## In code
```python
import numpy as np
import numpy.linalg as la

v1 = m[:, 0].astype(float)
v2 = m[:, 1].astype(float)

u1 = v1 / la.norm(v1)                 # normalize first
w2 = v2 - (v2 @ u1) * u1              # subtract projection onto u1
u2 = w2 / la.norm(w2)                 # normalize what's left
```
(`.astype(float)` avoids integer-division bugs when dividing by the length.)

## Connections
- Each subtraction = a projection (vₖᵀuⱼ)uⱼ — same operator as least squares.
- The result U has orthonormal columns → UᵀU = I.
- QR decomposition (np.linalg.qr) does Gram-Schmidt automatically and more
  stably — Q's columns are the orthonormal basis.
- Used to build orthonormal bases for subspaces (e.g. projection onto a plane),
  which simplifies the projection matrix from V(VᵀV)⁻¹Vᵀ to the clean UUᵀ.

## Where I got confused / gotchas
- The subtraction must use the **already-normalized** uⱼ (unit vectors), so you
  don't divide by ‖uⱼ‖² in the projection — it's just (vₖᵀuⱼ)uⱼ.
- For k vectors, each new one subtracts projections onto **all** previous u's,
  not just the one before it.
- `.astype(float)` before dividing — integer arrays truncate and break it.
- Normalizing (÷ length) is what makes it "ortho**normal**" vs just "orthogonal."

## Where I got confused
- Forgot it's **√λ**, not λ. Fix: "the norm squares it," so √λ in → λ out.
  Sanity check: ‖√λ·w‖² = λ‖w‖² ✓   (‖λ·w‖² = λ²‖w‖² would be wrong).
- Missed the **bridge fact** (stacking adds squared norms) — that's the whole
  key to the augmentation trick. Memorize that ONE fact and you can rederive
  everything.
