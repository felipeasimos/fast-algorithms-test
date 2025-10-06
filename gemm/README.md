# GEMM

General Matrix Multiplication.

The functions generally take the matrices `C`, `A` and `B`, alongside `ni`, `nj` and `nk` as arguments. The arguments relate to each other in this way:

```
C = A B
ni = number of rows in A and C
nj = number of columns in B and C
nk = number of columns in A and number of rows in B

C is (ni, nj)
A is (ni, nk)
B is (nk, nj)
```
