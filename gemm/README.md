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

### GEMM AVX

The majors matter a lot when trying to vectorize. The optimal solution is to avoid having to reduce vectors to a single scalars (reduction to get the sum).

* However, This is only possible if we partially compute multiple elements of C at the same time:
   * Works great! no reductions:
      * CCR/CCC: vectorize an A column, use a scalar from B and partially compute a column of C
      * RCR/RRR: get an A scalar, vectorize a B row and partially compute a row of C
   * needs reduction, completely compute a C element at once:
      * CRC/RRC: vectorize an A row and a B column, get C result by summing up the elements in the resulting vector
   * don't vectorize well:
      * CRR/RCC: A rows don't map to B columns. We also cannot map a partial computation (A rows or B columns) to a C vector using avx
