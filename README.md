# ITensorsDMT
Run Time-Evolving Block Decimation using Density Matrix Truncations (DMT) on MPDOs (Matrix Product Density Operators). 

This package implements [DMT](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.035127) on an MPS encoding of a density matrix. To do this, it introduces operator site types, such as "PauliOperator", which span the vector space of operators and a `superoperator` function which converts operators to superoperators.

See `test/runtests.org` for usage examples.
