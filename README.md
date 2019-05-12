# Zip tree in Rust

## Overview

This project implements Tarjan's zip tree, a treap-like data structure
with different insertion and deletion algorithms. The node ranks are organized
like that in skip-list. You can visit [Tarjans's paper](https://arxiv.org/abs/1806.06726)
to learn more details.

## Implementation

The ZipTree API mimics standard library's BTreeMap interface. It provides insertion, deletion,
and iterator interface. Zip tree supports `clone()` via O(n) deep copy.

## License

The project is published with MIT license.
