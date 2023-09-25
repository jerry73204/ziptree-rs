//! Tarjan's zip tree implementation in Rust.
//!
//! Zip tree can be seen as a variant of treap with different node rank distributions
//! and insertion and deletion algorithms. It is isomorphic to skip list, but
//! takes less space as each node only needs to store its left and right subtrees
//! and O(log log n) bit for its rank. Insertion and deletion are done by _zip_
//! and _unzip_ operations instead of a series of tree rotations. You can see
//! [1](https://arxiv.org/abs/1806.06726) and [2](https://arxiv.org/abs/2307.07660)
//! for more details.

use rand::distributions::{Distribution, Standard};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::fmt::Debug;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
struct Rank(u16);

impl Distribution<Rank> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Rank {
        let r1 = rng.gen::<u64>().leading_zeros() as u16;
        let r2 = rng.gen::<u16>() & 0x3FFu16;
        Rank((r1 << 10) + r2)
    }
}

type Tree<K, V> = Option<Box<Node<K, V>>>;

#[derive(Debug, Clone)]
struct Node<K, V> {
    key: K,
    value: V,
    rank: Rank,
    left: Tree<K, V>,
    right: Tree<K, V>,
}

impl<K, V> Node<K, V> {
    fn new(key: K, value: V, rank: Rank) -> Box<Self> {
        Box::new(Node {
            key,
            value,
            rank,
            left: None,
            right: None,
        })
    }
}

enum Side {
    Left,
    Right,
}

/// Tarjan's zip tree implementation.
///
/// The ZipTree API mimics the standard library's BTreeMap. It provides look-ups, insertions,
/// deletions and iterator interface. Cloning this tree will deep copy overall tree structure,
/// and thus takes O(n) time.

#[derive(Debug)]
pub struct ZipTree<K, V> {
    root: Tree<K, V>,
    length: usize,
    rng: SmallRng,
}

impl<K, V> ZipTree<K, V> {
    pub fn new() -> Self {
        ZipTree {
            root: None,
            length: 0,
            rng: SmallRng::from_entropy(),
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub fn clear(&mut self) {
        self.root = None;
        self.length = 0;
    }
}

impl<K, V> ZipTree<K, V>
where
    K: Ord,
{
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut cur = self.root.borrow();
        while let Some(node) = cur {
            match key.cmp(node.key.borrow()) {
                Ordering::Equal => return Some(&node.value),
                Ordering::Less => cur = &node.left,
                Ordering::Greater => cur = &node.right,
            }
        }

        None
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut cur = self.root.borrow_mut();
        while let Some(node) = cur {
            match key.cmp(node.key.borrow()) {
                Ordering::Equal => return Some(&mut node.value),
                Ordering::Less => cur = &mut node.left,
                Ordering::Greater => cur = &mut node.right,
            }
        }

        None
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    fn unzip(
        mut side: Side,
        key: &K,
        mut left_tree: &mut Tree<K, V>,
        mut right_tree: &mut Tree<K, V>,
    ) -> Option<V> {
        while let Some(node) = match side {
            Side::Left => left_tree.take(),
            Side::Right => right_tree.take(),
        } {
            match key.cmp(&node.key) {
                Ordering::Equal => {
                    let _ = std::mem::replace(left_tree, node.left);
                    let _ = std::mem::replace(right_tree, node.right);
                    return Some(node.value);
                }
                Ordering::Less => {
                    right_tree = &mut right_tree.insert(node).left;
                    side = Side::Right;
                }
                Ordering::Greater => {
                    left_tree = &mut left_tree.insert(node).right;
                    side = Side::Left;
                }
            }
        }
        None
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let rank = self.rng.gen::<Rank>();

        let mut cur = self.root.borrow_mut();

        while let Some(node) = cur {
            match key.cmp(&node.key) {
                Ordering::Equal => {
                    let old_val = std::mem::replace(&mut node.value, value);
                    return Some(old_val);
                }
                Ordering::Less => {
                    if rank < node.rank {
                        cur = &mut node.left;
                    } else {
                        let right_tree = {
                            let x = std::mem::replace(node, Node::new(key, value, rank));
                            &mut node.right.insert(x).left
                        };
                        let old_value =
                            Self::unzip(Side::Right, &node.key, &mut node.left, right_tree);
                        if old_value.is_none() {
                            self.length += 1;
                        }
                        return old_value;
                    }
                }
                Ordering::Greater => {
                    if rank <= node.rank {
                        cur = &mut node.right;
                    } else {
                        let left_tree = {
                            let x = std::mem::replace(node, Node::new(key, value, rank));
                            &mut node.left.insert(x).right
                        };
                        let old_value =
                            Self::unzip(Side::Left, &node.key, left_tree, &mut node.right);
                        if old_value.is_none() {
                            self.length += 1;
                        }
                        return old_value;
                    }
                }
            }
        }

        let _ = cur.insert(Node::new(key, value, rank));
        self.length += 1;

        None
    }

    fn zip(mut side: Side, mut cur: &mut Tree<K, V>, mut node: Box<Node<K, V>>) {
        while let Some(x) = cur.take() {
            let (left_node, right_node) = match side {
                Side::Left => (x, node),
                Side::Right => (node, x),
            };

            if left_node.rank >= right_node.rank {
                cur = &mut cur.insert(left_node).right;
                node = right_node;
                side = Side::Left;
            } else {
                cur = &mut cur.insert(right_node).left;
                node = left_node;
                side = Side::Right;
            }
        }
        let _ = cur.insert(node);
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut cur = self.root.borrow_mut();
        while let Some(node) = cur.take() {
            match key.cmp(node.key.borrow()) {
                Ordering::Less => cur = &mut cur.insert(node).left,
                Ordering::Greater => cur = &mut cur.insert(node).right,
                Ordering::Equal => {
                    let _ = std::mem::replace(cur, node.right);

                    if let Some(left_node) = node.left {
                        Self::zip(Side::Right, cur, left_node);
                    }

                    self.length -= 1;
                    return Some(node.value);
                }
            }
        }

        None
    }
}

impl<K, V> Default for ZipTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<'a, K: 'a, V: 'a> {
    stack: Vec<&'a Node<K, V>>,
    length: usize,
}

impl<K, V> ZipTree<K, V> {
    pub fn iter(&self) -> Iter<'_, K, V> {
        let mut stack = Vec::new();
        let mut tree = &self.root;
        while let Some(x) = tree {
            stack.push(&**x);
            tree = &x.left;
        }
        Iter {
            stack,
            length: self.length,
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|node| {
            let mut tree = &node.right;
            while let Some(x) = tree {
                self.stack.push(x);
                tree = &x.left;
            }
            (&node.key, &node.value)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

pub struct IterMut<'a, K: 'a, V: 'a> {
    stack: Vec<(&'a K, &'a mut V, &'a mut Tree<K, V>)>,
    length: usize,
}

impl<K, V> ZipTree<K, V> {
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let mut stack = Vec::new();
        let mut tree = &mut self.root;
        while let Some(x) = tree {
            stack.push((&x.key, &mut x.value, &mut x.right));
            tree = &mut x.left;
        }
        IterMut {
            stack,
            length: self.length,
        }
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|(key, value, right)| {
            let mut tree = right;
            while let Some(x) = tree {
                self.stack.push((&x.key, &mut x.value, &mut x.right));
                tree = &mut x.left;
            }
            (key, value)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

pub struct IntoIter<K, V> {
    stack: Vec<(K, V, Tree<K, V>)>,
    length: usize,
}

impl<K, V> IntoIterator for ZipTree<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        let mut stack = Vec::new();
        let mut tree = self.root;

        while let Some(node) = tree {
            tree = node.left;
            stack.push((node.key, node.value, node.right));
        }
        IntoIter {
            stack,
            length: self.length,
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|(key, value, right)| {
            let mut tree = right;
            while let Some(x) = tree {
                self.stack.push((x.key, x.value, x.right));
                tree = x.left;
            }
            (key, value)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

impl<K: Clone, V: Clone> Clone for ZipTree<K, V> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            length: self.length,
            rng: SmallRng::from_entropy(),
        }
    }
}
