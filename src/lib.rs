//! Tarjan's zip tree implementation in Rust.
//!
//! Zip tree is an efficient representation of skip list which takes only
//! O(log log n) bit extra space compared to a bare binary search tree.
//! Insertions and deletions are performed by top-down unmerging and merging paths
//! ("unzipping" and "zipping") rather than rotations which reduce pointer
//! changes and are highly amenable to concurrent implementations. Read
//! [1](https://arxiv.org/abs/1806.06726) and [2](https://arxiv.org/abs/2307.07660)
//! for more details.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::mem::replace;

/// The struct is created by [ZipTree::into_iter()](struct.ZipTree.html#impl-IntoIterator).
#[derive(Clone)]
pub struct IntoIter<K, V> {
    stack: Vec<(K, V, Tree<K, V>)>,
    length: usize,
}

/// The struct is created by [ZipTree::iter()](struct.ZipTree.html#method.iter).
#[derive(Clone)]
pub struct Iter<'a, K: 'a, V: 'a> {
    stack: Vec<&'a Node<K, V>>,
    length: usize,
}

/// The struct is created by [ZipTree::iter_mut()](struct.ZipTree.html#method.iter_mut).
pub struct IterMut<'a, K: 'a, V: 'a> {
    stack: Vec<(&'a K, &'a mut V, &'a mut Tree<K, V>)>,
    length: usize,
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

#[derive(Debug, Clone)]
struct Node<K, V> {
    key: K,
    value: V,
    rank: u16,
    left: Tree<K, V>,
    right: Tree<K, V>,
}

type Tree<K, V> = Option<Box<Node<K, V>>>;

enum Side {
    Left,
    Right,
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
        let mut cur = &self.root;
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
        let mut cur = &mut self.root;
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
                    *left_tree = node.left;
                    *right_tree = node.right;
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
        let rank = {
            // First 6 bits is drawn from geometric distribution with p = 0.5.
            let r1 = self.rng.gen::<u64>().leading_zeros() as u16;
            // Last 10 bits is drawn from uniform distribution.
            let r2 = self.rng.gen::<u16>() >> 6;
            (r1 << 10) | r2
        };

        let mut cur = &mut self.root;

        while let Some(node) = cur {
            match key.cmp(&node.key) {
                Ordering::Equal => {
                    return Some(replace(&mut node.value, value));
                }
                Ordering::Less => {
                    if rank < node.rank {
                        cur = &mut node.left;
                    } else {
                        let mut left = None;
                        let old_value = Self::unzip(Side::Right, &key, &mut left, &mut node.left);
                        let right_node = replace(
                            node,
                            Box::new(Node {
                                key,
                                value,
                                rank,
                                left,
                                right: None,
                            }),
                        );
                        node.right = Some(right_node);
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
                        let mut right = None;
                        let old_value = Self::unzip(Side::Left, &key, &mut node.right, &mut right);
                        let left_node = replace(
                            node,
                            Box::new(Node {
                                key,
                                value,
                                rank,
                                left: None,
                                right,
                            }),
                        );
                        node.left = Some(left_node);
                        if old_value.is_none() {
                            self.length += 1;
                        }
                        return old_value;
                    }
                }
            }
        }

        *cur = Some(Box::new(Node {
            key,
            value,
            rank,
            left: None,
            right: None,
        }));
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
        *cur = Some(node);
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut cur = &mut self.root;
        while let Some(node) = cur.take() {
            match key.cmp(node.key.borrow()) {
                Ordering::Less => cur = &mut cur.insert(node).left,
                Ordering::Greater => cur = &mut cur.insert(node).right,
                Ordering::Equal => {
                    *cur = node.right;

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

    pub fn iter(&self) -> Iter<'_, K, V> {
        let mut stack = Vec::new();
        let mut cur = &self.root;
        while let Some(x) = cur {
            stack.push(&**x);
            cur = &x.left;
        }
        Iter {
            stack,
            length: self.length,
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let mut stack = Vec::new();
        let mut cur = &mut self.root;
        while let Some(x) = cur {
            stack.push((&x.key, &mut x.value, &mut x.right));
            cur = &mut x.left;
        }
        IterMut {
            stack,
            length: self.length,
        }
    }
}

impl<K, V> Default for ZipTree<K, V> {
    fn default() -> Self {
        Self::new()
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

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|node| {
            let mut cur = &node.right;
            while let Some(x) = cur {
                self.stack.push(x);
                cur = &x.left;
            }
            (&node.key, &node.value)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|(key, value, right)| {
            let mut cur = right;
            while let Some(x) = cur {
                self.stack.push((&x.key, &mut x.value, &mut x.right));
                cur = &mut x.left;
            }
            (key, value)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

impl<K, V> IntoIterator for ZipTree<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        let mut stack = Vec::new();
        let mut cur = self.root;

        while let Some(node) = cur {
            cur = node.left;
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
            let mut cur = right;
            while let Some(x) = cur {
                self.stack.push((x.key, x.value, x.right));
                cur = x.left;
            }
            (key, value)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}
