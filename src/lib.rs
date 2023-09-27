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
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::{Bound, RangeBounds};

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
    fn new(key: K, value: V, rank: Rank, left: Tree<K, V>, right: Tree<K, V>) -> Box<Self> {
        Box::new(Node {
            key,
            value,
            rank,
            left,
            right,
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
        let rank = self.rng.gen::<Rank>();

        let mut cur = self.root.borrow_mut();

        while let Some(node) = cur {
            match key.cmp(&node.key) {
                Ordering::Equal => {
                    return Some(std::mem::replace(&mut node.value, value));
                }
                Ordering::Less => {
                    if rank < node.rank {
                        cur = &mut node.left;
                    } else {
                        let mut left = None;
                        let old_value = Self::unzip(Side::Right, &key, &mut left, &mut node.left);
                        let right_node =
                            std::mem::replace(node, Node::new(key, value, rank, left, None));
                        let _ = node.right.insert(right_node);
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
                        let left_node =
                            std::mem::replace(node, Node::new(key, value, rank, None, right));
                        let _ = node.left.insert(left_node);
                        if old_value.is_none() {
                            self.length += 1;
                        }
                        return old_value;
                    }
                }
            }
        }

        let _ = cur.insert(Node::new(key, value, rank, None, None));
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
}

impl<K, V> Default for ZipTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct Iter<'a, K: 'a, V: 'a> {
    stack: Vec<&'a Node<K, V>>,
    length: usize,
}

impl<K, V> ZipTree<K, V> {
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

pub struct IterMut<'a, K: 'a, V: 'a> {
    stack: Vec<(&'a K, &'a mut V, &'a mut Tree<K, V>)>,
    length: usize,
}

impl<K, V> ZipTree<K, V> {
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

#[derive(Clone)]
pub struct IntoIter<K, V> {
    stack: Vec<(K, V, Tree<K, V>)>,
    length: usize,
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

impl<K: Clone, V: Clone> Clone for ZipTree<K, V> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            length: self.length,
            rng: SmallRng::from_entropy(),
        }
    }
}

fn satisfies_start<T: Ord>(start_bound: Bound<T>, item: T) -> bool {
    match start_bound {
        Bound::Included(start) => start <= item,
        Bound::Excluded(start) => start < item,
        Bound::Unbounded => true,
    }
}

fn satisfies_end<T: Ord>(end_bound: Bound<T>, item: T) -> bool {
    match end_bound {
        Bound::Included(end) => item <= end,
        Bound::Excluded(end) => item < end,
        Bound::Unbounded => true,
    }
}

enum Marker<'a, K: 'a, V: 'a> {
    Item(&'a Node<K, V>),
    Tree(&'a Node<K, V>),
}
pub struct Range<'a, K: 'a, V: 'a>(VecDeque<Marker<'a, K, V>>);

impl<K, V> ZipTree<K, V> {
    pub fn range<T: ?Sized, R>(&self, range: R) -> Range<'_, K, V>
    where
        T: Ord,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let mut deque = VecDeque::new();
        let mut cur = &self.root;

        while let Some(root_node) = cur {
            if !satisfies_start(range.start_bound(), root_node.key.borrow()) {
                cur = &root_node.right;
            } else if !satisfies_end(range.end_bound(), root_node.key.borrow()) {
                cur = &root_node.left;
            } else {
                let mut left_tree = &root_node.left;
                while let Some(node) = left_tree {
                    if satisfies_start(range.start_bound(), node.key.borrow()) {
                        if let Some(ref right_node) = node.right {
                            deque.push_front(Marker::Tree(right_node));
                        }
                        deque.push_front(Marker::Item(node));
                        left_tree = &node.left;
                    } else {
                        left_tree = &node.right;
                    }
                }

                deque.push_back(Marker::Item(root_node));

                let mut right_tree = &root_node.right;
                while let Some(node) = right_tree {
                    if satisfies_end(range.end_bound(), node.key.borrow()) {
                        if let Some(ref left_node) = node.left {
                            deque.push_back(Marker::Tree(left_node));
                        }
                        deque.push_back(Marker::Item(node));
                        right_tree = &node.right;
                    } else {
                        right_tree = &node.left;
                    }
                }

                break;
            }
        }
        Range(deque)
    }
}

impl<'a, K, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_front().map(|m| match m {
            Marker::Item(node) => (&node.key, &node.value),
            Marker::Tree(mut node) => loop {
                if let Some(ref right_node) = node.right {
                    self.0.push_front(Marker::Tree(right_node));
                }
                if let Some(ref left_node) = node.left {
                    self.0.push_front(Marker::Item(node));
                    node = left_node;
                } else {
                    return (&node.key, &node.value);
                }
            },
        })
    }
}

impl<'a, K, V> DoubleEndedIterator for Range<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.pop_back().map(|m| match m {
            Marker::Item(node) => (&node.key, &node.value),
            Marker::Tree(mut node) => loop {
                if let Some(ref left_node) = node.left {
                    self.0.push_back(Marker::Tree(left_node));
                }
                if let Some(ref right_node) = node.right {
                    self.0.push_back(Marker::Item(node));
                    node = right_node;
                } else {
                    return (&node.key, &node.value);
                }
            },
        })
    }
}

enum MarkerMut<'a, K: 'a, V: 'a> {
    Item((&'a K, &'a mut V)),
    Tree(&'a mut Node<K, V>),
}
pub struct RangeMut<'a, K: 'a, V: 'a>(VecDeque<MarkerMut<'a, K, V>>);

impl<K, V> ZipTree<K, V> {
    pub fn range_mut<T: ?Sized, R>(&mut self, range: R) -> RangeMut<'_, K, V>
    where
        T: Ord,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let mut deque = VecDeque::new();
        let mut cur = &mut self.root;

        while let Some(root_node) = cur {
            if !satisfies_start(range.start_bound(), root_node.key.borrow()) {
                cur = &mut root_node.right;
            } else if !satisfies_end(range.end_bound(), root_node.key.borrow()) {
                cur = &mut root_node.left;
            } else {
                let mut left_tree = &mut root_node.left;
                while let Some(node) = left_tree {
                    if satisfies_start(range.start_bound(), node.key.borrow()) {
                        if let Some(ref mut right_node) = node.right {
                            deque.push_front(MarkerMut::Tree(right_node));
                        }
                        deque.push_front(MarkerMut::Item((&node.key, &mut node.value)));
                        left_tree = &mut node.left;
                    } else {
                        left_tree = &mut node.right;
                    }
                }

                deque.push_back(MarkerMut::Item((&root_node.key, &mut root_node.value)));

                let mut right_tree = &mut root_node.right;
                while let Some(node) = right_tree {
                    if satisfies_end(range.end_bound(), node.key.borrow()) {
                        if let Some(ref mut left_node) = node.left {
                            deque.push_back(MarkerMut::Tree(left_node));
                        }
                        deque.push_back(MarkerMut::Item((&node.key, &mut node.value)));
                        right_tree = &mut node.right;
                    } else {
                        right_tree = &mut node.left;
                    }
                }

                break;
            }
        }
        RangeMut(deque)
    }
}

impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_front().map(|m| match m {
            MarkerMut::Item(key_value) => key_value,
            MarkerMut::Tree(mut node) => loop {
                if let Some(ref mut right_node) = node.right {
                    self.0.push_front(MarkerMut::Tree(right_node));
                }
                if let Some(ref mut left_node) = node.left {
                    self.0
                        .push_front(MarkerMut::Item((&node.key, &mut node.value)));
                    node = left_node;
                } else {
                    return (&node.key, &mut node.value);
                }
            },
        })
    }
}

impl<'a, K, V> DoubleEndedIterator for RangeMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.pop_back().map(|m| match m {
            MarkerMut::Item(key_value) => key_value,
            MarkerMut::Tree(mut node) => loop {
                if let Some(ref mut left_node) = node.left {
                    self.0.push_back(MarkerMut::Tree(left_node));
                }
                if let Some(ref mut right_node) = node.right {
                    self.0
                        .push_back(MarkerMut::Item((&node.key, &mut node.value)));
                    node = right_node;
                } else {
                    return (&node.key, &mut node.value);
                }
            },
        })
    }
}
