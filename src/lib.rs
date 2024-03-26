//! Tarjan's zip tree implementation in Rust.
//!
//! Zip tree is a treap with different insertion and deletion algorithms.
//! It organizes node ranks like skip list, but takes less space than skip list.
//! Insertion and deletion are done by _zip_ and _unzip_ operations instead of
//! a series of tree rotations. You can see [Tarjans's paper](https://arxiv.org/abs/1806.06726)
//! for more details.

extern crate rand;

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::borrow::Borrow;
use std::mem::replace;
use std::cmp::Ordering;
use std::collections::{LinkedList, HashMap};
use std::fmt::Debug;
use std::ptr::null_mut;
use rand::prelude::*;

/// The struct is created by [ZipTree::into_iter()](struct.ZipTree.html#impl-IntoIterator).
#[derive(Clone)]
pub struct IntoIter<K, V> {
    iter: std::collections::linked_list::IntoIter<(K, V)>,
}

/// The struct is created by [ZipTree::iter()](struct.ZipTree.html#method.iter).
#[derive(Clone)]
pub struct Iter<'a, K: 'a, V: 'a> {
    tree: &'a ZipTree<K, V>,
    path: LinkedList<(*mut Node<K, V>, DfsAction)>,
    dummy: PhantomData<(&'a K, &'a V)>,
}

/// The struct is created by [ZipTree::iter_mut()](struct.ZipTree.html#method.iter_mut).
pub struct IterMut<'a, K: 'a, V: 'a> {
    tree: &'a mut ZipTree<K, V>,
    path: LinkedList<(*mut Node<K, V>, DfsAction)>,
    dummy: PhantomData<(&'a K, &'a mut V)>,
}

/// The struct is created by [ZipTree::keys()](struct.ZipTree.html#method.keys).
#[derive(Clone)]
pub struct Keys<'a, K: 'a, V: 'a> {
    tree: &'a ZipTree<K, V>,
    path: LinkedList<(*mut Node<K, V>, DfsAction)>,
    dummy: PhantomData<(&'a K, &'a V)>,
}

/// The struct is created by [ZipTree::values()](struct.ZipTree.html#method.values).
#[derive(Clone)]
pub struct Values<'a, K: 'a, V: 'a> {
    tree: &'a ZipTree<K, V>,
    path: LinkedList<(*mut Node<K, V>, DfsAction)>,
    dummy: PhantomData<(&'a K, &'a V)>,
}

/// The struct is created by [ZipTree::values_mut()](struct.ZipTree.html#method.values_mut).
pub struct ValuesMut<'a, K: 'a, V: 'a> {
    tree: &'a mut ZipTree<K, V>,
    path: LinkedList<(*mut Node<K, V>, DfsAction)>,
    dummy: PhantomData<(&'a K, &'a V)>,
}


/// Tarjan's zip tree implementation.
///
/// The ZipTree API mimics the standard library's BTreeMap. It provides look-ups, insertions,
/// deletions and iterator interface. Cloning this tree will deep copy overall tree structure,
/// and thus takes O(n) time.
pub struct ZipTree<K, V> {
    root: *mut Node<K, V>,
    n_nodes: usize,
}

struct Node<K, V> {
    key: K,
    value: V,
    rank: usize,
    left: *mut Node<K, V>,
    right: *mut Node<K, V>,
}

#[derive(Clone)]
enum DfsAction {
    Enter, Leave
}

impl<K, V> Node<K, V> where {
    fn new(key: K, value: V, rank: usize) -> Node<K, V> {
        Node {
            key,
            value,
            rank,
            left: null_mut(),
            right: null_mut(),
        }
    }
}

impl<K, V> ZipTree<K, V> where
    K: Ord,
{
    pub fn new() -> ZipTree<K, V> {
        ZipTree {
            root: null_mut(),
            n_nodes: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.n_nodes
    }

    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V> where
        K: Borrow<Q>,
        Q: Ord
    {
        let mut node = self.root;

        unsafe {
            while !node.is_null() {
                match key.cmp((*node).key.borrow()) {
                    Ordering::Equal => {
                        return Some(&(*node).value);
                    }
                    Ordering::Less =>
                        node = (*node).left,
                    Ordering::Greater =>
                        node = (*node).right,
                }
            }
        }
        None
    }

    pub fn get_mut<'a, Q: ?Sized>(&'a mut self, key: &Q) -> Option<&'a mut V> where
        K: Borrow<Q>,
        Q: Ord
    {
        let mut node = self.root;

        unsafe {
            while !node.is_null() {
                match key.cmp((*node).key.borrow()) {
                    Ordering::Equal =>
                        return Some(&mut (*node).value),
                    Ordering::Less =>
                        node = (*node).left,
                    Ordering::Greater =>
                        node = (*node).right,
                }
            }
        }

        None
    }

    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool where
        K: Borrow<Q>,
        Q: Ord,
    {
        self.get(key.borrow()).is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.n_nodes == 0
    }

    pub fn clear(&mut self) {
        let mut stack = LinkedList::new();

        if self.root.is_null() {
            return;
        }
        else {
            stack.push_back(self.root);
        }

        while !stack.is_empty() {
            let node = stack.pop_back().unwrap();
            if node.is_null() {
                continue;
            }

            unsafe {
                stack.push_back((*node).left);
                stack.push_back((*node).right);
                drop(Box::from_raw(node));
            }
        }

        self.n_nodes = 0;
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut rng = rand::thread_rng();

        let rank = {
            let mut rank = 0;
            while rng.gen_range(0..2) == 1 {
                rank += 1
            }
            rank
        };


        let (parent, child, key_cmp, less_nodes, greater_nodes) = unsafe {
            let mut prev = null_mut();
            let mut curr = self.root;
            let mut prev_cmp = None;

            // Locate insertion point
            let (parent, child, key_cmp) = loop {
                if curr.is_null() {
                    break (prev, curr, prev_cmp);
                }

                let key_cmp = key.cmp((*curr).key.borrow());
                let rank_cmp = rank.cmp(&(*curr).rank);

                match (rank_cmp, key_cmp) {
                    (Ordering::Less, _) =>
                        break (prev, curr, prev_cmp),
                    (Ordering::Equal, Ordering::Less) =>
                        break (prev, curr, prev_cmp),
                    _ => {}
                }

                match key_cmp {
                    Ordering::Equal => {
                        let prev_value = replace(&mut (*curr).value, value);
                        return Some(prev_value);
                    }
                    Ordering::Less => {
                        prev = curr;
                        curr = (*curr).left;
                    }
                    Ordering::Greater => {
                        prev = curr;
                        curr = (*curr).right;
                    }
                }
                prev_cmp = Some(key_cmp);
            };

            let mut less_nodes = Vec::new();
            let mut greater_nodes = Vec::new();

            // Find duplicate key in remaining subtree
            while !curr.is_null() {
                let curr_cmp = key.cmp((*curr).key.borrow());
                match curr_cmp {
                    Ordering::Equal => {
                        let prev_value = replace(&mut (*curr).value, value);
                        return Some(prev_value);
                    }
                    Ordering::Less => {
                        greater_nodes.push(curr);
                        curr = (*curr).left;
                    }
                    Ordering::Greater => {
                        less_nodes.push(curr);
                        curr = (*curr).right;
                    }
                }
            }

            (parent, child, key_cmp, less_nodes, greater_nodes)
        };

        let new_node = Box::into_raw(Box::new(Node::new(key, value, rank)));
        self.n_nodes += 1;

        if !parent.is_null() {
            unsafe {
                match key_cmp {
                    Some(Ordering::Less) =>
                        (*parent).left = new_node,
                    Some(Ordering::Greater) =>
                        (*parent).right = new_node,
                    None => {},
                    _ => panic!("bug"),
                }
            }
        }
        else if child.is_null() {
            assert!(self.root.is_null());
            self.root = new_node;
            return None;
        }
        else {
            assert!(child == self.root);
            self.root = new_node;
        }


        if !less_nodes.is_empty() {
            let n_nodes = less_nodes.len();
            let zip_left = less_nodes[0..(n_nodes - 1)].iter();
            let zip_right = less_nodes[1..n_nodes].iter();

            unsafe {
                for (prev, curr) in zip_left.zip(zip_right) {
                    (**prev).right = *curr;
                }

                let last_node = less_nodes.last().unwrap();
                (**last_node).right = null_mut();

                (*new_node).left = *less_nodes.first().unwrap();
            }

        }

        if !greater_nodes.is_empty() {
            let n_nodes = greater_nodes.len();
            let zip_left = greater_nodes[0..(n_nodes - 1)].iter();
            let zip_right = greater_nodes[1..n_nodes].iter();

            unsafe {
                for (prev, curr) in zip_left.zip(zip_right) {
                    (**prev).left = *curr;
                }

                let last_node = greater_nodes.last().unwrap();
                (**last_node).left = null_mut();

                (*new_node).right = *greater_nodes.first().unwrap();
            }
        }

        None
    }

    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V> where
        K: Borrow<Q>,
        Q: Ord,
    {
        let (parent, selected, key_cmp_opt, mut less_nodes, mut greater_nodes) = unsafe {
            let mut prev = null_mut();
            let mut curr = self.root;
            let mut prev_cmp = None;

            let (parent, selected, key_cmp) = loop {
                if curr.is_null() {
                    return None;
                }

                let curr_cmp = key.cmp((*curr).key.borrow());
                match curr_cmp {
                    Ordering::Less => {
                        prev = curr;
                        curr = (*curr).left;
                    }
                    Ordering::Greater => {
                        prev = curr;
                        curr = (*curr).right
                    }
                    Ordering::Equal => break (prev, curr, prev_cmp),
                }
                prev_cmp = Some(curr_cmp);
            };

            let mut less_nodes = LinkedList::new();
            let mut greater_nodes = LinkedList::new();

            if !(*selected).left.is_null() {
                less_nodes.push_back((*selected).left);

                loop {
                    let node = less_nodes.back().unwrap();
                    if !(**node).right.is_null() {
                        less_nodes.push_back((**node).right);
                    }
                    else {
                        break;
                    }
                }
            }

            if !(*selected).right.is_null() {
                greater_nodes.push_back((*selected).right);

                loop {
                    let node = greater_nodes.back().unwrap();
                    if !(**node).left.is_null() {
                        greater_nodes.push_back((**node).left);
                    }
                    else {
                        break;
                    }
                }
            }

            (parent, selected, key_cmp, less_nodes, greater_nodes)
        };

        let mut child_nodes = LinkedList::<*mut Node<K, V>>::new();
        let mut rng = rand::thread_rng();

        unsafe {
            loop {
                let node = match less_nodes.front() {
                    Some(less) => {
                        match greater_nodes.front() {
                            Some(greater) => {
                                match (**less).rank.cmp(&(**greater).rank) {
                                    Ordering::Less => greater_nodes.pop_front().unwrap(),
                                    Ordering::Greater => less_nodes.pop_front().unwrap(),
                                    Ordering::Equal => match rng.gen_bool(0.5) {
                                        true => greater_nodes.pop_front().unwrap(),
                                        false => less_nodes.pop_front().unwrap(),
                                    }
                                }
                            }
                            None => {
                                less_nodes.pop_front().unwrap()
                            }
                        }
                    }
                    None => {
                        match greater_nodes.front() {
                            Some(greater) => {
                                greater_nodes.pop_front().unwrap()
                            }
                            None => {
                                break;
                            }
                        }
                    }
                };

                if let Some(prev) = child_nodes.back() {
                    match (**prev).key.cmp(&(*node).key) {
                        Ordering::Equal => panic!("bug"),
                        Ordering::Less => (**prev).right = node,
                        Ordering::Greater => (**prev).left = node,
                    }
                }

                child_nodes.push_back(node);
            }
        }

        match key_cmp_opt {
            None => {
                assert!(selected == self.root && parent.is_null());
                match child_nodes.front() {
                    Some(child) => self.root = *child,
                    None => self.root = null_mut(),
                }
            }
            Some(Ordering::Less) => {
                unsafe {
                    (*parent).left = match child_nodes.front() {
                        Some(child) => *child,
                        None => null_mut(),
                    };
                }
            }
            Some(Ordering::Greater) => {
                unsafe {
                    (*parent).right = match child_nodes.front() {
                        Some(child) => *child,
                        None => null_mut(),
                    };
                }
            }
            _ => panic!("bug"),
        }

        self.n_nodes -= 1;
        unsafe {
            let selected_boxed = Box::from_raw(selected);
            let prev_value = selected_boxed.value;
            Some(prev_value)
        }
    }

    pub fn iter<'a>(&'a self) -> Iter<'a, K, V> {
        let mut path = LinkedList::new();
        path.push_back((self.root, DfsAction::Enter));

        Iter {
            tree: self,
            path,
            dummy: PhantomData,
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, K, V> {
        let mut path = LinkedList::new();
        path.push_back((self.root, DfsAction::Enter));

        IterMut {
            tree: self,
            path,
            dummy: PhantomData,
        }
    }

    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        let mut path = LinkedList::new();
        path.push_back((self.root, DfsAction::Enter));

        Keys {
            tree: self,
            path,
            dummy: PhantomData,
        }
    }

    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        let mut path = LinkedList::new();
        path.push_back((self.root, DfsAction::Enter));

        Values {
            tree: self,
            path,
            dummy: PhantomData,
        }
    }

    pub fn values_mut<'a>(&'a mut self) -> ValuesMut<'a, K, V> {
        let mut path = LinkedList::new();
        path.push_back((self.root, DfsAction::Enter));

        ValuesMut {
            tree: self,
            path,
            dummy: PhantomData,
        }
    }

    fn dump(&self) where
        K: Debug,
        V: Debug,
    {
        // This method is intended for debug purpose
        let mut stack = LinkedList::new();

        if self.root.is_null() {
            return;
        }
        else {
            stack.push_back((self.root, 0));
        }

        let mut rank_cnt = HashMap::new();
        let mut prev_key_opt = None;
        println!("ind\tkey\tval\trank\tleft\tright");

        loop {
            let (node, st) = match stack.pop_back() {
                None => break,
                Some(state) => state,
            };
            assert!(!node.is_null());

            unsafe {
                match st {
                    0 => {
                        if !(*node).left.is_null() {
                            stack.push_back(((*node).left, 0));
                        }

                        stack.push_back((node, 1));

                        if !(*node).right.is_null() {
                            stack.push_back(((*node).right, 0));
                        }
                    }
                    1 => {
                        // println!(
                        //     "{}\t{:?}\t{:?}\t{}\t{:?}\t{:?}\t{:?}",
                        //     stack.len(),
                        //     (*node).key,
                        //     (*node).value,
                        //     (*node).rank,
                        //     node,
                        //     (*node).left,
                        //     (*node).right,
                        // );

                        let curr_key = &(*node).key;
                        if let Some(prev) = prev_key_opt {
                            assert!(prev > curr_key);
                        }

                        rank_cnt.entry((*node).rank)
                            .and_modify(|cnt| { *cnt += 1; })
                            .or_insert(1);

                        prev_key_opt = Some(curr_key);
                    }
                    _ => panic!("bug"),
                }
            }
        }

        let mut ranks = rank_cnt.into_iter().collect::<Vec<_>>();
        ranks.sort_unstable_by_key(|(rank, _)| *rank);
        ranks.into_iter()
            .for_each(|(rank, cnt)| {
                println!("{}\t{}", rank, cnt);
            });
    }
}

impl<K, V, Q: ?Sized> Index<&Q> for ZipTree<K, V> where
    K: Borrow<Q> + Ord,
    Q: Ord,
{
    type Output = V;

    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V, Q: ?Sized> IndexMut<&Q> for ZipTree<K, V> where
    K: Borrow<Q> + Ord,
    Q: Ord,
{
    fn index_mut<'a>(&'a mut self, key: &Q) -> &'a mut Self::Output {
        self.get_mut(key).expect("no entry found for key")
    }
}

impl<K: Clone, V: Clone> Clone for ZipTree<K, V> {
    fn clone(&self) -> Self {
        let mut from_stack = LinkedList::new();
        let mut to_stack = LinkedList::new();

        if self.root.is_null() {
            return ZipTree {
                root: null_mut(),
                n_nodes: 0,
            };
        }

        let new_root = unsafe {
            let new_root = Box::into_raw(Box::new(Node::new(
                (*self.root).key.clone(),
                (*self.root).value.clone(),
                (*self.root).rank.clone(),
            )));

            from_stack.push_back(self.root);
            to_stack.push_back(new_root);

            while !from_stack.is_empty() {
                let from_node = from_stack.pop_back().unwrap();
                let to_node = to_stack.pop_back().unwrap();

                let left = (*from_node).left;
                let right = (*from_node).right;

                if !left.is_null() {
                    let new_node = Box::into_raw(Box::new(Node::new(
                        (*left).key.clone(),
                        (*left).value.clone(),
                        (*left).rank.clone(),
                    )));
                    (*to_node).left = new_node;

                    from_stack.push_back(left);
                    to_stack.push_back(new_node);
                }

                if !right.is_null() {
                    let new_node = Box::into_raw(Box::new(Node::new(
                        (*right).key.clone(),
                        (*right).value.clone(),
                        (*right).rank.clone(),
                    )));
                    (*to_node).right = new_node;

                    from_stack.push_back(right);
                    to_stack.push_back(new_node);
                }
            }

            new_root
        };

        ZipTree {
            root: new_root,
            n_nodes: self.n_nodes,
        }
    }
}

impl<K, V> Drop for ZipTree<K, V> {
    fn drop(&mut self) {
        let mut path = LinkedList::new();
        path.push_back((self.root, DfsAction::Enter));

        loop {
            let (node, action) = match path.pop_back() {
                None => break,
                Some(item) => item,
            };

            if node.is_null() {
                continue;
            }

            let (left, right) = unsafe {
                let left = (*node).left;
                let right = (*node).right;
                (left, right)
            };

            match action {
                DfsAction::Enter => {
                    path.push_back((right, DfsAction::Enter));
                    path.push_back((node, DfsAction::Leave));
                    path.push_back((left, DfsAction::Enter));
                }
                DfsAction::Leave => {
                    let node_boxed = unsafe {
                        Box::from_raw(node)
                    };
                    drop(node_boxed);
                }
            }
        }

        self.root = null_mut();
        self.n_nodes = 0;
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, action) = match self.path.pop_back() {
                None => return None,
                Some(item) => item,
            };

            if node.is_null() {
                continue;
            }

            let (left, right) = unsafe {
                let left = (*node).left;
                let right = (*node).right;
                (left, right)
            };

            match action {
                DfsAction::Enter => {
                    self.path.push_back((right, DfsAction::Enter));
                    self.path.push_back((node, DfsAction::Leave));
                    self.path.push_back((left, DfsAction::Enter));
                }
                DfsAction::Leave => {
                    let (key, value) = unsafe {
                        let key = &(*node).key;
                        let value = &(*node).value;
                        (key, value)
                    };

                    return Some((key, value));
                }
            }
        }
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, action) = match self.path.pop_back() {
                None => return None,
                Some(item) => item,
            };

            if node.is_null() {
                continue;
            }

            let (left, right) = unsafe {
                let left = (*node).left;
                let right = (*node).right;
                (left, right)
            };

            match action {
                DfsAction::Enter => {
                    self.path.push_back((right, DfsAction::Enter));
                    self.path.push_back((node, DfsAction::Leave));
                    self.path.push_back((left, DfsAction::Enter));
                }
                DfsAction::Leave => {
                    let (key, value) = unsafe {
                        let key = &(*node).key;
                        let value = &mut (*node).value;
                        (key, value)
                    };

                    return Some((key, value));
                }
            }
        }
    }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, action) = match self.path.pop_back() {
                None => return None,
                Some(item) => item,
            };

            if node.is_null() {
                continue;
            }

            let (left, right) = unsafe {
                let left = (*node).left;
                let right = (*node).right;
                (left, right)
            };

            match action {
                DfsAction::Enter => {
                    self.path.push_back((right, DfsAction::Enter));
                    self.path.push_back((node, DfsAction::Leave));
                    self.path.push_back((left, DfsAction::Enter));
                }
                DfsAction::Leave => {
                    let key = unsafe {
                        &(*node).key
                    };

                    return Some(key);
                }
            }
        }
    }
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, action) = match self.path.pop_back() {
                None => return None,
                Some(item) => item,
            };

            if node.is_null() {
                continue;
            }

            let (left, right) = unsafe {
                let left = (*node).left;
                let right = (*node).right;
                (left, right)
            };

            match action {
                DfsAction::Enter => {
                    self.path.push_back((right, DfsAction::Enter));
                    self.path.push_back((node, DfsAction::Leave));
                    self.path.push_back((left, DfsAction::Enter));
                }
                DfsAction::Leave => {
                    let value = unsafe {
                        &(*node).value
                    };

                    return Some(value);
                }
            }
        }
    }
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (node, action) = match self.path.pop_back() {
                None => return None,
                Some(item) => item,
            };

            if node.is_null() {
                continue;
            }

            let (left, right) = unsafe {
                let left = (*node).left;
                let right = (*node).right;
                (left, right)
            };

            match action {
                DfsAction::Enter => {
                    self.path.push_back((right, DfsAction::Enter));
                    self.path.push_back((node, DfsAction::Leave));
                    self.path.push_back((left, DfsAction::Enter));
                }
                DfsAction::Leave => {
                    let value = unsafe {
                        &mut (*node).value
                    };

                    return Some(value);
                }
            }
        }
    }
}


impl<K, V> IntoIterator for ZipTree<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(mut self) -> Self::IntoIter {
        let mut entries = LinkedList::new();
        let mut path = LinkedList::new();
        path.push_back((self.root, DfsAction::Enter));

        loop {
            let (node, action) = match path.pop_back() {
                None => break,
                Some(item) => item,
            };

            if node.is_null() {
                continue;
            }

            let (left, right) = unsafe {
                let left = (*node).left;
                let right = (*node).right;
                (left, right)
            };

            match action {
                DfsAction::Enter => {
                    path.push_back((right, DfsAction::Enter));
                    path.push_back((node, DfsAction::Leave));
                    path.push_back((left, DfsAction::Enter));
                }
                DfsAction::Leave => {
                    let node_boxed = unsafe {
                        Box::from_raw(node)
                    };
                    entries.push_back((node_boxed.key, node_boxed.value));
                }
            }
        }

        self.root = null_mut();
        self.n_nodes = 0;
        drop(self);

        IntoIter {
            iter: entries.into_iter(),
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
