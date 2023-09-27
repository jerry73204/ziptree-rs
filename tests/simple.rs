use rand::Rng;
use std::collections::BTreeMap;

#[test]
fn simple_test() {
    let mut rng = rand::thread_rng();
    let mut tree1 = BTreeMap::new();
    let mut tree2 = ziptree::ZipTree::new();

    for i in 0..1_000_000 {
        if rng.gen_bool(0.5) {
            let (key, val) = rng.gen::<(u16, usize)>();
            assert!(tree1.insert(key, val) == tree2.insert(key, val));
        } else {
            let key = rng.gen();
            assert!(tree1.remove(&key) == tree2.remove(&key));
        }
        assert!(tree1.len() == tree2.len());

        if i % 1000 == 0 {
            assert!(tree1.iter().eq(tree2.iter()));

            let (mut key1, mut key2) = rng.gen::<(u16, u16)>();
            // We have to do this or BTreeMap will panic
            if key1 > key2 {
                std::mem::swap(&mut key1, &mut key2);
            }
            assert!(tree1.range(key1..key2).eq(tree2.range(key1..key2)));
            assert!(tree1
                .range(key1..key2)
                .rev()
                .eq(tree2.range(key1..key2).rev()));
        }
    }
    assert!(tree1.into_iter().eq(tree2.into_iter()));
}
