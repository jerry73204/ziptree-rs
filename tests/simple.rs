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
        }
    }
    assert!(tree1.into_iter().eq(tree2.into_iter()));
}
