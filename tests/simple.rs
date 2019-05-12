use std::fmt::Debug;
use std::collections::{HashMap, BTreeMap};
use rand::Rng;

#[test]
fn simple_test() -> Result<(), Box<dyn Debug>> {
    let mut rng = rand::thread_rng();
    let mut tree = ziptree::ZipTree::<usize, usize>::new();
    let mut pairs = BTreeMap::new();

    for _ in 0..(1<<13) {

        if rng.gen_bool(0.5) {
            let key = rng.gen();
            let val = rng.gen();
            tree.insert(key, val);
            pairs.insert(key, val);
            // println!("insert {} {}", key, val);
        }
        else {
            let key = rng.gen();
            let expect_val = pairs.remove(&key);
            let val = tree.remove(&key);

            // println!("remove {}, {:?} ==? {:?}", key, expect_val, val);
            assert!(val == expect_val);
        }

        // tree.dump();
        for ((key, val), (expect_key, expect_val)) in tree.iter().zip(pairs.iter()) {
            assert!(key == expect_key && val == expect_val);
        }
        assert!(tree.len() == pairs.len());
    }

    for ((key, val), (expect_key, expect_val)) in tree.into_iter().zip(pairs.into_iter()) {
        assert!(key == expect_key && val == expect_val);
    }
    // tree.dump();

    Ok(())
}
