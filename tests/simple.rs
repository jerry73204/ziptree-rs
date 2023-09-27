use rand::Rng;
use std::collections::BTreeMap;
use std::fmt::Debug;

#[test]
fn simple_test() -> Result<(), Box<dyn Debug>> {
    let mut rng = rand::thread_rng();
    let mut tree = ziptree::ZipTree::new();
    let mut pairs = BTreeMap::new();

    for _ in 0..1_000_000 {
        if rng.gen_bool(0.5) {
            let key = rng.gen::<u16>();
            let val = rng.gen::<usize>();
            let expect_old_val: Option<usize> = pairs.insert(key, val);
            let old_val = tree.insert(key, val);

            // println!("insert {:?} {:?}, {:?} ==? {:?}", key, val, expect_old_val, old_val);
            assert!(old_val == expect_old_val);
        } else {
            let key = rng.gen();
            let expect_val = pairs.remove(&key);
            let val = tree.remove(&key);

            // println!("remove {:?}, {:?} ==? {:?}", key, expect_val, val);
            assert!(val == expect_val);
        }

        // for ((key, val), (expect_key, expect_val)) in tree.iter().zip(pairs.iter()) {
        //     assert!(key == expect_key && val == expect_val);
        // }
        assert!(tree.len() == pairs.len());
    }


    for _ in 0..1000 {
        let mut key1 = rng.gen::<u16>();
        let mut key2 = rng.gen::<u16>();
        // We have to do this or BTreeMap will panic
        if key1 > key2 {
            std::mem::swap(&mut key1, &mut key2);
        }

        for ((key, val), (expect_key, expect_val)) in tree.range(key1..key2).zip(pairs.range(key1..key2)) {
            assert!(key == expect_key && val == expect_val);
        }
    }

    // println!("{:?}", tree);
    for ((key, val), (expect_key, expect_val)) in tree.into_iter().zip(pairs.into_iter()) {
        assert!(key == expect_key && val == expect_val);
    }

    Ok(())
}
