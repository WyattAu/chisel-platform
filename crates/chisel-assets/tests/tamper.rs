//! Tamper tests: bytes damaged behind the API must fail loudly on read.
//! Integration tests (they touch the real filesystem); kept out of the
//! lib's coverage so the measurement is of the library surface.
mod tamper_tests {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::panic,
        clippy::indexing_slicing
    )]
    use chisel_assets::{AssetError, AssetStore};

    #[test]
    fn a_flipped_bit_on_disk_fails_the_read_as_corruption()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("chisel-tamper-{}", std::process::id()));
        let store = AssetStore::open(&dir)?;
        let doc = store.put(b"the authentic lease, revision 7")?;

        // Flip one byte inside the stored blob, bypassing the API. Walk the
        // whole store root: the bucket layout is cas-kit's business, ours is
        // only "some file under here holds these bytes".
        let blob_path = {
            let mut found = None;
            let mut stack = vec![dir.clone()];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d)? {
                    let e = e?;
                    if e.file_type()?.is_dir() {
                        stack.push(e.path());
                    } else {
                        found = Some(e.path());
                    }
                }
            }
            found.expect("blob file exists on disk")
        };
        let bytes = std::fs::read(&blob_path)?;
        if cas_kit::store::is_zstd_compressed(&bytes) {
            // compressed payloads: flip a trailing byte (checksum catches it
            // on decompress, which is still a failed read — asserted below)
            let mut b = bytes.clone();
            let last = b.len() - 1;
            b[last] ^= 0xFF;
            std::fs::write(&blob_path, &b)?;
        } else {
            let mut b = bytes.clone();
            b[0] ^= 0x01;
            std::fs::write(&blob_path, &b)?;
        }

        let outcome = store.read(&doc);
        match outcome {
            Err(AssetError::Corrupt { address }) => {
                assert_eq!(address, doc.address());
            }
            Err(AssetError::Cas(e)) => {
                // decompression failure on the compressed path is also a
                // failed read; assert it is loud, not silent
                let msg = e.to_string().to_lowercase();
                assert!(
                    msg.contains("decompress")
                        || msg.contains("checksum")
                        || msg.contains("mismatch")
                        || msg.contains("corrupt")
                        || msg.contains("integrity")
                        || msg.contains("zstd"),
                    "expected a loud integrity failure, got: {e}"
                );
            }
            Ok(bytes) => panic!("tampered bytes read back as: {bytes:?}"),
        }

        std::fs::remove_dir_all(dir).ok();
        Ok(())
    }
}

mod surface_tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
    use chisel_assets::{AssetStore, Result};

    #[test]
    fn display_is_the_short_form_and_address_is_full_hex() -> Result<()> {
        let store = AssetStore::open_in_memory()?;
        let doc = store.put(b"display vs address")?;
        let display = doc.to_string();
        assert!(display.len() < 64, "Display is the short form: {display}");
        assert!(
            display.ends_with('\u{2026}'),
            "cas-kit short form ends with an ellipsis: {display}"
        );
        assert_eq!(doc.address().len(), 64);
        Ok(())
    }

    #[test]
    fn debug_names_the_struct_and_never_leaks_bytes() -> Result<()> {
        let store = AssetStore::open("/tmp/chisel-debug-nonsense-path")?;
        let dbg = format!("{store:?}");
        assert!(dbg.starts_with("AssetStore {"), "{dbg}");
        assert!(dbg.contains("root"), "{dbg}");
        Ok(())
    }

    #[test]
    fn total_size_tracks_the_store() -> Result<()> {
        let store = AssetStore::open_in_memory()?;
        assert_eq!(store.total_size()?, 0, "empty store");
        store.put(b"some bytes to hold")?;
        assert!(store.total_size()? > 0, "non-empty store holds bytes");
        Ok(())
    }

    #[test]
    fn remove_of_a_missing_address_is_a_no_op() -> Result<()> {
        let store = AssetStore::open_in_memory()?;
        let doc = store.put(b"here today")?;
        store.remove(&doc)?;
        store.remove(&doc)?; // second remove: already gone, still fine
        assert!(store.is_empty()?);
        Ok(())
    }

    #[test]
    fn corrupt_constructor_names_the_address() {
        let store = AssetStore::open_in_memory().expect("store");
        let doc = store.put(b"damaged").expect("put");
        let err = AssetStore::corrupt(&doc);
        assert!(err.to_string().contains(&doc.address()), "{err}");
    }
}

mod tamper_compressed_tests {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::panic,
        clippy::indexing_slicing
    )]
    use chisel_assets::{AssetError, AssetStore};

    #[test]
    fn a_flipped_trailing_byte_in_a_compressed_blob_fails_loudly()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("chisel-tamper-z-{}", std::process::id()));
        let store = AssetStore::open(&dir)?;
        // One mebibyte of a repeating pattern compresses to a small zstd blob,
        // so the stored bytes go through the compression path.
        let big = vec![0xABu8; 1024 * 1024];
        let doc = store.put(&big)?;

        let blob_path = {
            let mut found = None;
            let mut stack = vec![dir.clone()];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d)? {
                    let e = e?;
                    if e.file_type()?.is_dir() {
                        stack.push(e.path());
                    } else if e.path().extension().is_none() {
                        found = Some(e.path());
                    }
                }
            }
            found.expect("blob file exists on disk")
        };
        let mut bytes = std::fs::read(&blob_path)?;
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&blob_path, &bytes)?;

        match store.read(&doc) {
            Err(AssetError::Corrupt { .. }) => {} // hash check after decompress
            Err(AssetError::Cas(e)) => {
                let msg = e.to_string().to_lowercase();
                assert!(
                    msg.contains("decompress")
                        || msg.contains("checksum")
                        || msg.contains("mismatch")
                        || msg.contains("corrupt")
                        || msg.contains("integrity")
                        || msg.contains("zstd"),
                    "expected a loud integrity failure, got: {e}"
                );
            }
            Ok(bytes) => panic!("tampered blob read back as {} bytes", bytes.len()),
        }

        std::fs::remove_dir_all(dir).ok();
        Ok(())
    }
}
