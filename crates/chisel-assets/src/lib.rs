//! Content-addressed document store for Chisel.
//!
//! Every document revision is stored once, addressed by its own hash, and
//! re-verified on the way out. Two properties fall out of that shape and are
//! the reason this crate exists:
//!
//! - **Deduplication by construction.** Saving the same bytes twice stores
//!   them once; there is no "duplicate document" to clean up later.
//! - **Tamper is detectable, not silent.** A flipped bit on disk — corruption,
//!   a sync conflict, an editor of the wrong kind — fails the read instead of
//!   rendering. Verify-on-read is ON by default and turning it off is an
//!   explicit, greppable decision.
//!
//! Thin over [`cas_kit`] by design: hashing (BLAKE3), the on-disk bucket
//! layout, compression, and GC are the kit's tested surface. This crate adds
//! the document nouns ([`StoredDocument`], [`AssetError`]) and the default
//! posture (verify-on-read).
//!
//! # Example
//!
//! ```
//! use chisel_assets::AssetStore;
//!
//! # fn main() -> Result<(), chisel_assets::AssetError> {
//! let store = AssetStore::open_in_memory()?;
//!
//! let doc = store.put(b"%PDF-1.7 revision A")?;
//! let again = store.put(b"%PDF-1.7 revision A")?;
//! assert_eq!(doc.hash(), again.hash(), "identical bytes store once");
//! assert_eq!(store.len()?, 1);
//!
//! let bytes = store.read(&doc)?;
//! assert_eq!(bytes, b"%PDF-1.7 revision A");
//! # Ok(())
//! # }
//! ```

pub use cas_kit::Hash;

use std::path::PathBuf;

use cas_kit::BlobStore;

/// Asset-store failures.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    /// The underlying CAS refused the operation.
    #[error("content store error: {0}")]
    Cas(#[from] cas_kit::CasError),
    /// A stored document failed verification on read: the bytes on disk no
    /// longer match their address. This is the signal to restore from
    /// backup, not to retry.
    #[error("integrity failure at {address}: stored bytes do not match their hash")]
    Corrupt {
        /// The address of the damaged blob.
        address: String,
    },
}

/// Convenient alias for asset operations.
pub type Result<T> = std::result::Result<T, AssetError>;

/// A stored document revision: its content address. Clone freely — the hash
/// is the identity, not a handle to in-memory bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StoredDocument(Hash);

impl StoredDocument {
    /// The BLAKE3 content address.
    #[must_use]
    pub fn hash(&self) -> &Hash {
        &self.0
    }

    /// The address as full 64-char lowercase hex (URLs, audit entries,
    /// repair tooling). For log lines prefer `Display`, which is the short
    /// form cas-kit prints.
    #[must_use]
    pub fn address(&self) -> String {
        self.0.to_hex()
    }
}

/// Short form (first 12 hex chars) — for log lines and panel rows. The full
/// address is [`StoredDocument::address`].
impl std::fmt::Display for StoredDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The document store. Cheap to clone and share — the CAS is `Send + Sync`
/// and shared behind an `Arc`; every handle sees the same blobs.
#[derive(Clone)]
pub struct AssetStore {
    inner: std::sync::Arc<BlobStore>,
}

impl std::fmt::Debug for AssetStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssetStore")
            .field("root", &self.inner.root())
            .finish()
    }
}

impl AssetStore {
    /// Opens (creating if needed) an on-disk store with verify-on-read ON.
    ///
    /// # Errors
    ///
    /// When the root cannot be created or is unusable.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let mut inner = BlobStore::new(root)?;
        inner.set_verify_on_read(true);
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }
    // (verify-on-read is set before the Arc wrap: BlobStore's verify flag is
    // interior-mutable state, and the posture is per-store, not per-handle.)

    /// An in-memory store with the same posture — for tests and for
    /// local-only sessions that must not touch disk.
    ///
    /// # Errors
    ///
    /// When the scratch directory cannot be created.
    pub fn open_in_memory() -> Result<Self> {
        let (_temp, mut inner) = BlobStore::open_in_memory()?;
        inner.set_verify_on_read(true);
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Stores one document revision and returns its address. Storing the same
    /// bytes again is free: the address already exists, nothing is written.
    ///
    /// # Errors
    ///
    /// Propagates store write errors.
    pub fn put(&self, bytes: &[u8]) -> Result<StoredDocument> {
        let hash = self.inner.put_blob(bytes)?;
        Ok(StoredDocument(hash))
    }

    /// Reads a stored revision, verifying its bytes against the address.
    ///
    /// # Errors
    ///
    /// [`AssetError::Corrupt`] when the bytes on disk no longer match the
    /// address — restore from backup; [`AssetError::Cas`] for absent
    /// addresses and store failures.
    pub fn read(&self, doc: &StoredDocument) -> Result<Vec<u8>> {
        match self.inner.get_blob(doc.hash()) {
            Ok(bytes) => Ok(bytes),
            Err(e) if Self::is_verification(&e) => Err(AssetError::Corrupt {
                address: doc.address(),
            }),
            Err(e) => Err(e.into()),
        }
    }

    /// Whether the store holds this address. Does not read or verify bytes.
    #[must_use]
    pub fn contains(&self, doc: &StoredDocument) -> bool {
        self.inner.has_blob(doc.hash())
    }

    /// Removes one revision. Other documents are unaffected: equal bytes
    /// share an address, so deletion is only safe when the caller knows the
    /// revision is unreferenced — which is GC's job, not this method's.
    ///
    /// # Errors
    ///
    /// Propagates store errors; a missing address is not one.
    pub fn remove(&self, doc: &StoredDocument) -> Result<()> {
        if self.inner.has_blob(doc.hash()) {
            self.inner.delete_blob(doc.hash())?;
        }
        Ok(())
    }

    /// Distinct revisions held.
    ///
    /// # Errors
    ///
    /// Propagates store errors.
    pub fn len(&self) -> Result<u64> {
        Ok(self.inner.blob_count()?)
    }

    /// Whether the store holds no revisions.
    ///
    /// # Errors
    ///
    /// Propagates store errors.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Bytes held on disk, post-compression.
    ///
    /// # Errors
    ///
    /// Propagates store errors.
    pub fn total_size(&self) -> Result<u64> {
        Ok(self.inner.total_size()?)
    }

    /// Every address held, unordered.
    ///
    /// # Errors
    ///
    /// Propagates store errors.
    pub fn revisions(&self) -> Result<Vec<StoredDocument>> {
        Ok(self
            .inner
            .list_blobs()?
            .into_iter()
            .map(StoredDocument)
            .collect())
    }

    /// Marks the address damaged without reading it — for repair tooling.
    #[must_use]
    pub fn corrupt(doc: &StoredDocument) -> AssetError {
        AssetError::Corrupt {
            address: doc.address(),
        }
    }

    fn is_verification(e: &cas_kit::CasError) -> bool {
        matches!(e, cas_kit::CasError::HashMismatch { .. })
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::panic,
        clippy::indexing_slicing
    )]
    use super::*;

    #[test]
    fn round_trip_and_dedup() -> Result<()> {
        let store = AssetStore::open_in_memory()?;
        let bytes = b"%PDF-1.7 the lease, revision 3".to_vec();

        let first = store.put(&bytes)?;
        let second = store.put(&bytes)?;
        assert_eq!(first, second, "equal bytes share an address");
        assert_eq!(store.len()?, 1, "deduplicated by construction");

        assert_eq!(store.read(&first)?, bytes);
        Ok(())
    }

    #[test]
    fn distinct_revisions_get_distinct_addresses() -> Result<()> {
        let store = AssetStore::open_in_memory()?;
        let a = store.put(b"revision 1")?;
        let b = store.put(b"revision 2")?;
        assert_ne!(a, b);
        assert_eq!(store.len()?, 2);
        assert_eq!(store.read(&a)?, b"revision 1");
        assert_eq!(store.read(&b)?, b"revision 2");
        Ok(())
    }

    #[test]
    fn missing_address_is_a_cas_error_not_corruption() -> Result<()> {
        let store = AssetStore::open_in_memory()?;
        let ghost = store.put(b"temporary")?;
        store.remove(&ghost)?;
        assert!(!store.contains(&ghost));
        assert!(matches!(store.read(&ghost), Err(AssetError::Cas(_))));
        Ok(())
    }

    #[test]
    fn empty_store_reports_empty() -> Result<()> {
        let store = AssetStore::open_in_memory()?;
        assert!(store.is_empty()?);
        assert_eq!(store.revisions()?.len(), 0);
        Ok(())
    }

    #[test]
    fn on_disk_store_round_trips() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("chisel-assets-{}", std::process::id()));
        let store = AssetStore::open(&dir)?;
        let doc = store.put(b"persisted revision")?;
        drop(store);

        let reopened = AssetStore::open(&dir)?;
        assert!(reopened.contains(&doc));
        assert_eq!(reopened.read(&doc)?, b"persisted revision");
        std::fs::remove_dir_all(dir).ok();
        Ok(())
    }

    #[test]
    fn address_is_stable_hex() -> Result<()> {
        let store = AssetStore::open_in_memory()?;
        let doc = store.put(b"stable")?;
        let hex = doc.address();
        assert_eq!(hex.len(), 64, "BLAKE3 hex");
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
        assert_eq!(
            cas_kit::Hash::from_hex(&hex).unwrap_or_else(|e| panic!("hex parses: {e:?}")),
            *doc.hash()
        );
        Ok(())
    }
}
