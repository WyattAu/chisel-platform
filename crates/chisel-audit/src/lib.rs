//! Tamper-evident document audit trail for Chisel.
//!
//! Wraps the estate's [`tamper_audit`] chain in the document vocabulary the
//! PDF editor already speaks: every document mutation (open, edit, redact,
//! sign, export) becomes one append with an actor, an action, a document
//! reference, and a JSON detail payload. The SHA-256 chain underneath makes
//! the trail verifiable after the fact — the defence-grade property this
//! crate exists for.
//!
//! # Layering
//!
//! `chisel-audit` is deliberately a thin vocabulary layer, not a second
//! implementation: chaining, verification, and querying are all
//! [`tamper_audit`]'s tested surface. What this crate adds is the
//! domain-constrained API (`AuditEvent`, `DocumentRef`), the document-level
//! error type, and the seams the service side will continue later.
//!
//! # Example
//!
//! ```no_run
//! # async fn demo() -> chisel_audit::Result<()> {
//! use chisel_audit::{AuditEvent, DocumentRef, DocumentTrail};
//!
//! let trail = DocumentTrail::in_memory();
//! let doc = DocumentRef::new("contract.pdf");
//!
//! trail.record(AuditEvent::opened("alice", &doc)).await?;
//! trail.record(AuditEvent::redacted("bob", &doc, "page 3, para 2")).await?;
//!
//! let report = trail.verify().await?;
//! assert!(report.is_valid());
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;
use tamper_audit::AuditLog;

/// Audit failures, re-exposed with the document context callers actually have.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The underlying chain rejected the append or a query.
    #[error("audit chain error: {0}")]
    Chain(#[from] tamper_audit::AuditError),
    /// A detail payload could not be serialized; the event was NOT recorded.
    #[error("event detail is not serializable: {0}")]
    Detail(#[from] serde_json::Error),
}

/// Convenient alias for fallible audit operations.
pub type Result<T> = std::result::Result<T, Error>;

/// A document within the trail: the file name the editor shows, stable across
/// renames for the life of the session. (A content hash joins in
/// `chisel-assets`; the audit layer deliberately does not depend on it.)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct DocumentRef(String);

impl DocumentRef {
    /// Names the document being audited.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The document name as recorded in trail entries.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DocumentRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One document mutation, in the editor's vocabulary.
///
/// Constructors fix the `action` string per variant so trail queries can rely
/// on a closed action vocabulary instead of free-form strings.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct AuditEvent {
    /// Who performed the mutation (user name, or `system` for automation).
    pub actor: String,
    /// The closed-vocabulary action, e.g. `document.open`, `redact.apply`.
    pub action: String,
    /// The document the mutation applies to.
    pub document: DocumentRef,
    /// Free-form detail: what changed, where, and why.
    pub detail: Value,
}

impl AuditEvent {
    fn new(actor: &str, action: &str, document: &DocumentRef, detail: Value) -> Self {
        Self {
            actor: actor.to_owned(),
            action: action.to_owned(),
            document: document.clone(),
            detail,
        }
    }

    /// A document was opened for editing.
    #[must_use]
    pub fn opened(actor: &str, document: &DocumentRef) -> Self {
        Self::new(actor, "document.open", document, Value::Null)
    }

    /// Content was modified on `where_` (page, object path, …).
    #[must_use]
    pub fn edited(actor: &str, document: &DocumentRef, where_: &str) -> Self {
        Self::new(actor, "document.edit", document, Value::from(where_))
    }

    /// A redaction was applied over `scope` — recorded because redactions are
    /// the mutations a reviewer will ask about first.
    #[must_use]
    pub fn redacted(actor: &str, document: &DocumentRef, scope: &str) -> Self {
        Self::new(actor, "redact.apply", document, Value::from(scope))
    }

    /// A digital signature was applied.
    #[must_use]
    pub fn signed(actor: &str, document: &DocumentRef, field: &str) -> Self {
        Self::new(actor, "sign.apply", document, Value::from(field))
    }

    /// The document was exported (format in `detail`).
    #[must_use]
    pub fn exported(actor: &str, document: &DocumentRef, format: &str) -> Self {
        Self::new(actor, "document.export", document, Value::from(format))
    }
}

/// The document audit trail: an append-only, SHA-256-chained, queryable log.
///
/// Cheap to clone — the underlying log is shared behind an `Arc`; every handle
/// sees the same chain, which is the point.
#[derive(Clone)]
pub struct DocumentTrail {
    inner: Arc<AuditLog>,
}

impl Default for DocumentTrail {
    fn default() -> Self {
        Self::in_memory()
    }
}

impl DocumentTrail {
    /// A trail kept in memory (the desktop path; the service continuation
    /// takes over on publish).
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            inner: Arc::new(AuditLog::new()),
        }
    }

    /// Append one mutation to the chain. The payload is the event serialized
    /// as-is, so the chain verifies bytes the UI could reconstruct.
    ///
    /// # Errors
    ///
    /// Propagates the chain's append error; a detail payload that cannot
    /// serialize is a [`Error::Detail`] and nothing is recorded (audit entries
    /// are never half-written).
    pub async fn record(&self, event: AuditEvent) -> Result<()> {
        let payload = serde_json::to_value(&event)?;
        self.inner
            .append(
                &event.actor,
                &event.action,
                event.document.as_str(),
                payload,
            )
            .await?;
        Ok(())
    }

    /// Verify the whole chain from genesis.
    ///
    /// # Errors
    ///
    /// Propagates I/O or encoding errors from the store; a broken chain is a
    /// *report*, not an error — callers decide policy.
    pub async fn verify(&self) -> Result<tamper_audit::verify::VerificationResult> {
        Ok(self.inner.verify_chain().await?)
    }

    /// Every recorded mutation for one document, oldest first.
    ///
    /// # Errors
    ///
    /// Propagates query errors from the store.
    pub async fn for_document(
        &self,
        document: &DocumentRef,
    ) -> Result<Vec<tamper_audit::AuditEntry>> {
        Ok(self.inner.query_by_resource(document.as_str()).await?)
    }

    /// Every recorded mutation by one actor, oldest first.
    ///
    /// # Errors
    ///
    /// Propagates query errors from the store.
    pub async fn by_actor(&self, actor: &str) -> Result<Vec<tamper_audit::AuditEntry>> {
        Ok(self.inner.query_by_actor(actor).await?)
    }
}

/// Chain-verification result, re-exported so callers need not name
/// `tamper_audit` to check a trail.
pub use tamper_audit::verify::VerificationResult;

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use super::*;

    #[tokio::test]
    async fn a_recorded_trail_verifies_and_queries_by_document() -> Result<()> {
        let trail = DocumentTrail::in_memory();
        let doc = DocumentRef::new("lease.pdf");

        trail.record(AuditEvent::opened("alice", &doc)).await?;
        trail
            .record(AuditEvent::edited("alice", &doc, "page 1"))
            .await?;
        trail
            .record(AuditEvent::redacted("bob", &doc, "page 2 SSN"))
            .await?;

        let report = trail.verify().await?;
        assert!(report.is_valid(), "chain must verify after appends");

        let entries = trail.for_document(&doc).await?;
        assert_eq!(entries.len(), 3);
        let actions: Vec<&str> = entries.iter().map(|e| e.action.as_str()).collect();
        assert_eq!(actions, ["document.open", "document.edit", "redact.apply"]);
        Ok(())
    }

    #[tokio::test]
    async fn two_trails_do_not_share_a_chain() -> Result<()> {
        let a = DocumentTrail::in_memory();
        let b = DocumentTrail::in_memory();
        let doc = DocumentRef::new("doc.pdf");

        a.record(AuditEvent::opened("alice", &doc)).await?;
        b.record(AuditEvent::opened("bob", &doc)).await?;

        assert_eq!(a.for_document(&doc).await?.len(), 1);
        assert_eq!(b.for_document(&doc).await?.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn actor_queries_partition_the_trail() -> Result<()> {
        let trail = DocumentTrail::in_memory();
        let doc = DocumentRef::new("doc.pdf");

        trail.record(AuditEvent::opened("alice", &doc)).await?;
        trail
            .record(AuditEvent::redacted("bob", &doc, "p1"))
            .await?;
        trail
            .record(AuditEvent::exported("alice", &doc, "pdf/a"))
            .await?;

        let alice = trail.by_actor("alice").await?;
        assert_eq!(alice.len(), 2);
        assert!(alice.iter().all(|e| e.actor == "alice"));
        Ok(())
    }
}
