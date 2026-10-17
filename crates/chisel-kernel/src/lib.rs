//! Chisel platform kernel: typed configuration and validated identifiers.
//!
//! Two jobs, both boring on purpose:
//!
//! 1. **Typed, layered settings** for the desktop app and the service, driven
//!    by the estate's [`config_kit`] (file → env → overrides, strict keys,
//!    redacted secrets).
//! 2. **Validated identifiers** — [`DocumentId`], [`UserId`], [`SessionId`] —
//!    so the rest of the platform cannot pass a filename where a document id
//!    belongs.
//!
//! Deliberately absent: anything PDF-shaped. The editor owns that; the kernel
//! only carries what every platform crate needs.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Kernel failures.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Settings failed to load or validate.
    #[error("configuration error: {0}")]
    Config(String),
}

/// Convenient alias for kernel operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Generates a newtype over `String` that validates on construction and
/// refuses mutation, so a `UserId` can never hold an empty or whitespace name.
macro_rules! validated_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl $name {
            fn validate(raw: &str) -> std::result::Result<(), String> {
                if raw.is_empty() {
                    return Err(concat!(stringify!($name), " cannot be empty").to_owned());
                }
                if raw.chars().any(char::is_whitespace) {
                    return Err(concat!(stringify!($name), " cannot contain whitespace").to_owned());
                }
                Ok(())
            }

            /// The identifier as borrowed text.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl TryFrom<String> for $name {
            type Error = String;

            fn try_from(raw: String) -> std::result::Result<Self, Self::Error> {
                Self::validate(&raw)?;
                Ok(Self(raw))
            }
        }

        impl TryFrom<&str> for $name {
            type Error = String;

            fn try_from(raw: &str) -> std::result::Result<Self, Self::Error> {
                Self::try_from(raw.to_owned())
            }
        }

        impl From<$name> for String {
            fn from(id: $name) -> Self {
                id.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

validated_id! {
    /// A document identity inside the platform (not the file name; the stable
    /// id the assets store indexes by).
    DocumentId
}
validated_id! {
    /// A human or machine actor on the platform.
    UserId
}
validated_id! {
    /// An authenticated session.
    SessionId
}

#[cfg(test)]
mod id_tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use super::*;

    #[test]
    fn ids_accept_plain_names() {
        assert!(DocumentId::try_from("contract-2026").is_ok());
        assert!(UserId::try_from("alice").is_ok());
    }

    #[test]
    fn ids_reject_empty_and_whitespace() {
        assert!(DocumentId::try_from("").is_err());
        assert!(UserId::try_from("alice smith").is_err());
        assert!(SessionId::try_from("\t").is_err());
    }

    #[test]
    fn ids_round_trip_through_serde() {
        let doc: DocumentId = serde_json::from_str("\"contract-2026\"").expect("valid json");
        assert_eq!(doc.as_str(), "contract-2026");
        assert!(serde_json::from_str::<DocumentId>("\"\"").is_err());
    }

    #[test]
    fn display_matches_inner() {
        let user = UserId::try_from("bob").expect("valid");
        assert_eq!(user.to_string(), "bob");
    }
}

/// Service settings: everything the platform side needs before a single
/// document moves. Secrets live in [`Sensitive`](config_kit::Sensitive) so a
/// stray `{:?}` cannot leak them. Deliberately not `Clone` and **not
/// `Serialize`**: `Sensitive`'s serializer is opt-in for a reason — a settings
/// struct that can be serialized is one debug endpoint away from leaking its
/// DSN.
#[derive(Debug, Deserialize)]
pub struct ServiceSettings {
    /// Bind address for the HTTP service.
    #[serde(default = "default_bind")]
    pub bind: String,
    /// Postgres DSN for the outbox and trail continuation.
    pub database_url: config_kit::Sensitive<String>,
    /// Blob store root (S3 URL or a filesystem path).
    pub asset_store: String,
}

fn default_bind() -> String {
    "127.0.0.1:8080".to_owned()
}

/// Desktop settings: what the editor itself reads at startup. Everything
/// optional-with-default, because a fresh install must open with zero config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DesktopSettings {
    /// Where team assets are fetched from; absent means local-only mode.
    #[serde(default)]
    pub team_library: Option<String>,
    /// Opt-in telemetry endpoint; absent means fully off.
    #[serde(default)]
    pub telemetry_endpoint: Option<String>,
}

impl ServiceSettings {
    /// Loads service settings from `file`, then `CHISEL_*` env — the
    /// documented precedence (later layers win), strict about unknown keys.
    ///
    /// # Errors
    ///
    /// Any layer failing to parse, or strict unknown keys.
    pub fn load(file: &std::path::Path) -> Result<Self> {
        config_kit::ConfigBuilder::new()
            .layer(config_kit::ConfigLayer::file(file))
            .layer(config_kit::ConfigLayer::env_prefix("CHISEL"))
            .load_strict()
            .map_err(|e| Error::Config(e.to_string()))
    }
}

impl DesktopSettings {
    /// Loads desktop settings leniently (unknown keys ignored: an old config
    /// file must not strand a fresh install) with no file when absent.
    ///
    /// # Errors
    ///
    /// Only when a file exists and fails to parse.
    pub fn load(file: Option<&std::path::Path>) -> Result<Self> {
        match file {
            Some(f) => config_kit::ConfigBuilder::new()
                .layer(config_kit::ConfigLayer::file(f))
                .load()
                .map_err(|e| Error::Config(e.to_string())),
            None => Ok(Self::default()),
        }
    }
}

#[cfg(test)]
mod settings_tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use super::*;
    use std::io::Write as _;

    fn tmp_file(name: &str, contents: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("{name}-{}", std::process::id()));
        let mut f = std::fs::File::create(&p).expect("create temp file");
        f.write_all(contents.as_bytes()).expect("write temp file");
        p
    }

    #[test]
    fn service_settings_load_strictly() {
        let p = tmp_file(
            "svc",
            r#"bind = "127.0.0.1:9000"
database_url = "postgres://localhost/chisel"
asset_store = "/var/lib/chisel"
"#,
        );
        let s = ServiceSettings::load(&p).expect("loads");
        assert_eq!(s.bind, "127.0.0.1:9000");
        assert_eq!(s.database_url.expose(), "postgres://localhost/chisel");
        // redaction: Debug must not leak the DSN
        let dbg = format!("{:?}", s.database_url);
        assert!(!dbg.contains("postgres"), "secrets must be redacted: {dbg}");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn service_settings_reject_unknown_keys() {
        let p = tmp_file(
            "svc-bad",
            r#"bind = "127.0.0.1:9000"
database_url = "postgres://localhost/chisel"
asset_store = "/var/lib/chisel"
mystery = true
"#,
        );
        assert!(
            ServiceSettings::load(&p).is_err(),
            "unknown keys must fail strict load"
        );
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn desktop_settings_default_when_no_file() {
        let s = DesktopSettings::load(None).expect("defaults");
        assert!(s.team_library.is_none());
        assert!(s.telemetry_endpoint.is_none());
    }

    #[test]
    fn desktop_settings_tolerate_unknown_keys() {
        let p = tmp_file(
            "desk",
            r#"team_library = "https://team.example"
leftover_from_old_version = 7
"#,
        );
        let s = DesktopSettings::load(Some(&p)).expect("lenient load");
        assert_eq!(s.team_library.as_deref(), Some("https://team.example"));
        std::fs::remove_file(p).ok();
    }
}
