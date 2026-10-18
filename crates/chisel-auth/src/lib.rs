//! Chisel platform authentication: JWT sessions with a closed role
//! vocabulary, and the OIDC desktop-loopback seam over the kit estate.
//!
//! Two kit surfaces composed here, both kept behind thin Chisel vocabulary:
//!
//! - [`tokenkit`](https://crates.io/crates/tokenkit) — JWT issue/validate
//!   with rotation-capable configuration ([`SessionService`]).
//! - [`oauth_toolkit`](https://crates.io/crates/oauth-toolkit) — the OIDC
//!   desktop-loopback flow with PKCE and CSRF state ([`LoginFlow`]).
//!
//! The v0.1 rule: everything here is testable offline. The one network step
//! in OIDC (the authorization-code exchange at the provider's token
//! endpoint) is behind the [`TokenExchanger`] trait — the HTTP
//! implementation arrives with the service slice, and tests inject a stub.

use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};

/// Authentication failures.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The presented token is invalid, expired, or wrongly scoped.
    #[error("invalid session token: {0}")]
    Token(String),
    /// The actor's roles do not permit the operation.
    #[error("role {actor_role:?} does not permit {needed:?}")]
    Forbidden {
        /// The role the session carried.
        actor_role: String,
        /// The capability that was requested.
        needed: String,
    },
    /// The underlying kit rejected an operation.
    #[error("kit error: {0}")]
    Kit(String),
    /// A configuration error (empty secret, malformed issuer, …).
    #[error("configuration error: {0}")]
    Config(String),
}

/// Convenient alias for auth operations.
pub type Result<T> = std::result::Result<T, Error>;

/// The closed role vocabulary for the platform, declared weakest-first so
/// the derived ordering matches strength (`Auditor < Reviewer < Editor <
/// Owner`); capabilities accumulate upward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Read-only, plus the full audit trail.
    Auditor,
    /// Read and comment; no mutations.
    Reviewer,
    /// Edit documents and manage redactions.
    Editor,
    /// Full control, including membership and deletion.
    Owner,
}

impl Role {
    /// The role's name as serialized into tokens.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Owner => "owner",
            Role::Editor => "editor",
            Role::Reviewer => "reviewer",
            Role::Auditor => "auditor",
        }
    }

    /// Parses a role name from a token claim.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "owner" => Some(Role::Owner),
            "editor" => Some(Role::Editor),
            "reviewer" => Some(Role::Reviewer),
            "auditor" => Some(Role::Auditor),
            _ => None,
        }
    }

    /// Whether this role permits the given capability. Capabilities are a
    /// closed set; new capabilities must name every role that holds them.
    #[must_use]
    pub fn permits(self, capability: Capability) -> bool {
        match capability {
            Capability::Read => true,
            Capability::ReadAudit => true,
            Capability::Edit => matches!(self, Role::Owner | Role::Editor),
            Capability::Redact => matches!(self, Role::Owner | Role::Editor),
            Capability::Sign => matches!(self, Role::Owner | Role::Editor),
            Capability::ManageMembers => matches!(self, Role::Owner),
            Capability::Delete => matches!(self, Role::Owner),
        }
    }
}

/// The closed capability set. A capability is a verb-like permission the UI
/// and the service both check before acting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Read document content.
    Read,
    /// Read the audit trail (every role can).
    ReadAudit,
    /// Mutate document content.
    Edit,
    /// Apply redactions.
    Redact,
    /// Apply digital signatures.
    Sign,
    /// Manage membership.
    ManageMembers,
    /// Delete documents.
    Delete,
}

/// The claims carried by a Chisel session token: `tokenkit`'s standard
/// claims (whose `role` field carries the session's role name) plus the
/// session id.
///
/// A session carries exactly one role — the Acrobat actor model is one
/// identity, one job — so the vocabulary stays a flat enum, not a set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiselClaims {
    /// The standard registered claims; `role` holds the role name.
    #[serde(flatten)]
    pub standard: tokenkit::claims::StandardClaims,
    /// The session this token belongs to (refresh pairs share it).
    pub sid: String,
}

impl ChiselClaims {
    /// The session's parsed role, if the name is in the current vocabulary.
    #[must_use]
    pub fn role(&self) -> Option<Role> {
        self.standard.role.as_deref().and_then(Role::parse)
    }

    /// Whether the session's role permits a capability.
    #[must_use]
    pub fn permits(&self, capability: Capability) -> bool {
        self.role().is_some_and(|r| r.permits(capability))
    }

    /// Enforces a capability, mapping failure to [`Error::Forbidden`].
    ///
    /// # Errors
    ///
    /// When the session's role does not permit the capability.
    pub fn require(&self, capability: Capability) -> Result<()> {
        if self.permits(capability) {
            Ok(())
        } else {
            Err(Error::Forbidden {
                actor_role: self.role().map_or("none".into(), |r| r.as_str().to_owned()),
                needed: format!("{capability:?}"),
            })
        }
    }
}

/// Issues and validates Chisel session tokens over `tokenkit`'s JWT service
/// with rotation-capable secrets.
#[derive(Clone)]
pub struct SessionService {
    inner: std::sync::Arc<tokenkit::service::JwtService>,
    issuer: String,
    audience: String,
    session_ttl: Duration,
    refresh_ttl: Duration,
}

impl std::fmt::Debug for SessionService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionService")
            .field("issuer", &self.issuer)
            .field("audience", &self.audience)
            .field("session_ttl", &self.session_ttl)
            .finish_non_exhaustive()
    }
}

/// The token pair handed out at login and refresh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTokens {
    /// Short-lived access token (carries the roles).
    pub access: String,
    /// Long-lived refresh token (same `sid`, longer expiry).
    pub refresh: String,
    /// The session id both tokens share.
    pub sid: String,
}

impl SessionService {
    /// Creates a service.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when the signing secret is empty or the issuer or
    /// audience is empty — a service started unconfigured must fail loudly
    /// now, not at the first login.
    pub fn new(
        signing_secret: impl Into<String>,
        issuer: impl Into<String>,
        audience: impl Into<String>,
    ) -> Result<Self> {
        let secret = signing_secret.into();
        let issuer = issuer.into();
        let audience = audience.into();
        if secret.is_empty() {
            return Err(Error::Config("signing secret cannot be empty".into()));
        }
        if issuer.is_empty() || audience.is_empty() {
            return Err(Error::Config("issuer and audience cannot be empty".into()));
        }
        // v0.1 pins a single HS256 secret. tokenkit's secret-rotation
        // builder is behind its `rotation` feature; enable it here when the
        // service slice lands and the secrets move to the settings store.
        // The config carries the issuer and audience so VALIDATION enforces
        // that tokens presented to this service are its own — not just
        // correctly signed by it.
        let config = tokenkit::service::JwtConfig {
            secret,
            issuer: Some(issuer.clone()),
            audience: Some(audience.clone()),
            ..tokenkit::service::JwtConfig::default()
        };
        Ok(Self {
            inner: std::sync::Arc::new(tokenkit::service::JwtService::new(config)),
            issuer,
            audience,
            session_ttl: Duration::hours(12),
            refresh_ttl: Duration::days(30),
        })
    }

    /// Issues an access + refresh pair for a user with the given role.
    ///
    /// # Errors
    ///
    /// Propagates encoding failures from the kit.
    pub fn issue(&self, user: &chisel_kernel::UserId, role: Role) -> Result<SessionTokens> {
        let sid = chisel_kernel::SessionId::try_from(format!(
            "{}-{}",
            user.as_str(),
            Utc::now().timestamp_millis()
        ))
        .map_err(Error::Config)?;
        let access = self.encode_session(user, role, &sid, self.session_ttl)?;
        let refresh = self.encode_session(user, role, &sid, self.refresh_ttl)?;
        Ok(SessionTokens {
            access,
            refresh,
            sid: sid.to_string(),
        })
    }

    /// Validates an access token into its claims.
    ///
    /// # Errors
    ///
    /// [`Error::Token`] when the kit rejects the token (signature, expiry,
    /// audience); propagation otherwise.
    pub fn validate(&self, token: &str) -> Result<ChiselClaims> {
        self.inner
            .decode::<ChiselClaims>(token)
            .map_err(|e| Error::Token(e.to_string()))
    }

    /// Validates a refresh token and issues a fresh pair for the same
    /// session, preserving the role recorded at login.
    ///
    /// # Errors
    ///
    /// Same as [`validate`](Self::validate).
    pub fn refresh(&self, refresh_token: &str) -> Result<SessionTokens> {
        let claims = self.validate(refresh_token)?;
        let role = claims.role().ok_or_else(|| {
            Error::Token("refresh token carries no role from the current vocabulary".into())
        })?;
        let user = chisel_kernel::UserId::try_from(claims.standard.sub.clone().unwrap_or_default())
            .map_err(Error::Token)?;
        let sid = chisel_kernel::SessionId::try_from(claims.sid.clone()).map_err(Error::Token)?;
        let access = self.encode_session(&user, role, &sid, self.session_ttl)?;
        let refresh = self.encode_session(&user, role, &sid, self.refresh_ttl)?;
        Ok(SessionTokens {
            access,
            refresh,
            sid: sid.to_string(),
        })
    }

    fn encode_session(
        &self,
        user: &chisel_kernel::UserId,
        role: Role,
        sid: &chisel_kernel::SessionId,
        ttl: Duration,
    ) -> Result<String> {
        let now = Utc::now();
        let claims = ChiselClaims {
            standard: tokenkit::claims::StandardClaims {
                sub: Some(user.as_str().to_owned()),
                iss: Some(self.issuer.clone()),
                aud: Some(self.audience.clone()),
                exp: Some(now + ttl),
                iat: Some(now),
                jti: None,
                role: Some(role.as_str().to_owned()),
                permissions: Vec::new(),
                extra: std::collections::HashMap::new(),
            },
            sid: sid.to_string(),
        };
        self.inner
            .encode(&claims)
            .map_err(|e| Error::Kit(e.to_string()))
    }
}

/// The OIDC provider facts the desktop login needs. The app supplies them
/// (from its settings or the provider's discovery document).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OidcProviderConfig {
    /// The provider's issuer URL (e.g. `https://id.example.com`).
    pub issuer: String,
    /// The authorization endpoint.
    pub authorization_url: String,
    /// The token endpoint (used by the service-side exchanger).
    pub token_url: String,
    /// The OIDC client id registered for the desktop app.
    pub client_id: String,
    /// Scopes to request; `openid email profile` is the floor.
    pub scopes: Vec<String>,
}

/// The outcome of starting a desktop login: open `url` in a browser, then
/// `wait()` for the loopback capture.
pub struct LoginStart {
    flow: oauth_toolkit::loopback::LoopbackFlow,
}

impl LoginStart {
    /// The URL to open in the system browser.
    #[must_use]
    pub fn url(&self) -> &str {
        self.flow.authorization_url()
    }

    /// Blocks for up to the flow's timeout waiting for the provider to call
    /// back on the loopback listener.
    ///
    /// # Errors
    ///
    /// Propagates the kit's capture errors (timeout, state mismatch, deny).
    pub fn wait(self) -> Result<oauth_toolkit::loopback::CapturedCode> {
        self.flow
            .wait_for_code()
            .map_err(|e| Error::Kit(e.to_string()))
    }
}

/// Exchanges an authorization code for OIDC tokens at the provider's token
/// endpoint. The HTTP implementation belongs to the service slice; tests
/// inject closures.
pub trait TokenExchanger: Send + Sync {
    /// The token response as a raw JSON value (the caller validates it via
    /// the kit's OIDC validator).
    fn exchange(
        &self,
        token_url: &str,
        code: &str,
        code_verifier: &str,
        client_id: &str,
    ) -> Result<serde_json::Value>;
}

impl<F> TokenExchanger for F
where
    F: Fn(&str, &str, &str, &str) -> Result<serde_json::Value> + Send + Sync,
{
    fn exchange(
        &self,
        token_url: &str,
        code: &str,
        code_verifier: &str,
        client_id: &str,
    ) -> Result<serde_json::Value> {
        self(token_url, code, code_verifier, client_id)
    }
}

/// Starts a desktop OIDC login against `provider`: builds the PKCE pair and
/// CSRF state via the kit, binds the loopback listener, and returns the URL
/// to open. Complete the flow by waiting and exchanging the code.
///
/// # Errors
///
/// Propagates the kit's flow-start errors (bind failure, URL build).
pub fn start_login(
    provider: &OidcProviderConfig,
    scopes: &[String],
    timeout: std::time::Duration,
) -> Result<LoginStart> {
    let flow = oauth_toolkit::loopback::LoopbackFlow::start(
        &provider.authorization_url,
        &provider.client_id,
        scopes,
        None,
        timeout,
    )
    .map_err(|e| Error::Kit(e.to_string()))?;
    Ok(LoginStart { flow })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use super::*;
    use chisel_kernel::UserId;

    const SECRET: &str = "test-signing-secret-do-not-use";
    const ISSUER: &str = "https://auth.chisel.test";
    const AUDIENCE: &str = "chisel-desktop";

    fn service() -> SessionService {
        SessionService::new(SECRET, ISSUER, AUDIENCE).expect("valid config")
    }

    #[test]
    fn config_failures_are_loud_and_early() {
        assert!(SessionService::new("", ISSUER, AUDIENCE).is_err());
        assert!(SessionService::new(SECRET, "", AUDIENCE).is_err());
        assert!(SessionService::new(SECRET, ISSUER, "").is_err());
    }

    #[test]
    fn issue_then_validate_round_trips_role_and_identity() -> Result<()> {
        let svc = service();
        let user = UserId::try_from("alice").map_err(Error::Token)?;
        let tokens = svc.issue(&user, Role::Editor)?;

        let claims = svc.validate(&tokens.access)?;
        assert_eq!(claims.standard.sub.as_deref(), Some("alice"));
        assert_eq!(claims.standard.role.as_deref(), Some("editor"));
        assert_eq!(claims.sid, tokens.sid);
        assert!(claims.permits(Capability::Edit));
        assert!(!claims.permits(Capability::ManageMembers));
        Ok(())
    }

    #[test]
    fn refresh_preserves_the_session_and_role() -> Result<()> {
        let svc = service();
        let user = UserId::try_from("bob").map_err(Error::Token)?;
        let tokens = svc.issue(&user, Role::Auditor)?;

        let fresh = svc.refresh(&tokens.refresh)?;
        assert_eq!(fresh.sid, tokens.sid);
        let claims = svc.validate(&fresh.access)?;
        assert_eq!(claims.standard.role.as_deref(), Some("auditor"));
        assert!(claims.permits(Capability::ReadAudit));
        assert!(!claims.permits(Capability::Edit));
        Ok(())
    }

    #[test]
    fn a_foreign_issuer_is_rejected() -> Result<()> {
        let svc = service();
        let user = UserId::try_from("alice").map_err(Error::Token)?;
        let tokens = svc.issue(&user, Role::Reviewer)?;

        let other = SessionService::new(SECRET, "https://elsewhere.test", AUDIENCE)?;
        assert!(matches!(
            other.validate(&tokens.access),
            Err(Error::Token(_))
        ));
        Ok(())
    }

    #[test]
    fn capability_checks_name_the_missing_role() -> Result<()> {
        let svc = service();
        let user = UserId::try_from("dan").map_err(Error::Token)?;
        let tokens = svc.issue(&user, Role::Reviewer)?;
        let claims = svc.validate(&tokens.access)?;
        let err = claims.require(Capability::Redact).unwrap_err();
        assert!(err.to_string().contains("reviewer"), "{err}");
        assert!(err.to_string().contains("Redact"), "{err}");
        Ok(())
    }

    #[test]
    fn the_role_vocabulary_is_total_and_ordered() {
        for (name, role) in [
            ("owner", Role::Owner),
            ("editor", Role::Editor),
            ("reviewer", Role::Reviewer),
            ("auditor", Role::Auditor),
        ] {
            assert_eq!(Role::parse(name), Some(role));
            assert_eq!(role.as_str(), name);
        }
        assert!(Role::Auditor < Role::Reviewer);
        assert!(Role::Reviewer < Role::Editor);
        assert!(Role::Editor < Role::Owner);
        for r in [Role::Owner, Role::Editor, Role::Reviewer, Role::Auditor] {
            assert!(r.permits(Capability::Read));
            assert!(!r.permits(Capability::Delete) || r == Role::Owner);
        }
    }

    #[test]
    fn start_login_builds_a_loopback_flow_against_the_provider() -> Result<()> {
        let provider = OidcProviderConfig {
            issuer: "https://id.example.test".into(),
            authorization_url: "https://id.example.test/authorize".into(),
            token_url: "https://id.example.test/token".into(),
            client_id: "chisel-desktop".into(),
            scopes: vec!["openid".into(), "email".into(), "profile".into()],
        };
        let start = start_login(
            &provider,
            &provider.scopes,
            std::time::Duration::from_secs(30),
        )?;
        assert!(
            start.url().starts_with("https://id.example.test/authorize"),
            "{}",
            start.url()
        );
        assert!(
            start.url().contains("code_challenge="),
            "PKCE challenge present"
        );
        assert!(start.url().contains("state="), "CSRF state present");
        Ok(())
    }
}
