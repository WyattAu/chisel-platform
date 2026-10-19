//! Passkey (WebAuthn) ceremonies over `webauthn-kit`.
//!
//! Wraps the kit's registration/authentication protocol functions behind
//! the Chisel vocabulary: user ids from the kernel, a closed role model,
//! and per-user challenge state. The browser-facing step (shipping the
//! returned options to `navigator.credentials.{create,get}()`) is the
//! app's job; this module verifies the responses and manages the
//! credential store.
//!
//! The ceremony crypto is the kit's tested code (see
//! `webauthn_kit::protocol`); tests inject a stub verifier so the
//! challenge state machine and store interactions here are covered
//! without fabricating ES256 signatures.

use super::{Error, Result};
use chisel_kernel::UserId;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use webauthn_kit::attestation::AttestationPolicy;
use webauthn_kit::config::WebauthnConfig;
use webauthn_kit::credential::{
    AllowCredential, AuthenticationOptions, AuthenticationResult, AuthenticatorSelection,
    ExcludeCredential, PubKeyCredParam, RegistrationOptions, RegistrationResult, RelyingParty,
    WebauthnUser,
};
use webauthn_kit::crypto::generate_challenge_bytes;
use webauthn_kit::policy::CredentialPolicy;
use webauthn_kit::protocol::AuthenticationParams;

/// A stored passkey credential.
#[derive(Debug, Clone)]
pub struct StoredPasskey {
    /// The user this credential belongs to.
    pub user: UserId,
    /// Base64url credential id (the lookup key).
    pub credential_id: String,
    /// COSE public key captured at registration.
    pub public_key_cose: Vec<u8>,
    /// The assertion counter as of the last ceremony.
    pub sign_count: u32,
    /// A human label for the credential list.
    pub label: String,
}

/// Persistence seam for passkey credentials.
pub trait CredentialStore: Send + Sync {
    /// Stores or replaces a credential by its id.
    fn put(&self, credential: StoredPasskey);
    /// Fetches a credential by id.
    fn get(&self, credential_id: &str) -> Option<StoredPasskey>;
    /// Every credential belonging to a user.
    fn for_user(&self, user: &UserId) -> Vec<StoredPasskey>;
    /// Removes a credential by id.
    fn remove(&self, credential_id: &str);
}

/// The ceremony surface of `webauthn-kit`, as a trait: the crypto is the
/// kit's tested code in production; tests inject outcomes so the challenge
/// state machine, store interactions, and error mapping here are covered
/// without fabricating ES256 signatures.
pub trait CeremonyVerifier: Send + Sync {
    /// Verifies a registration ceremony against the stored challenge.
    fn verify_registration(
        &self,
        challenge_bytes: &[u8],
        client_data_json_b64: &str,
        attestation_object_b64: &str,
        rp_id: &str,
        rp_origins: &[String],
    ) -> std::result::Result<RegistrationResult, webauthn_kit::WebauthnError>;

    /// Verifies an authentication ceremony for a known credential.
    fn verify_authentication(
        &self,
        params: &AuthenticationParams,
    ) -> std::result::Result<AuthenticationResult, webauthn_kit::WebauthnError>;
}

/// The production verifier: the kit's protocol functions with the service's
/// config, credential policy, and strict attestation posture.
#[derive(Debug, Clone)]
pub struct KitVerifier {
    /// Relying party id used in ceremony validation.
    pub rp_id: String,
    /// Origins accepted for ceremony validation.
    pub rp_origins: Vec<String>,
    /// Registration-time attestation posture (strict).
    pub attestation: AttestationPolicy,
    /// Registration-time credential policy (algorithm allowlist, UV, backup).
    pub policy: CredentialPolicy,
}

impl CeremonyVerifier for KitVerifier {
    fn verify_registration(
        &self,
        challenge_bytes: &[u8],
        client_data_json_b64: &str,
        attestation_object_b64: &str,
        rp_id: &str,
        rp_origins: &[String],
    ) -> std::result::Result<RegistrationResult, webauthn_kit::WebauthnError> {
        webauthn_kit::protocol::verify_registration(
            challenge_bytes,
            client_data_json_b64,
            attestation_object_b64,
            "",
            rp_id,
            rp_origins,
            &self.attestation,
            &self.policy,
        )
    }

    fn verify_authentication(
        &self,
        params: &AuthenticationParams,
    ) -> std::result::Result<AuthenticationResult, webauthn_kit::WebauthnError> {
        webauthn_kit::protocol::verify_authentication(params)
    }
}

/// One pending challenge per user, keyed by the user's id, with expiry.
struct PendingChallenge {
    bytes: Vec<u8>,
    user: UserId,
    expires: Instant,
}

/// Per-user challenge state machine (generate, take, expire).
struct ChallengeMap {
    map: Mutex<HashMap<String, PendingChallenge>>,
    ttl: Duration,
}

impl ChallengeMap {
    fn new(ttl: Duration) -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            ttl,
        }
    }
    fn put(&self, user: &UserId, bytes: Vec<u8>) {
        self.map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                user.as_str().to_owned(),
                PendingChallenge {
                    bytes,
                    user: user.clone(),
                    expires: Instant::now() + self.ttl,
                },
            );
    }
    fn take(&self, user: &str) -> Option<PendingChallenge> {
        let mut map = self
            .map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match map.remove(user) {
            Some(c) if c.expires > Instant::now() => Some(c),
            other => {
                map.retain(|_, c| c.expires > Instant::now());
                other
            }
        }
    }
}

/// The passkey service: ceremonies behind the verifier seam, persistence
/// over the store seam, challenge state per user.
#[derive(Clone)]
pub struct PasskeyService {
    config: WebauthnConfig,
    verifier: Arc<dyn CeremonyVerifier>,
    store: Arc<dyn CredentialStore>,
    challenges: Arc<ChallengeMap>,
    timeout_ms: u64,
}

impl std::fmt::Debug for PasskeyService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PasskeyService")
            .field("rp_id", &self.config.rp_id)
            .field("rp_origins", &self.config.rp_origins)
            .finish_non_exhaustive()
    }
}

impl PasskeyService {
    /// Creates a service from the relying-party facts, the credential
    /// policy, and a credential store. Production uses [`KitVerifier`].
    #[must_use]
    pub fn new(
        rp_id: impl Into<String>,
        rp_name: impl Into<String>,
        rp_origins: Vec<String>,
        store: Arc<dyn CredentialStore>,
    ) -> Self {
        let rp_id = rp_id.into();
        let verifier = KitVerifier {
            rp_id: rp_id.clone(),
            rp_origins: rp_origins.clone(),
            attestation: AttestationPolicy::strict(),
            policy: CredentialPolicy::default(),
        };
        Self {
            config: WebauthnConfig {
                rp_id: rp_id.clone(),
                rp_name: rp_name.into(),
                rp_origins,
                ..WebauthnConfig::default()
            },
            verifier: Arc::new(verifier),
            store,
            challenges: Arc::new(ChallengeMap::new(Duration::from_secs(300))),
            timeout_ms: 60_000,
        }
    }

    /// Builds the `navigator.credentials.create()` options for a user and
    /// remembers the challenge. Existing passkeys for the user are excluded
    /// (an authenticator cannot re-register the same credential).
    #[must_use]
    pub fn registration_options(&self, user: &UserId, label: &str) -> RegistrationOptions {
        let bytes = generate_challenge_bytes();
        let exclude: Vec<ExcludeCredential> = self
            .store
            .for_user(user)
            .into_iter()
            .map(|c| ExcludeCredential {
                id: c.credential_id,
                type_: "public-key".into(),
                transports: None,
            })
            .collect();
        self.challenges.put(user, bytes.clone());
        RegistrationOptions {
            challenge: webauthn_kit::crypto::base64_encode_urlsafe(&bytes),
            rp: RelyingParty {
                id: self.config.rp_id.clone(),
                name: self.config.rp_name.clone(),
            },
            user: WebauthnUser {
                id: user.as_str().to_owned(),
                name: user.as_str().to_owned(),
                display_name: label.to_owned(),
            },
            pub_key_cred_params: self
                .config
                .allowed_algorithms
                .iter()
                .map(|&alg| PubKeyCredParam {
                    alg,
                    type_: "public-key".into(),
                })
                .collect(),
            timeout: self.timeout_ms,
            exclude_credentials: exclude,
            attestation: "none".into(),
            authenticator_selection: AuthenticatorSelection {
                resident_key: "required".into(),
                user_verification: "preferred".into(),
                require_resident_key: false,
            },
        }
    }

    /// Verifies a registration response and stores the credential.
    ///
    /// # Errors
    ///
    /// An expired or missing challenge, a challenge issued to a different
    /// user, or ceremony verification failures.
    pub fn finish_registration(
        &self,
        user: &UserId,
        client_data_json_b64: &str,
        attestation_object_b64: &str,
    ) -> Result<RegistrationResult> {
        let pending = self
            .challenges
            .take(user.as_str())
            .ok_or_else(|| Error::Token("no in-flight passkey challenge for this user".into()))?;
        if &pending.user != user {
            return Err(Error::Token("challenge belongs to another user".into()));
        }
        let result = self
            .verifier
            .verify_registration(
                &pending.bytes,
                client_data_json_b64,
                attestation_object_b64,
                &self.config.rp_id,
                &self.config.rp_origins,
            )
            .map_err(|e| Error::Kit(e.to_string()))?;
        self.store.put(StoredPasskey {
            user: user.clone(),
            credential_id: result.credential_id.clone(),
            public_key_cose: result.public_key_cose.clone(),
            sign_count: 0,
            label: String::new(),
        });
        Ok(result)
    }

    /// Builds the `navigator.credentials.get()` options for a user, allowing
    /// exactly that user's stored credentials, and remembers the challenge.
    #[must_use]
    pub fn authentication_options(&self, user: &UserId) -> AuthenticationOptions {
        let bytes = generate_challenge_bytes();
        let allow: Vec<AllowCredential> = self
            .store
            .for_user(user)
            .into_iter()
            .map(|c| AllowCredential {
                id: c.credential_id.clone(),
                type_: "public-key".into(),
                transports: None,
            })
            .collect();
        self.challenges.put(user, bytes.clone());
        AuthenticationOptions {
            challenge: webauthn_kit::crypto::base64_encode_urlsafe(&bytes),
            rp_id: self.config.rp_id.clone(),
            allow_credentials: allow,
            timeout: self.timeout_ms,
            user_verification: "preferred".into(),
        }
    }

    /// Verifies an authentication response against the user's stored
    /// credential. On success the caller issues session tokens via the
    /// [`SessionService`](super::SessionService) for the returned user.
    ///
    /// # Errors
    ///
    /// Ceremony failures, unknown credentials, a credential belonging to a
    /// different user, or expired/missing challenges.
    pub fn finish_authentication(
        &self,
        user: &UserId,
        credential_id: &str,
        client_data_json_b64: &str,
        authenticator_data_b64: &str,
        signature_b64: &str,
    ) -> Result<()> {
        let stored = self
            .store
            .get(credential_id)
            .ok_or_else(|| Error::Token("unknown passkey credential".into()))?;
        if &stored.user != user {
            return Err(Error::Token("credential belongs to another user".into()));
        }
        let pending = self
            .challenges
            .take(user.as_str())
            .ok_or_else(|| Error::Token("no in-flight passkey challenge for this user".into()))?;
        let params = AuthenticationParams {
            challenge_bytes: pending.bytes,
            client_data_json_b64: client_data_json_b64.to_owned(),
            authenticator_data_b64: authenticator_data_b64.to_owned(),
            signature_b64: signature_b64.to_owned(),
            credential_id_b64: credential_id.to_owned(),
            public_key_cose: stored.public_key_cose.clone(),
            current_sign_count: stored.sign_count,
            allowed_credential_ids: vec![credential_id.to_owned()],
            rp_id: self.config.rp_id.clone(),
            rp_origins: self.config.rp_origins.clone(),
            policy: self.verifier_policy(),
        };
        let _result = self
            .verifier
            .verify_authentication(&params)
            .map_err(|e| Error::Kit(e.to_string()))?;
        Ok(())
    }

    fn verifier_policy(&self) -> CredentialPolicy {
        CredentialPolicy::default()
    }
}
