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
        Self::with_verifier(rp_id, rp_name, rp_origins, store, Arc::new(verifier))
    }

    /// Creates a service with an explicit ceremony verifier. Production
    /// builds the service with [`Self::new`]; tests and alternative
    /// backends inject their own [`CeremonyVerifier`].
    #[must_use]
    pub fn with_verifier(
        rp_id: impl Into<String>,
        rp_name: impl Into<String>,
        rp_origins: Vec<String>,
        store: Arc<dyn CredentialStore>,
        verifier: Arc<dyn CeremonyVerifier>,
    ) -> Self {
        Self {
            config: WebauthnConfig {
                rp_id: rp_id.into(),
                rp_name: rp_name.into(),
                rp_origins,
                ..WebauthnConfig::default()
            },
            verifier,
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

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;
    use webauthn_kit::attestation::{AttestationFormat, AttestationResult, TrustLevel};

    /// An in-memory credential store.
    #[derive(Default)]
    struct MemStore(Mutex<Vec<StoredPasskey>>);

    impl CredentialStore for MemStore {
        fn put(&self, credential: StoredPasskey) {
            let mut all = self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            all.retain(|c| c.credential_id != credential.credential_id);
            all.push(credential);
        }
        fn get(&self, credential_id: &str) -> Option<StoredPasskey> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
                .find(|c| c.credential_id == credential_id)
                .cloned()
        }
        fn for_user(&self, user: &UserId) -> Vec<StoredPasskey> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
                .filter(|c| &c.user == user)
                .cloned()
                .collect()
        }
        fn remove(&self, credential_id: &str) {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retain(|c| c.credential_id != credential_id);
        }
    }

    /// A verifier with programmable outcomes that records the last
    /// authentication ceremony it was handed.
    #[derive(Default)]
    struct Stub {
        refuse_registration: bool,
        refuse_authentication: bool,
        seen: Mutex<Option<AuthenticationParams>>,
    }

    impl CeremonyVerifier for Stub {
        fn verify_registration(
            &self,
            _: &[u8],
            _: &str,
            _: &str,
            _: &str,
            _: &[String],
        ) -> std::result::Result<RegistrationResult, webauthn_kit::WebauthnError> {
            if self.refuse_registration {
                return Err(webauthn_kit::WebauthnError::Config(
                    "stub refused the registration".into(),
                ));
            }
            Ok(RegistrationResult {
                credential_id: "cred-1".into(),
                public_key_cose: vec![1, 2, 3],
                device_name: "test device".into(),
                attestation_format: "none".into(),
                attestation: AttestationResult {
                    format: AttestationFormat::None,
                    aaguid: None,
                    trust_level: TrustLevel::None,
                    warnings: vec![],
                },
                user_verified: true,
                backup_eligible: false,
                backup_state: false,
            })
        }

        fn verify_authentication(
            &self,
            params: &AuthenticationParams,
        ) -> std::result::Result<AuthenticationResult, webauthn_kit::WebauthnError> {
            *self
                .seen
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(params.clone());
            if self.refuse_authentication {
                return Err(webauthn_kit::WebauthnError::Config(
                    "stub refused the authentication".into(),
                ));
            }
            Ok(AuthenticationResult {
                credential_id: params.credential_id_b64.clone(),
                new_sign_count: 7,
                user_verified: true,
                backup_state: false,
                backup_eligible: false,
            })
        }
    }

    fn service(store: Arc<MemStore>, stub: Stub) -> PasskeyService {
        PasskeyService::with_verifier(
            "chisel.example",
            "Chisel Docs",
            vec!["https://chisel.example".into()],
            store,
            Arc::new(stub),
        )
    }

    fn alice() -> UserId {
        UserId::try_from("alice").unwrap()
    }

    fn bob() -> UserId {
        UserId::try_from("bob").unwrap()
    }

    fn passkey(user: &UserId, id: &str, cose: &[u8], sign_count: u32) -> StoredPasskey {
        StoredPasskey {
            user: user.clone(),
            credential_id: id.into(),
            public_key_cose: cose.to_vec(),
            sign_count,
            label: String::new(),
        }
    }

    #[test]
    fn registration_options_exclude_existing_keys_and_carry_the_relying_party() {
        let store = Arc::new(MemStore::default());
        store.put(passkey(&alice(), "old-key", &[9], 3));
        let svc = service(store, Stub::default());
        let options = svc.registration_options(&alice(), "Alice's laptop");
        assert!(!options.challenge.is_empty());
        assert_eq!(options.rp.id, "chisel.example");
        assert_eq!(options.rp.name, "Chisel Docs");
        assert_eq!(options.user.id, "alice");
        assert_eq!(options.user.display_name, "Alice's laptop");
        assert_eq!(options.exclude_credentials.len(), 1);
        assert_eq!(options.exclude_credentials[0].id, "old-key");
        assert_eq!(options.authenticator_selection.resident_key, "required");
    }

    #[test]
    fn finish_registration_stores_the_verified_credential() {
        let store = Arc::new(MemStore::default());
        let svc = service(Arc::clone(&store), Stub::default());
        let _ = svc.registration_options(&alice(), "Alice's laptop");
        let result = svc
            .finish_registration(&alice(), "client-data-b64", "attestation-b64")
            .expect("a stubbed registration finishes");
        assert_eq!(result.credential_id, "cred-1");
        let stored = store.get("cred-1").expect("the credential is persisted");
        assert_eq!(stored.user, alice());
        assert_eq!(stored.public_key_cose, vec![1, 2, 3]);
        assert_eq!(stored.sign_count, 0);
    }

    #[test]
    fn finish_registration_needs_an_in_flight_challenge() {
        let svc = service(Arc::new(MemStore::default()), Stub::default());
        let err = svc.finish_registration(&alice(), "c", "a").unwrap_err();
        assert!(
            err.to_string().contains("no in-flight passkey challenge"),
            "{err}"
        );
    }

    #[test]
    fn finish_registration_maps_verifier_failures_to_kit_errors() {
        let svc = service(
            Arc::new(MemStore::default()),
            Stub {
                refuse_registration: true,
                ..Stub::default()
            },
        );
        let _ = svc.registration_options(&alice(), "Alice's laptop");
        let err = svc.finish_registration(&alice(), "c", "a").unwrap_err();
        assert!(
            err.to_string().contains("stub refused the registration"),
            "{err}"
        );
    }

    #[test]
    fn authentication_options_allow_only_the_users_own_keys() {
        let store = Arc::new(MemStore::default());
        store.put(passkey(&alice(), "a-key", &[1], 0));
        store.put(passkey(&bob(), "b-key", &[2], 0));
        let svc = service(store, Stub::default());
        let options = svc.authentication_options(&alice());
        assert_eq!(options.allow_credentials.len(), 1);
        assert_eq!(options.allow_credentials[0].id, "a-key");
        assert_eq!(options.rp_id, "chisel.example");
        assert!(!options.challenge.is_empty());
    }

    #[test]
    fn finish_authentication_verifies_against_the_stored_key() {
        let store = Arc::new(MemStore::default());
        store.put(passkey(&alice(), "a-key", &[7, 7], 41));
        let stub = Arc::new(Stub::default());
        let svc = PasskeyService::with_verifier(
            "chisel.example",
            "Chisel Docs",
            vec!["https://chisel.example".into()],
            store,
            Arc::clone(&stub) as Arc<dyn CeremonyVerifier>,
        );
        let _ = svc.authentication_options(&alice());
        svc.finish_authentication(&alice(), "a-key", "cdj-b64", "auth-data-b64", "sig-b64")
            .expect("a stubbed authentication finishes");
        let seen = stub
            .seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ceremony = seen.as_ref().expect("the verifier saw the ceremony");
        assert_eq!(ceremony.public_key_cose, vec![7, 7]);
        assert_eq!(ceremony.current_sign_count, 41);
        assert_eq!(ceremony.rp_id, "chisel.example");
        assert_eq!(ceremony.allowed_credential_ids, ["a-key"]);
        assert_eq!(ceremony.credential_id_b64, "a-key");
    }

    #[test]
    fn finish_authentication_refuses_unknown_and_foreign_credentials() {
        let store = Arc::new(MemStore::default());
        store.put(passkey(&bob(), "b-key", &[2], 0));
        let svc = service(store, Stub::default());
        let err = svc
            .finish_authentication(&alice(), "nope", "c", "a", "s")
            .unwrap_err();
        assert!(
            err.to_string().contains("unknown passkey credential"),
            "{err}"
        );
        let err = svc
            .finish_authentication(&alice(), "b-key", "c", "a", "s")
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("credential belongs to another user"),
            "{err}"
        );
    }

    #[test]
    fn finish_authentication_needs_an_in_flight_challenge() {
        let store = Arc::new(MemStore::default());
        store.put(passkey(&alice(), "a-key", &[1], 0));
        let svc = service(store, Stub::default());
        let err = svc
            .finish_authentication(&alice(), "a-key", "c", "a", "s")
            .unwrap_err();
        assert!(
            err.to_string().contains("no in-flight passkey challenge"),
            "{err}"
        );
    }

    #[test]
    fn finish_authentication_maps_verifier_failures_to_kit_errors() {
        let store = Arc::new(MemStore::default());
        store.put(passkey(&alice(), "a-key", &[1], 0));
        let svc = service(
            store,
            Stub {
                refuse_authentication: true,
                ..Stub::default()
            },
        );
        let _ = svc.authentication_options(&alice());
        let err = svc
            .finish_authentication(&alice(), "a-key", "c", "a", "s")
            .unwrap_err();
        assert!(
            err.to_string().contains("stub refused the authentication"),
            "{err}"
        );
    }

    #[test]
    fn both_constructors_report_the_relying_party_in_debug() {
        let store = Arc::new(MemStore::default());
        let injected = service(Arc::clone(&store), Stub::default());
        assert!(format!("{injected:?}").contains("chisel.example"));
        let kit = PasskeyService::new(
            "kit.example",
            "Chisel Docs",
            vec!["https://kit.example".into()],
            store,
        );
        assert!(format!("{kit:?}").contains("kit.example"));
    }
}
