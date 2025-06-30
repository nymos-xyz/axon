//! Authentication service for Axon protocol

use axon_core::{
    identity::{QuIDIdentity, IdentityProof, AuthChallenge},
    crypto::AxonVerifyingKey,
    types::Timestamp,
    Result, AxonError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Authentication service for managing user sessions
#[derive(Debug)]
pub struct AuthenticationService {
    /// Active authentication sessions
    sessions: RwLock<HashMap<SessionId, AuthSession>>,
    /// Challenge cache
    challenges: RwLock<HashMap<[u8; 32], AuthChallenge>>,
    /// Service configuration
    config: AuthConfig,
}

/// Session identifier
pub type SessionId = [u8; 32];

/// Authentication session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthSession {
    pub session_id: SessionId,
    pub identity: QuIDIdentity,
    pub created_at: Timestamp,
    pub expires_at: Timestamp,
    pub last_activity: Timestamp,
    pub capabilities: Vec<axon_core::identity::IdentityCapability>,
    pub domain_access: Vec<String>,
    pub anonymous: bool,
}

/// Authentication configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Session duration in seconds
    pub session_duration: u64,
    /// Challenge timeout in seconds
    pub challenge_timeout: u64,
    /// Maximum concurrent sessions per identity
    pub max_sessions_per_identity: u32,
    /// Enable anonymous sessions
    pub enable_anonymous_sessions: bool,
    /// Session cleanup interval
    pub cleanup_interval: u64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            session_duration: 86400, // 24 hours
            challenge_timeout: 300,   // 5 minutes
            max_sessions_per_identity: 10,
            enable_anonymous_sessions: true,
            cleanup_interval: 3600,   // 1 hour
        }
    }
}

/// Authentication result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthResult {
    pub session_id: SessionId,
    pub identity_hash: [u8; 32],
    pub capabilities: Vec<axon_core::identity::IdentityCapability>,
    pub expires_at: Timestamp,
    pub anonymous: bool,
}

/// Session verification result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: SessionId,
    pub identity: QuIDIdentity,
    pub capabilities: Vec<axon_core::identity::IdentityCapability>,
    pub domain_access: Vec<String>,
    pub time_remaining: u64,
    pub anonymous: bool,
}

impl AuthenticationService {
    /// Create new authentication service
    pub fn new(config: AuthConfig) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            challenges: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Create authentication challenge
    pub async fn create_auth_challenge(
        &self,
        required_capabilities: Vec<axon_core::identity::IdentityCapability>,
    ) -> Result<AuthChallenge> {
        let challenge = AuthChallenge::new(required_capabilities);
        
        let mut challenges = self.challenges.write().await;
        challenges.insert(challenge.challenge_id, challenge.clone());

        Ok(challenge)
    }

    /// Authenticate user with identity proof
    pub async fn authenticate(
        &self,
        proof: IdentityProof,
        challenge_id: [u8; 32],
        anonymous: bool,
    ) -> Result<AuthResult> {
        // Verify challenge exists and is valid
        let challenge = {
            let challenges = self.challenges.read().await;
            let challenge = challenges.get(&challenge_id)
                .ok_or(AxonError::AuthenticationFailed)?;
            
            if !challenge.is_valid() {
                return Err(AxonError::AuthenticationFailed);
            }
            
            challenge.clone()
        };

        // Verify identity proof
        proof.verify(&challenge.challenge_data)?;

        // Check required capabilities
        for required_capability in &challenge.required_capabilities {
            if !proof.identity.has_capability(required_capability) {
                return Err(AxonError::PermissionDenied);
            }
        }

        // Create session
        let session_id = self.generate_session_id();
        let now = Timestamp::now();
        let expires_at = Timestamp(now.0 + self.config.session_duration);

        let session = AuthSession {
            session_id,
            identity: proof.identity.clone(),
            created_at: now,
            expires_at,
            last_activity: now,
            capabilities: proof.identity.metadata.capabilities.clone(),
            domain_access: vec![], // Would be populated based on identity permissions
            anonymous,
        };

        // Check session limits
        if !anonymous {
            self.enforce_session_limits(&proof.identity.identity_hash).await?;
        }

        // Store session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session_id, session);
        }

        // Clean up used challenge
        {
            let mut challenges = self.challenges.write().await;
            challenges.remove(&challenge_id);
        }

        Ok(AuthResult {
            session_id,
            identity_hash: proof.identity.identity_hash,
            capabilities: proof.identity.metadata.capabilities,
            expires_at,
            anonymous,
        })
    }

    /// Verify session and get session info
    pub async fn verify_session(&self, session_id: SessionId) -> Result<SessionInfo> {
        let mut sessions = self.sessions.write().await;
        
        let session = sessions.get_mut(&session_id)
            .ok_or(AxonError::AuthenticationFailed)?;

        let now = Timestamp::now();
        
        // Check if session is expired
        if now.0 >= session.expires_at.0 {
            sessions.remove(&session_id);
            return Err(AxonError::AuthenticationFailed);
        }

        // Update last activity
        session.last_activity = now;
        
        let time_remaining = session.expires_at.0 - now.0;

        Ok(SessionInfo {
            session_id,
            identity: session.identity.clone(),
            capabilities: session.capabilities.clone(),
            domain_access: session.domain_access.clone(),
            time_remaining,
            anonymous: session.anonymous,
        })
    }

    /// Refresh session expiration
    pub async fn refresh_session(&self, session_id: SessionId) -> Result<Timestamp> {
        let mut sessions = self.sessions.write().await;
        
        let session = sessions.get_mut(&session_id)
            .ok_or(AxonError::AuthenticationFailed)?;

        let now = Timestamp::now();
        
        // Check if session is still valid
        if now.0 >= session.expires_at.0 {
            sessions.remove(&session_id);
            return Err(AxonError::AuthenticationFailed);
        }

        // Extend session
        session.expires_at = Timestamp(now.0 + self.config.session_duration);
        session.last_activity = now;

        Ok(session.expires_at)
    }

    /// End session (logout)
    pub async fn end_session(&self, session_id: SessionId) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        
        if sessions.remove(&session_id).is_some() {
            Ok(())
        } else {
            Err(AxonError::AuthenticationFailed)
        }
    }

    /// Grant domain access to session
    pub async fn grant_domain_access(
        &self,
        session_id: SessionId,
        domain: String,
    ) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        
        let session = sessions.get_mut(&session_id)
            .ok_or(AxonError::AuthenticationFailed)?;

        if !session.domain_access.contains(&domain) {
            session.domain_access.push(domain);
        }

        Ok(())
    }

    /// Check if session has domain access
    pub async fn has_domain_access(
        &self,
        session_id: SessionId,
        domain: &str,
    ) -> Result<bool> {
        let sessions = self.sessions.read().await;
        
        let session = sessions.get(&session_id)
            .ok_or(AxonError::AuthenticationFailed)?;

        // Check if session is still valid
        let now = Timestamp::now();
        if now.0 >= session.expires_at.0 {
            return Err(AxonError::AuthenticationFailed);
        }

        Ok(session.domain_access.contains(&domain.to_string()))
    }

    /// Get all active sessions for an identity
    pub async fn get_identity_sessions(
        &self,
        identity_hash: &[u8; 32],
    ) -> Vec<SessionInfo> {
        let sessions = self.sessions.read().await;
        let now = Timestamp::now();

        sessions.values()
            .filter(|session| {
                &session.identity.identity_hash == identity_hash &&
                now.0 < session.expires_at.0
            })
            .map(|session| SessionInfo {
                session_id: session.session_id,
                identity: session.identity.clone(),
                capabilities: session.capabilities.clone(),
                domain_access: session.domain_access.clone(),
                time_remaining: session.expires_at.0 - now.0,
                anonymous: session.anonymous,
            })
            .collect()
    }

    /// Clean up expired sessions and challenges
    pub async fn cleanup_expired(&self) -> Result<(u32, u32)> {
        let now = Timestamp::now();
        let mut removed_sessions = 0;
        let mut removed_challenges = 0;

        // Clean up expired sessions
        {
            let mut sessions = self.sessions.write().await;
            let initial_count = sessions.len();
            sessions.retain(|_, session| session.expires_at.0 > now.0);
            removed_sessions = (initial_count - sessions.len()) as u32;
        }

        // Clean up expired challenges
        {
            let mut challenges = self.challenges.write().await;
            let initial_count = challenges.len();
            challenges.retain(|_, challenge| challenge.expires_at.0 > now.0);
            removed_challenges = (initial_count - challenges.len()) as u32;
        }

        Ok((removed_sessions, removed_challenges))
    }

    /// Get authentication service statistics
    pub async fn get_auth_stats(&self) -> AuthStats {
        let sessions = self.sessions.read().await;
        let challenges = self.challenges.read().await;
        let now = Timestamp::now();

        let active_sessions = sessions.values()
            .filter(|session| session.expires_at.0 > now.0)
            .count();

        let anonymous_sessions = sessions.values()
            .filter(|session| session.anonymous && session.expires_at.0 > now.0)
            .count();

        AuthStats {
            total_sessions: sessions.len(),
            active_sessions,
            anonymous_sessions,
            active_challenges: challenges.len(),
            avg_session_duration: self.calculate_avg_session_duration(&sessions).await,
        }
    }

    /// Generate unique session ID
    fn generate_session_id(&self) -> SessionId {
        rand::random::<[u8; 32]>()
    }

    /// Enforce session limits per identity
    async fn enforce_session_limits(&self, identity_hash: &[u8; 32]) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        let now = Timestamp::now();

        // Count active sessions for this identity
        let identity_sessions: Vec<_> = sessions.iter()
            .filter(|(_, session)| {
                &session.identity.identity_hash == identity_hash &&
                session.expires_at.0 > now.0 &&
                !session.anonymous
            })
            .map(|(id, _)| *id)
            .collect();

        // Remove oldest sessions if limit exceeded
        if identity_sessions.len() >= self.config.max_sessions_per_identity as usize {
            let sessions_to_remove = identity_sessions.len() - (self.config.max_sessions_per_identity as usize - 1);
            
            // Sort by creation time and remove oldest
            let mut sorted_sessions: Vec<_> = identity_sessions.iter()
                .map(|id| (*id, sessions.get(id).unwrap().created_at))
                .collect();
            
            sorted_sessions.sort_by_key(|(_, created_at)| *created_at);
            
            for (session_id, _) in sorted_sessions.iter().take(sessions_to_remove) {
                sessions.remove(session_id);
            }
        }

        Ok(())
    }

    /// Calculate average session duration
    async fn calculate_avg_session_duration(&self, sessions: &HashMap<SessionId, AuthSession>) -> u64 {
        if sessions.is_empty() {
            return 0;
        }

        let now = Timestamp::now();
        let total_duration: u64 = sessions.values()
            .map(|session| {
                if session.expires_at.0 > now.0 {
                    now.0 - session.created_at.0
                } else {
                    session.expires_at.0 - session.created_at.0
                }
            })
            .sum();

        total_duration / sessions.len() as u64
    }
}

/// Authentication service statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthStats {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub anonymous_sessions: usize,
    pub active_challenges: usize,
    pub avg_session_duration: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{crypto::AxonSigningKey, identity::QuIDIdentity};

    #[tokio::test]
    async fn test_auth_service_creation() {
        let config = AuthConfig::default();
        let auth_service = AuthenticationService::new(config);
        
        let stats = auth_service.get_auth_stats().await;
        assert_eq!(stats.total_sessions, 0);
        assert_eq!(stats.active_challenges, 0);
    }

    #[tokio::test]
    async fn test_authentication_flow() {
        let config = AuthConfig::default();
        let auth_service = AuthenticationService::new(config);
        let signing_key = AxonSigningKey::generate();
        let identity = QuIDIdentity::new(&signing_key);

        // Create challenge
        let challenge = auth_service.create_auth_challenge(vec![
            axon_core::identity::IdentityCapability::SignTransactions
        ]).await.unwrap();

        // Create proof
        let proof = IdentityProof::new(identity, challenge.challenge_data, &signing_key);

        // Authenticate
        let auth_result = auth_service.authenticate(proof, challenge.challenge_id, false).await;
        assert!(auth_result.is_ok());

        let result = auth_result.unwrap();
        assert!(!result.anonymous);

        // Verify session
        let session_info = auth_service.verify_session(result.session_id).await;
        assert!(session_info.is_ok());
    }

    #[tokio::test]
    async fn test_session_refresh() {
        let config = AuthConfig::default();
        let auth_service = AuthenticationService::new(config);
        let signing_key = AxonSigningKey::generate();
        let identity = QuIDIdentity::new(&signing_key);

        // Authenticate to get session
        let challenge = auth_service.create_auth_challenge(vec![]).await.unwrap();
        let proof = IdentityProof::new(identity, challenge.challenge_data, &signing_key);
        let auth_result = auth_service.authenticate(proof, challenge.challenge_id, false).await.unwrap();

        // Refresh session
        let new_expiry = auth_service.refresh_session(auth_result.session_id).await;
        assert!(new_expiry.is_ok());
        assert!(new_expiry.unwrap().0 > auth_result.expires_at.0);
    }

    #[tokio::test]
    async fn test_domain_access_control() {
        let config = AuthConfig::default();
        let auth_service = AuthenticationService::new(config);
        let signing_key = AxonSigningKey::generate();
        let identity = QuIDIdentity::new(&signing_key);

        // Authenticate
        let challenge = auth_service.create_auth_challenge(vec![]).await.unwrap();
        let proof = IdentityProof::new(identity, challenge.challenge_data, &signing_key);
        let auth_result = auth_service.authenticate(proof, challenge.challenge_id, false).await.unwrap();

        // Test domain access
        let domain = "testdomain.axon".to_string();
        
        let has_access = auth_service.has_domain_access(auth_result.session_id, &domain).await.unwrap();
        assert!(!has_access);

        auth_service.grant_domain_access(auth_result.session_id, domain.clone()).await.unwrap();
        
        let has_access = auth_service.has_domain_access(auth_result.session_id, &domain).await.unwrap();
        assert!(has_access);
    }
}