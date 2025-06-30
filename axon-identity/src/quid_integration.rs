//! QuID integration for Axon protocol

use axon_core::{
    identity::{QuIDIdentity, IdentityProof, AuthChallenge, AnonymousIdentityProof},
    crypto::{AxonSigningKey, AxonVerifyingKey},
    types::Timestamp,
    Result, AxonError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

/// QuID integration service for Axon
#[derive(Debug)]
pub struct QuIDIntegration {
    /// Registered identities
    identities: RwLock<HashMap<[u8; 32], QuIDIdentity>>,
    /// Active authentication challenges
    challenges: RwLock<HashMap<[u8; 32], AuthChallenge>>,
    /// Anonymous identity commitments
    anonymous_commitments: RwLock<HashMap<[u8; 32], AnonymousIdentityRecord>>,
    /// Service configuration
    config: QuIDConfig,
}

/// Anonymous identity record for privacy preservation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnonymousIdentityRecord {
    pub commitment: [u8; 32],
    pub capability_commitments: Vec<[u8; 32]>,
    pub created_at: Timestamp,
    pub last_used: Timestamp,
    pub usage_count: u64,
}

/// QuID service configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuIDConfig {
    /// Challenge expiration time in seconds
    pub challenge_expiration: u64,
    /// Maximum concurrent challenges per identity
    pub max_challenges_per_identity: u32,
    /// Enable anonymous identity proofs
    pub enable_anonymous_proofs: bool,
    /// Maximum identity cache size
    pub max_identity_cache: usize,
}

impl Default for QuIDConfig {
    fn default() -> Self {
        Self {
            challenge_expiration: 300, // 5 minutes
            max_challenges_per_identity: 5,
            enable_anonymous_proofs: true,
            max_identity_cache: 10000,
        }
    }
}

impl QuIDIntegration {
    /// Create new QuID integration service
    pub fn new(config: QuIDConfig) -> Self {
        Self {
            identities: RwLock::new(HashMap::new()),
            challenges: RwLock::new(HashMap::new()),
            anonymous_commitments: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Register a new QuID identity
    pub async fn register_identity(&self, identity: QuIDIdentity) -> Result<()> {
        let mut identities = self.identities.write().await;
        
        // Check if identity already exists
        if identities.contains_key(&identity.identity_hash) {
            return Err(AxonError::IdentityError(
                "Identity already registered".to_string(),
            ));
        }

        // Validate identity structure
        self.validate_identity(&identity)?;

        // Cache size management
        if identities.len() >= self.config.max_identity_cache {
            // Remove oldest identity (simple LRU simulation)
            if let Some(oldest_key) = identities.keys().next().cloned() {
                identities.remove(&oldest_key);
            }
        }

        identities.insert(identity.identity_hash, identity);
        Ok(())
    }

    /// Get identity by hash
    pub async fn get_identity(&self, identity_hash: &[u8; 32]) -> Option<QuIDIdentity> {
        let identities = self.identities.read().await;
        identities.get(identity_hash).cloned()
    }

    /// Create authentication challenge
    pub async fn create_challenge(
        &self,
        required_capabilities: Vec<axon_core::identity::IdentityCapability>,
    ) -> Result<AuthChallenge> {
        let challenge = AuthChallenge::new(required_capabilities);
        
        let mut challenges = self.challenges.write().await;
        
        // Clean up expired challenges
        self.cleanup_expired_challenges(&mut challenges).await;
        
        challenges.insert(challenge.challenge_id, challenge.clone());
        
        Ok(challenge)
    }

    /// Verify identity proof against challenge
    pub async fn verify_identity_proof(
        &self,
        proof: &IdentityProof,
        challenge_id: &[u8; 32],
    ) -> Result<bool> {
        let challenges = self.challenges.read().await;
        
        let challenge = challenges.get(challenge_id)
            .ok_or(AxonError::AuthenticationFailed)?;

        // Check if challenge is still valid
        if !challenge.is_valid() {
            return Err(AxonError::AuthenticationFailed);
        }

        // Verify the proof
        proof.verify(&challenge.challenge_data)?;

        // Check if identity has required capabilities
        for required_capability in &challenge.required_capabilities {
            if !proof.identity.has_capability(required_capability) {
                return Err(AxonError::PermissionDenied);
            }
        }

        Ok(true)
    }

    /// Create anonymous identity commitment
    pub async fn create_anonymous_commitment(
        &self,
        identity: &QuIDIdentity,
        proof: AnonymousIdentityProof,
    ) -> Result<[u8; 32]> {
        if !self.config.enable_anonymous_proofs {
            return Err(AxonError::IdentityError(
                "Anonymous proofs not enabled".to_string(),
            ));
        }

        // Verify anonymous proof (placeholder - would use actual zk-STARK verification)
        self.verify_anonymous_proof(&proof)?;

        let commitment = proof.identity_commitment;
        let record = AnonymousIdentityRecord {
            commitment,
            capability_commitments: proof.attribute_proofs.iter()
                .map(|attr| attr.commitment)
                .collect(),
            created_at: Timestamp::now(),
            last_used: Timestamp::now(),
            usage_count: 1,
        };

        let mut commitments = self.anonymous_commitments.write().await;
        commitments.insert(commitment, record);

        Ok(commitment)
    }

    /// Verify anonymous identity proof
    pub async fn verify_anonymous_commitment(
        &self,
        commitment: &[u8; 32],
        proof: &AnonymousIdentityProof,
    ) -> Result<bool> {
        let mut commitments = self.anonymous_commitments.write().await;
        
        let record = commitments.get_mut(commitment)
            .ok_or(AxonError::IdentityError("Unknown commitment".to_string()))?;

        // Verify proof commitment matches
        if &proof.identity_commitment != commitment {
            return Err(AxonError::IdentityError("Commitment mismatch".to_string()));
        }

        // Verify the anonymous proof
        self.verify_anonymous_proof(proof)?;

        // Update usage tracking
        record.last_used = Timestamp::now();
        record.usage_count += 1;

        Ok(true)
    }

    /// Get identity statistics
    pub async fn get_identity_stats(&self) -> IdentityStats {
        let identities = self.identities.read().await;
        let challenges = self.challenges.read().await;
        let commitments = self.anonymous_commitments.read().await;

        IdentityStats {
            total_identities: identities.len(),
            active_challenges: challenges.len(),
            anonymous_commitments: commitments.len(),
            identity_capabilities: self.calculate_capability_distribution(&identities).await,
        }
    }

    /// Validate identity structure
    fn validate_identity(&self, identity: &QuIDIdentity) -> Result<()> {
        // Check identity hash consistency
        let expected_hash = axon_core::crypto::hash_content(&identity.public_key.to_bytes());
        if identity.identity_hash != *expected_hash.as_bytes() {
            return Err(AxonError::IdentityError(
                "Invalid identity hash".to_string(),
            ));
        }

        // Validate metadata version
        if identity.metadata.version == 0 {
            return Err(AxonError::IdentityError(
                "Invalid metadata version".to_string(),
            ));
        }

        // Validate capabilities
        if identity.metadata.capabilities.is_empty() {
            return Err(AxonError::IdentityError(
                "Identity must have at least one capability".to_string(),
            ));
        }

        Ok(())
    }

    /// Verify anonymous proof (placeholder implementation)
    fn verify_anonymous_proof(&self, _proof: &AnonymousIdentityProof) -> Result<()> {
        // Placeholder for zk-STARK proof verification
        // In a real implementation, this would:
        // 1. Verify the zk-STARK proof of capability possession
        // 2. Verify attribute proofs
        // 3. Check nullifier for replay protection
        // 4. Verify timestamp validity
        
        Ok(())
    }

    /// Clean up expired challenges
    async fn cleanup_expired_challenges(&self, challenges: &mut HashMap<[u8; 32], AuthChallenge>) {
        let now = Timestamp::now();
        challenges.retain(|_, challenge| {
            challenge.expires_at.0 > now.0
        });
    }

    /// Calculate capability distribution
    async fn calculate_capability_distribution(
        &self,
        identities: &HashMap<[u8; 32], QuIDIdentity>,
    ) -> HashMap<String, u32> {
        let mut distribution = HashMap::new();
        
        for identity in identities.values() {
            for capability in &identity.metadata.capabilities {
                let key = format!("{:?}", capability);
                *distribution.entry(key).or_insert(0) += 1;
            }
        }
        
        distribution
    }

    /// Create identity recovery proof
    pub async fn create_recovery_proof(
        &self,
        identity_hash: &[u8; 32],
        recovery_key: &AxonSigningKey,
    ) -> Result<IdentityProof> {
        let identities = self.identities.read().await;
        let identity = identities.get(identity_hash)
            .ok_or(AxonError::IdentityError("Identity not found".to_string()))?;

        // Check if recovery key is authorized
        let recovery_verifying_key = recovery_key.verifying_key();
        if !identity.metadata.recovery_keys.contains(&recovery_verifying_key) {
            return Err(AxonError::PermissionDenied);
        }

        // Create recovery challenge
        let challenge = rand::random::<[u8; 32]>();
        let proof = IdentityProof::new(identity.clone(), challenge, recovery_key);

        Ok(proof)
    }

    /// Update identity metadata
    pub async fn update_identity(
        &self,
        identity_hash: &[u8; 32],
        updated_identity: QuIDIdentity,
        update_proof: IdentityProof,
    ) -> Result<()> {
        // Verify update authorization
        update_proof.verify(&rand::random::<[u8; 32]>())?;

        let mut identities = self.identities.write().await;
        let existing_identity = identities.get(identity_hash)
            .ok_or(AxonError::IdentityError("Identity not found".to_string()))?;

        // Verify update is authorized by identity owner or recovery key
        if update_proof.identity.public_key != existing_identity.public_key &&
           !existing_identity.metadata.recovery_keys.contains(&update_proof.identity.public_key) {
            return Err(AxonError::PermissionDenied);
        }

        // Validate updated identity
        self.validate_identity(&updated_identity)?;

        // Update identity
        identities.insert(*identity_hash, updated_identity);

        Ok(())
    }
}

/// Identity service statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentityStats {
    pub total_identities: usize,
    pub active_challenges: usize,
    pub anonymous_commitments: usize,
    pub identity_capabilities: HashMap<String, u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::crypto::AxonSigningKey;

    #[tokio::test]
    async fn test_quid_integration_creation() {
        let config = QuIDConfig::default();
        let integration = QuIDIntegration::new(config);
        
        let stats = integration.get_identity_stats().await;
        assert_eq!(stats.total_identities, 0);
        assert_eq!(stats.active_challenges, 0);
    }

    #[tokio::test]
    async fn test_identity_registration() {
        let config = QuIDConfig::default();
        let integration = QuIDIntegration::new(config);
        let signing_key = AxonSigningKey::generate();
        let identity = QuIDIdentity::new(&signing_key);

        let result = integration.register_identity(identity.clone()).await;
        assert!(result.is_ok());

        let retrieved = integration.get_identity(&identity.identity_hash).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().identity_hash, identity.identity_hash);
    }

    #[tokio::test]
    async fn test_challenge_creation_and_verification() {
        let config = QuIDConfig::default();
        let integration = QuIDIntegration::new(config);
        let signing_key = AxonSigningKey::generate();
        let identity = QuIDIdentity::new(&signing_key);

        // Register identity
        integration.register_identity(identity.clone()).await.unwrap();

        // Create challenge
        let challenge = integration.create_challenge(vec![
            axon_core::identity::IdentityCapability::SignTransactions
        ]).await.unwrap();

        // Create proof
        let proof = IdentityProof::new(identity, challenge.challenge_data, &signing_key);

        // Verify proof
        let result = integration.verify_identity_proof(&proof, &challenge.challenge_id).await;
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[tokio::test]
    async fn test_identity_stats() {
        let config = QuIDConfig::default();
        let integration = QuIDIntegration::new(config);
        let signing_key = AxonSigningKey::generate();
        let identity = QuIDIdentity::new(&signing_key);

        integration.register_identity(identity).await.unwrap();

        let stats = integration.get_identity_stats().await;
        assert_eq!(stats.total_identities, 1);
        assert!(stats.identity_capabilities.contains_key("SignTransactions"));
    }
}