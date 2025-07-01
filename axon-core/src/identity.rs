//! Identity management for Axon protocol (QuID integration)

use crate::{
    crypto::{AxonSignature, AxonSigningKey, AxonVerifyingKey},
    types::Timestamp,
    AxonError, Result,
};
use serde::{Deserialize, Serialize};

/// QuID identity representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuIDIdentity {
    pub public_key: AxonVerifyingKey,
    pub identity_hash: [u8; 32],
    pub created_at: Timestamp,
    pub metadata: IdentityMetadata,
}

impl PartialEq for QuIDIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.public_key == other.public_key &&
        self.identity_hash == other.identity_hash &&
        self.created_at == other.created_at &&
        self.metadata == other.metadata
    }
}

impl Eq for QuIDIdentity {}

/// Identity metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentityMetadata {
    pub version: u32,
    pub recovery_keys: Vec<AxonVerifyingKey>,
    pub capabilities: Vec<IdentityCapability>,
    pub attributes: Vec<IdentityAttribute>,
}

impl PartialEq for IdentityMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version &&
        self.recovery_keys.len() == other.recovery_keys.len() &&
        self.recovery_keys.iter().zip(&other.recovery_keys).all(|(a, b)| a == b) &&
        self.capabilities == other.capabilities &&
        self.attributes == other.attributes
    }
}

impl Eq for IdentityMetadata {}

/// Identity capabilities
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityCapability {
    /// Can sign transactions
    SignTransactions,
    /// Can register domains
    RegisterDomains,
    /// Can create content
    CreateContent,
    /// Can participate in governance
    Governance,
    /// Can operate nodes
    NodeOperation,
}

/// Identity attributes (can be selectively revealed)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityAttribute {
    pub key: String,
    pub value_commitment: [u8; 32], // Committed value for privacy
    pub revealed: bool,
    pub revealed_value: Option<String>,
}

/// Identity proof for authentication
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentityProof {
    pub identity: QuIDIdentity,
    pub challenge: [u8; 32],
    pub signature: AxonSignature,
    pub timestamp: Timestamp,
    pub nonce: u64,
}

/// Anonymous identity proof (zero-knowledge)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnonymousIdentityProof {
    pub identity_commitment: [u8; 32],
    pub capability_proof: Vec<u8>, // zk-STARK proof of capabilities
    pub attribute_proofs: Vec<AttributeProof>,
    pub nullifier: [u8; 32], // Prevents replay attacks
    pub timestamp: Timestamp,
}

/// Proof of identity attribute without revealing value
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttributeProof {
    pub attribute_key: String,
    pub proof: Vec<u8>, // zk-STARK proof
    pub commitment: [u8; 32],
}

/// Identity authentication challenge
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthChallenge {
    pub challenge_id: [u8; 32],
    pub challenge_data: [u8; 32],
    pub required_capabilities: Vec<IdentityCapability>,
    pub expires_at: Timestamp,
    pub created_at: Timestamp,
}

/// Multi-signature identity for communities/organizations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiSigIdentity {
    pub identity_hash: [u8; 32],
    pub required_signatures: u32,
    pub signers: Vec<AxonVerifyingKey>,
    pub created_at: Timestamp,
    pub metadata: IdentityMetadata,
}

/// Multi-signature transaction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiSigTransaction {
    pub transaction_data: Vec<u8>,
    pub signatures: Vec<(AxonVerifyingKey, AxonSignature)>,
    pub threshold_met: bool,
    pub created_at: Timestamp,
}

impl QuIDIdentity {
    /// Create a new identity from signing key
    pub fn new(signing_key: &AxonSigningKey) -> Self {
        let public_key = signing_key.verifying_key();
        let identity_hash = crate::crypto::hash_content(&public_key.to_bytes()).as_bytes().clone();
        
        Self {
            public_key,
            identity_hash,
            created_at: Timestamp::now(),
            metadata: IdentityMetadata {
                version: 1,
                recovery_keys: vec![],
                capabilities: vec![
                    IdentityCapability::SignTransactions,
                    IdentityCapability::CreateContent,
                ],
                attributes: vec![],
            },
        }
    }

    /// Add a capability to the identity
    pub fn add_capability(&mut self, capability: IdentityCapability) {
        if !self.metadata.capabilities.contains(&capability) {
            self.metadata.capabilities.push(capability);
        }
    }

    /// Check if identity has a specific capability
    pub fn has_capability(&self, capability: &IdentityCapability) -> bool {
        self.metadata.capabilities.contains(capability)
    }

    /// Add a recovery key
    pub fn add_recovery_key(&mut self, recovery_key: AxonVerifyingKey) {
        if !self.metadata.recovery_keys.contains(&recovery_key) {
            self.metadata.recovery_keys.push(recovery_key);
        }
    }

    /// Add an attribute with commitment
    pub fn add_attribute(&mut self, key: String, value: String, revealed: bool) {
        let commitment = crate::crypto::pedersen_commit(
            value.len() as u64,
            &crate::crypto::hash_content(value.as_bytes()).as_bytes().clone(),
        );
        
        let attribute = IdentityAttribute {
            key,
            value_commitment: commitment,
            revealed,
            revealed_value: if revealed { Some(value) } else { None },
        };
        
        self.metadata.attributes.push(attribute);
    }

    /// Get the identity ID as a string (hex-encoded identity hash)
    pub fn get_id(&self) -> String {
        hex::encode(self.identity_hash)
    }

    /// Create a test identity for unit tests
    #[cfg(test)]
    pub fn new_for_test(name: &str) -> Self {
        // Create a deterministic signing key for testing
        let mut key_bytes = [0u8; 32];
        let name_bytes = name.as_bytes();
        let copy_len = std::cmp::min(name_bytes.len(), 32);
        key_bytes[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
        
        let signing_key = AxonSigningKey::from_bytes(&key_bytes).expect("Valid test key");
        Self::new(&signing_key)
    }
}

impl IdentityProof {
    /// Create a new identity proof
    pub fn new(
        identity: QuIDIdentity,
        challenge: [u8; 32],
        signing_key: &AxonSigningKey,
    ) -> Self {
        let timestamp = Timestamp::now();
        let nonce = rand::random::<u64>();
        
        let proof_data = bincode::serialize(&(&identity, &challenge, &timestamp, &nonce))
            .expect("Serialization should not fail");
        
        let signature = signing_key.sign(&proof_data);
        
        Self {
            identity,
            challenge,
            signature,
            timestamp,
            nonce,
        }
    }

    /// Verify the identity proof
    pub fn verify(&self, expected_challenge: &[u8; 32]) -> Result<()> {
        if &self.challenge != expected_challenge {
            return Err(AxonError::AuthenticationFailed);
        }

        // Check timestamp (should be recent)
        let now = Timestamp::now();
        if now.0 - self.timestamp.0 > 300 { // 5 minutes
            return Err(AxonError::AuthenticationFailed);
        }

        let proof_data = bincode::serialize(&(&self.identity, &self.challenge, &self.timestamp, &self.nonce))
            .map_err(|e| AxonError::SerializationError(e.to_string()))?;

        self.identity.public_key.verify(&proof_data, &self.signature)
    }
}

impl AuthChallenge {
    /// Generate a new authentication challenge
    pub fn new(required_capabilities: Vec<IdentityCapability>) -> Self {
        let challenge_id = rand::random::<[u8; 32]>();
        let challenge_data = rand::random::<[u8; 32]>();
        let created_at = Timestamp::now();
        let expires_at = Timestamp(created_at.0 + 300); // 5 minutes

        Self {
            challenge_id,
            challenge_data,
            required_capabilities,
            expires_at,
            created_at,
        }
    }

    /// Check if challenge is still valid
    pub fn is_valid(&self) -> bool {
        Timestamp::now().0 < self.expires_at.0
    }
}

impl MultiSigIdentity {
    /// Create a new multi-signature identity
    pub fn new(signers: Vec<AxonVerifyingKey>, required_signatures: u32) -> Result<Self> {
        if required_signatures == 0 || required_signatures > signers.len() as u32 {
            return Err(AxonError::IdentityError(
                "Invalid signature threshold".to_string(),
            ));
        }

        let identity_data = bincode::serialize(&(&signers, &required_signatures))
            .map_err(|e| AxonError::SerializationError(e.to_string()))?;
        
        let identity_hash = crate::crypto::hash_content(&identity_data).as_bytes().clone();

        Ok(Self {
            identity_hash,
            required_signatures,
            signers,
            created_at: Timestamp::now(),
            metadata: IdentityMetadata {
                version: 1,
                recovery_keys: vec![],
                capabilities: vec![IdentityCapability::SignTransactions],
                attributes: vec![],
            },
        })
    }

    /// Check if a transaction has enough signatures
    pub fn verify_transaction(&self, transaction: &MultiSigTransaction) -> Result<()> {
        if transaction.signatures.len() < self.required_signatures as usize {
            return Err(AxonError::AuthenticationFailed);
        }

        let mut valid_signatures = 0;
        for (signer, signature) in &transaction.signatures {
            if self.signers.contains(signer) {
                if signer.verify(&transaction.transaction_data, signature).is_ok() {
                    valid_signatures += 1;
                }
            }
        }

        if valid_signatures >= self.required_signatures {
            Ok(())
        } else {
            Err(AxonError::AuthenticationFailed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_creation() {
        let signing_key = AxonSigningKey::generate();
        let identity = QuIDIdentity::new(&signing_key);
        
        assert_eq!(identity.public_key, signing_key.verifying_key());
        assert!(identity.has_capability(&IdentityCapability::SignTransactions));
        assert!(identity.has_capability(&IdentityCapability::CreateContent));
    }

    #[test]
    fn test_identity_proof() {
        let signing_key = AxonSigningKey::generate();
        let identity = QuIDIdentity::new(&signing_key);
        let challenge = rand::random::<[u8; 32]>();
        
        let proof = IdentityProof::new(identity, challenge, &signing_key);
        assert!(proof.verify(&challenge).is_ok());
        
        let wrong_challenge = rand::random::<[u8; 32]>();
        assert!(proof.verify(&wrong_challenge).is_err());
    }

    #[test]
    fn test_multisig_identity() {
        let key1 = AxonSigningKey::generate();
        let key2 = AxonSigningKey::generate();
        let key3 = AxonSigningKey::generate();
        
        let signers = vec![
            key1.verifying_key(),
            key2.verifying_key(),
            key3.verifying_key(),
        ];
        
        let multisig = MultiSigIdentity::new(signers, 2).unwrap();
        assert_eq!(multisig.required_signatures, 2);
        assert_eq!(multisig.signers.len(), 3);
    }
}