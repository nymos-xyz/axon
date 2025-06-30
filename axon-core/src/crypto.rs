//! Cryptographic utilities for Axon protocol

use crate::{ContentHash, AxonError, Result};
use ed25519_dalek::{Keypair, PublicKey, Signature, Signer, Verifier};
use sha3::{digest::{Update, ExtendableOutput, XofReader}, Shake256};
use serde::{Deserialize, Serialize};

/// Ed25519 signing key wrapper
pub struct AxonSigningKey {
    keypair: Keypair,
}

impl Clone for AxonSigningKey {
    fn clone(&self) -> Self {
        let secret_bytes = self.keypair.secret.to_bytes();
        Self::from_bytes(&secret_bytes).expect("Valid keypair should clone successfully")
    }
}

impl AxonSigningKey {
    /// Generate a new random signing key
    pub fn generate() -> Self {
        use rand::rngs::OsRng;
        let mut rng = OsRng::default();
        Self {
            keypair: Keypair::generate(&mut rng),
        }
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self> {
        let secret_key = ed25519_dalek::SecretKey::from_bytes(bytes)
            .map_err(|e| AxonError::CryptoError(format!("Invalid secret key: {}", e)))?;
        let public_key = PublicKey::from(&secret_key);
        let keypair = Keypair { secret: secret_key, public: public_key };
        Ok(Self { keypair })
    }

    /// Sign data and return signature
    pub fn sign(&self, data: &[u8]) -> AxonSignature {
        let signature = self.keypair.sign(data);
        AxonSignature { signature }
    }

    /// Get the corresponding verifying key
    pub fn verifying_key(&self) -> AxonVerifyingKey {
        AxonVerifyingKey {
            key_bytes: self.keypair.public.to_bytes(),
        }
    }

    /// Export key as bytes
    pub fn to_bytes(&self) -> [u8; 32] {
        self.keypair.secret.to_bytes()
    }
}

/// Ed25519 verifying key wrapper
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AxonVerifyingKey {
    key_bytes: [u8; 32],
}

impl PartialEq for AxonVerifyingKey {
    fn eq(&self, other: &Self) -> bool {
        self.key_bytes == other.key_bytes
    }
}

impl Eq for AxonVerifyingKey {}

impl std::hash::Hash for AxonVerifyingKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key_bytes.hash(state);
    }
}

impl AxonVerifyingKey {
    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self> {
        // Validate the key bytes by attempting to create a PublicKey
        PublicKey::from_bytes(bytes)
            .map_err(|e| AxonError::CryptoError(format!("Invalid verifying key: {}", e)))?;
        Ok(Self { key_bytes: *bytes })
    }

    /// Get the public key
    fn public_key(&self) -> Result<PublicKey> {
        PublicKey::from_bytes(&self.key_bytes)
            .map_err(|e| AxonError::CryptoError(format!("Invalid public key: {}", e)))
    }

    /// Verify a signature
    pub fn verify(&self, data: &[u8], signature: &AxonSignature) -> Result<()> {
        let public_key = self.public_key()?;
        public_key
            .verify(data, &signature.signature)
            .map_err(|_| AxonError::InvalidSignature)
    }

    /// Export key as bytes
    pub fn to_bytes(&self) -> [u8; 32] {
        self.key_bytes
    }
}

/// Ed25519 signature wrapper
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxonSignature {
    #[serde(with = "signature_serde")]
    signature: Signature,
}

// Custom serialization for Signature
mod signature_serde {
    use super::*;
    use serde::{Deserializer, Serializer, de::Error};

    pub fn serialize<S>(signature: &Signature, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&signature.to_bytes())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Signature, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        if bytes.len() != 64 {
            return Err(Error::custom("Invalid signature length"));
        }
        let mut signature_bytes = [0u8; 64];
        signature_bytes.copy_from_slice(&bytes);
        Signature::from_bytes(&signature_bytes)
            .map_err(|e| Error::custom(format!("Invalid signature: {}", e)))
    }
}

impl AxonSignature {
    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8; 64]) -> Result<Self> {
        let signature = Signature::from_bytes(bytes)
            .map_err(|e| AxonError::CryptoError(format!("Invalid signature: {}", e)))?;
        Ok(Self { signature })
    }

    /// Export signature as bytes
    pub fn to_bytes(&self) -> [u8; 64] {
        self.signature.to_bytes()
    }
}

/// Generate SHAKE256 hash for content
pub fn hash_content(data: &[u8]) -> ContentHash {
    let mut hasher = Shake256::default();
    hasher.update(data);
    let mut output = [0u8; 32];
    hasher.finalize_xof().read(&mut output);
    ContentHash::new(output)
}

/// Generate a commitment using Pedersen commitment (simplified)
pub fn pedersen_commit(value: u64, randomness: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Shake256::default();
    hasher.update(&value.to_le_bytes());
    hasher.update(randomness);
    let mut output = [0u8; 32];
    hasher.finalize_xof().read(&mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let signing_key = AxonSigningKey::generate();
        let verifying_key = signing_key.verifying_key();
        
        let data = b"test message";
        let signature = signing_key.sign(data);
        
        assert!(verifying_key.verify(data, &signature).is_ok());
    }

    #[test]
    fn test_key_cloning() {
        let signing_key = AxonSigningKey::generate();
        let cloned_key = signing_key.clone();
        
        let data = b"test message";
        let signature1 = signing_key.sign(data);
        let signature2 = cloned_key.sign(data);
        
        let verifying_key = signing_key.verifying_key();
        assert!(verifying_key.verify(data, &signature1).is_ok());
        assert!(verifying_key.verify(data, &signature2).is_ok());
    }

    #[test]
    fn test_key_serialization() {
        let signing_key = AxonSigningKey::generate();
        let verifying_key = signing_key.verifying_key();
        
        // Test serialization/deserialization
        let serialized = serde_json::to_string(&verifying_key).unwrap();
        let deserialized: AxonVerifyingKey = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(verifying_key, deserialized);
    }

    #[test]
    fn test_signature_serialization() {
        let signing_key = AxonSigningKey::generate();
        let data = b"test message";
        let signature = signing_key.sign(data);
        
        let serialized = serde_json::to_string(&signature).unwrap();
        let deserialized: AxonSignature = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(signature, deserialized);
    }

    #[test]
    fn test_content_hash() {
        let data = b"test content";
        let hash1 = hash_content(data);
        let hash2 = hash_content(data);
        
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.as_bytes().len(), 32);
    }

    #[test]
    fn test_content_hash_different_data() {
        let data1 = b"test content 1";
        let data2 = b"test content 2";
        let hash1 = hash_content(data1);
        let hash2 = hash_content(data2);
        
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_key_round_trip() {
        let signing_key = AxonSigningKey::generate();
        let bytes = signing_key.to_bytes();
        let restored_key = AxonSigningKey::from_bytes(&bytes).unwrap();
        
        let data = b"test data";
        let sig1 = signing_key.sign(data);
        let sig2 = restored_key.sign(data);
        
        let verifying_key = signing_key.verifying_key();
        assert!(verifying_key.verify(data, &sig1).is_ok());
        assert!(verifying_key.verify(data, &sig2).is_ok());
    }

    #[test]
    fn test_signature_verification_failure() {
        let signing_key1 = AxonSigningKey::generate();
        let signing_key2 = AxonSigningKey::generate();
        
        let data = b"test message";
        let signature = signing_key1.sign(data);
        let wrong_verifying_key = signing_key2.verifying_key();
        
        assert!(wrong_verifying_key.verify(data, &signature).is_err());
    }

    #[test]
    fn test_pedersen_commit() {
        let value = 12345u64;
        let randomness = [42u8; 32];
        
        let commitment1 = pedersen_commit(value, &randomness);
        let commitment2 = pedersen_commit(value, &randomness);
        
        assert_eq!(commitment1, commitment2);
        
        // Different randomness should produce different commitment
        let different_randomness = [43u8; 32];
        let commitment3 = pedersen_commit(value, &different_randomness);
        assert_ne!(commitment1, commitment3);
    }
}