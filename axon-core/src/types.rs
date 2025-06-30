//! Core type definitions for Axon protocol

use serde::{Deserialize, Serialize};
use std::fmt;

/// Content hash using SHAKE256
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    pub fn new(data: [u8; 32]) -> Self {
        Self(data)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Domain name in the .axon namespace
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DomainName(String);

impl DomainName {
    pub fn new(name: String) -> Result<Self, crate::errors::AxonError> {
        if name.len() < crate::MIN_DOMAIN_LENGTH || name.len() > crate::MAX_DOMAIN_LENGTH {
            return Err(crate::errors::AxonError::InvalidDomain(format!(
                "Domain name length must be between {} and {} characters",
                crate::MIN_DOMAIN_LENGTH,
                crate::MAX_DOMAIN_LENGTH
            )));
        }
        
        // Basic validation - could be extended with more complex rules
        if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            return Err(crate::errors::AxonError::InvalidDomain(
                "Domain name contains invalid characters".to_string()
            ));
        }
        
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn full_domain(&self) -> String {
        format!("{}.axon", self.0)
    }
}

impl fmt::Display for DomainName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.full_domain())
    }
}

/// Timestamp in Unix epoch seconds
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp(pub u64);

impl Timestamp {
    pub fn now() -> Self {
        Self(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs())
    }

    pub fn as_secs(&self) -> u64 {
        self.0
    }
}

/// Content visibility levels
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisibilityLevel {
    /// Public content visible to all
    Public,
    /// Unlisted content (accessible with direct link)
    Unlisted,
    /// Followers-only content
    Followers,
    /// Private content (only owner)
    Private,
    /// Custom access control
    Custom(Vec<String>),
}

/// Content type enumeration
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    Text,
    Image,
    Video, 
    Audio,
    Document,
    Poll,
    Repost,
    Reply,
}

/// Domain types with different pricing tiers
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainType {
    /// Standard domains (5+ characters)
    Standard,
    /// Premium domains (2-4 characters)
    Premium,
    /// Vanity domains (emoji, special chars)
    Vanity,
    /// Organization domains (verified entities)
    Organization,
    /// Community domains (multi-sig controlled)
    Community,
}