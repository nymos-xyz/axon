//! Domain management structures for Axon protocol

use crate::{
    crypto::{AxonSignature, AxonVerifyingKey},
    types::{ContentHash, DomainName, DomainType, Timestamp},
    AxonError, Result,
};
use serde::{Deserialize, Serialize};

/// Domain registration record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainRecord {
    pub domain_name: DomainName,
    pub owner: AxonVerifyingKey,
    pub content_root_hash: ContentHash,
    pub registered_at: Timestamp,
    pub expires_at: Timestamp,
    pub domain_type: DomainType,
    pub auto_renewal: Option<AutoRenewalConfig>,
    pub transfer_rules: TransferPolicy,
    pub metadata: DomainMetadata,
    pub signature: AxonSignature,
}

/// Auto-renewal configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoRenewalConfig {
    pub enabled: bool,
    pub renewal_duration_years: u32,
    pub max_renewal_price: u64, // Maximum price willing to pay for renewal
    pub funding_source: FundingSource,
}

/// Funding source for auto-renewal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FundingSource {
    /// Use tokens from specific account
    Account(AxonVerifyingKey),
    /// Use earnings from domain
    DomainEarnings,
    /// Use escrow account
    Escrow,
}

/// Transfer policy for domains
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferPolicy {
    pub transferable: bool,
    pub requires_approval: bool,
    pub authorized_transferees: Vec<AxonVerifyingKey>,
    pub transfer_cooldown_days: u32,
}

/// Domain metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainMetadata {
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub contact_info: Option<String>,
    pub category: Option<String>,
    pub reputation_score: u32,
    pub verification_status: VerificationStatus,
}

/// Domain verification status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VerificationStatus {
    Unverified,
    Pending,
    Verified,
    Organization,
    Community,
}

/// Domain registration request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainRegistrationRequest {
    pub domain_name: DomainName,
    pub domain_type: DomainType,
    pub duration_years: u32,
    pub owner: AxonVerifyingKey,
    pub initial_content_hash: ContentHash,
    pub auto_renewal: Option<AutoRenewalConfig>,
    pub metadata: DomainMetadata,
    pub payment_proof: Vec<u8>, // Payment proof for registration
    pub signature: AxonSignature,
}

/// Domain pricing structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainPricing {
    pub base_prices: DomainTypePricing,
    pub network_health_multiplier: f32,
    pub demand_multiplier: f32,
    pub updated_at: Timestamp,
}

/// Pricing for different domain types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainTypePricing {
    pub standard: u64,      // Price per year in smallest token unit
    pub premium: u64,       // 2-4 character domains
    pub vanity: u64,        // Emoji/special character domains
    pub organization: u64,  // Verified organization domains
    pub community: u64,     // Multi-sig community domains
}

/// Domain search and discovery
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainSearchResult {
    pub domain_name: DomainName,
    pub domain_type: DomainType,
    pub verification_status: VerificationStatus,
    pub registration_date: Timestamp,
    pub reputation_score: u32,
    pub preview_content: Option<String>,
}

impl DomainRecord {
    /// Check if domain is expired
    pub fn is_expired(&self) -> bool {
        Timestamp::now().0 > self.expires_at.0
    }

    /// Check if domain expires within given days
    pub fn expires_within_days(&self, days: u32) -> bool {
        let seconds_in_day = 24 * 60 * 60;
        let expiry_threshold = Timestamp::now().0 + (days as u64 * seconds_in_day);
        self.expires_at.0 <= expiry_threshold
    }

    /// Verify domain record signature
    pub fn verify_signature(&self) -> Result<()> {
        let content_bytes = bincode::serialize(&(
            &self.domain_name,
            &self.owner,
            &self.content_root_hash,
            &self.registered_at,
            &self.expires_at,
            &self.domain_type,
            &self.auto_renewal,
            &self.transfer_rules,
            &self.metadata,
        ))
        .map_err(|e| AxonError::SerializationError(e.to_string()))?;
        
        self.owner.verify(&content_bytes, &self.signature)
    }

    /// Calculate renewal cost based on current pricing
    pub fn calculate_renewal_cost(&self, pricing: &DomainPricing, years: u32) -> u64 {
        let base_price = match self.domain_type {
            DomainType::Standard => pricing.base_prices.standard,
            DomainType::Premium => pricing.base_prices.premium,
            DomainType::Vanity => pricing.base_prices.vanity,
            DomainType::Organization => pricing.base_prices.organization,
            DomainType::Community => pricing.base_prices.community,
        };
        
        let adjusted_price = (base_price as f32 * pricing.network_health_multiplier * pricing.demand_multiplier) as u64;
        adjusted_price * years as u64
    }
}

impl DomainRegistrationRequest {
    /// Verify registration request signature
    pub fn verify_signature(&self) -> Result<()> {
        let content_bytes = bincode::serialize(&(
            &self.domain_name,
            &self.domain_type,
            &self.duration_years,
            &self.owner,
            &self.initial_content_hash,
            &self.auto_renewal,
            &self.metadata,
            &self.payment_proof,
        ))
        .map_err(|e| AxonError::SerializationError(e.to_string()))?;
        
        self.owner.verify(&content_bytes, &self.signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{AxonSigningKey, hash_content};

    #[test]
    fn test_domain_registration() {
        let signing_key = AxonSigningKey::generate();
        let owner = signing_key.verifying_key();
        let domain_name = DomainName::new("testdomain".to_string()).unwrap();
        
        let request = DomainRegistrationRequest {
            domain_name: domain_name.clone(),
            domain_type: DomainType::Standard,
            duration_years: 1,
            owner: owner.clone(),
            initial_content_hash: hash_content(b"initial content"),
            auto_renewal: None,
            metadata: DomainMetadata {
                description: Some("Test domain".to_string()),
                keywords: vec!["test".to_string()],
                contact_info: None,
                category: None,
                reputation_score: 0,
                verification_status: VerificationStatus::Unverified,
            },
            payment_proof: vec![],
            signature: signing_key.sign(&[]), // Placeholder
        };
        
        // In real implementation, signature would be calculated properly
        assert_eq!(request.domain_name, domain_name);
        assert_eq!(request.owner, owner);
    }

    #[test]
    fn test_domain_expiry_check() {
        let signing_key = AxonSigningKey::generate();
        let owner = signing_key.verifying_key();
        
        let past_time = Timestamp(Timestamp::now().0 - 3600); // 1 hour ago
        let future_time = Timestamp(Timestamp::now().0 + 3600); // 1 hour from now
        
        let expired_domain = DomainRecord {
            domain_name: DomainName::new("expired".to_string()).unwrap(),
            owner: owner.clone(),
            content_root_hash: hash_content(b"content"),
            registered_at: Timestamp::now(),
            expires_at: past_time,
            domain_type: DomainType::Standard,
            auto_renewal: None,
            transfer_rules: TransferPolicy {
                transferable: true,
                requires_approval: false,
                authorized_transferees: vec![],
                transfer_cooldown_days: 0,
            },
            metadata: DomainMetadata {
                description: None,
                keywords: vec![],
                contact_info: None,
                category: None,
                reputation_score: 0,
                verification_status: VerificationStatus::Unverified,
            },
            signature: signing_key.sign(&[]), // Placeholder
        };
        
        assert!(expired_domain.is_expired());
        
        let valid_domain = DomainRecord {
            expires_at: future_time,
            ..expired_domain
        };
        
        assert!(!valid_domain.is_expired());
    }
}