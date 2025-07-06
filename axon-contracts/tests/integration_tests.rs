//! Integration tests for Axon smart contracts
//! Testing the complete domain registry system including .quid and .axon domains

use axon_contracts::quid_axon_registry::{
    QuidAxonRegistry, NymverseDomainConfig, QuidDomainPricing, AxonDomainPricing,
    RevenueDistribution, GovernanceSettings, NymverseDomainType, ProposalType,
    ProposalChanges, VoteType,
};
use axon_core::{
    domain::{DomainMetadata, VerificationStatus},
    types::DomainName,
    crypto::AxonSigningKey,
};
use nym_core::NymIdentity;
use std::collections::HashMap;

fn create_test_config() -> NymverseDomainConfig {
    let mut length_multiplier = HashMap::new();
    length_multiplier.insert(1, 10.0);  // Single character domains are 10x more expensive
    length_multiplier.insert(2, 5.0);   // Two character domains are 5x more expensive
    length_multiplier.insert(3, 2.0);   // Three character domains are 2x more expensive
    
    let mut premium_keywords = HashMap::new();
    premium_keywords.insert("crypto".to_string(), 3.0);
    premium_keywords.insert("nft".to_string(), 2.5);
    premium_keywords.insert("defi".to_string(), 2.0);
    
    NymverseDomainConfig {
        quid_pricing: QuidDomainPricing {
            base_price: 100_000,  // 100,000 NYM tokens
            renewal_price: 50_000,
            transfer_fee: 10_000,
            identity_verification_bonus: 0.1,  // 10% discount for verified identities
        },
        axon_pricing: AxonDomainPricing {
            base_price: 50_000,   // 50,000 NYM tokens
            renewal_price: 25_000,
            transfer_fee: 5_000,
            length_multiplier,
            premium_keywords,
        },
        revenue_distribution: RevenueDistribution {
            development_fund: 0.2,    // 20% to development
            ecosystem_fund: 0.15,     // 15% to ecosystem
            token_burn: 0.3,          // 30% burned (deflationary)
            validator_rewards: 0.25,  // 25% to validators
            creator_rewards: 0.1,     // 10% to creators
        },
        governance_settings: GovernanceSettings {
            voting_threshold: 0.1,    // 10% participation required
            proposal_delay: 86400,    // 24 hours delay
            emergency_admin: None,
        },
    }
}

#[test]
fn test_quid_domain_registration() {
    let admin_key = AxonSigningKey::generate();
    let user_key = AxonSigningKey::generate();
    let config = create_test_config();
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    // Create QuID identity
    let quid_identity = NymIdentity::from_bytes(&[1; 32]).unwrap();
    let domain_name = DomainName::new("alice.quid".to_string()).unwrap();
    
    // Register .quid domain
    let result = registry.register_quid_domain(
        domain_name.clone(),
        quid_identity.clone(),
        user_key.verifying_key(),
        vec![0u8; 32], // Payment proof placeholder
    );
    
    assert!(result.is_ok());
    
    // Verify domain is registered
    let registered_domain = registry.get_quid_domain_for_identity(&quid_identity);
    assert_eq!(registered_domain, Some(&domain_name));
    
    // Try to register another .quid domain for the same identity (should fail)
    let another_domain = DomainName::new("bob.quid".to_string()).unwrap();
    let result = registry.register_quid_domain(
        another_domain,
        quid_identity,
        user_key.verifying_key(),
        vec![0u8; 32],
    );
    
    assert!(result.is_err());
}

#[test]
fn test_axon_domain_registration() {
    let admin_key = AxonSigningKey::generate();
    let user_key = AxonSigningKey::generate();
    let config = create_test_config();
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    // Create QuID identity
    let quid_identity = NymIdentity::from_bytes(&[1; 32]).unwrap();
    
    // Register multiple .axon domains for the same identity
    let domains = vec![
        "myblog.axon",
        "portfolio.axon",
        "shop.axon",
    ];
    
    for domain_str in domains {
        let domain_name = DomainName::new(domain_str.to_string()).unwrap();
        let metadata = DomainMetadata {
            description: Some(format!("Test domain: {}", domain_str)),
            keywords: vec!["test".to_string()],
            contact_info: None,
            category: Some("personal".to_string()),
            reputation_score: 0,
            verification_status: VerificationStatus::Unverified,
        };
        
        let result = registry.register_axon_domain(
            domain_name,
            quid_identity.clone(),
            user_key.verifying_key(),
            vec![0u8; 32], // Payment proof placeholder
            metadata,
        );
        
        assert!(result.is_ok());
    }
    
    // Verify all domains are registered for the identity
    let registered_domains = registry.get_axon_domains_for_identity(&quid_identity);
    assert_eq!(registered_domains.len(), 3);
}

#[test]
fn test_premium_pricing() {
    let admin_key = AxonSigningKey::generate();
    let user_key = AxonSigningKey::generate();
    let config = create_test_config();
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    let quid_identity = NymIdentity::from_bytes(&[1; 32]).unwrap();
    
    // Register premium keyword domain
    let crypto_domain = DomainName::new("crypto.axon".to_string()).unwrap();
    let metadata = DomainMetadata {
        description: Some("Crypto domain".to_string()),
        keywords: vec!["crypto".to_string()],
        contact_info: None,
        category: Some("finance".to_string()),
        reputation_score: 0,
        verification_status: VerificationStatus::Unverified,
    };
    
    let result = registry.register_axon_domain(
        crypto_domain,
        quid_identity.clone(),
        user_key.verifying_key(),
        vec![0u8; 32],
        metadata,
    );
    
    assert!(result.is_ok());
    
    // Register short domain (should be more expensive)
    let short_domain = DomainName::new("a.axon".to_string()).unwrap();
    let metadata = DomainMetadata {
        description: Some("Short domain".to_string()),
        keywords: vec!["short".to_string()],
        contact_info: None,
        category: Some("personal".to_string()),
        reputation_score: 0,
        verification_status: VerificationStatus::Unverified,
    };
    
    let result = registry.register_axon_domain(
        short_domain,
        quid_identity,
        user_key.verifying_key(),
        vec![0u8; 32],
        metadata,
    );
    
    assert!(result.is_ok());
}

#[test]
fn test_quid_domain_transfer() {
    let admin_key = AxonSigningKey::generate();
    let user1_key = AxonSigningKey::generate();
    let user2_key = AxonSigningKey::generate();
    let config = create_test_config();
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    // Create two QuID identities
    let identity1 = NymIdentity::from_bytes(&[1; 32]).unwrap();
    let identity2 = NymIdentity::from_bytes(&[2; 32]).unwrap();
    
    let domain_name = DomainName::new("alice.quid".to_string()).unwrap();
    
    // Register domain for identity1
    let result = registry.register_quid_domain(
        domain_name.clone(),
        identity1.clone(),
        user1_key.verifying_key(),
        vec![0u8; 32],
    );
    assert!(result.is_ok());
    
    // Transfer domain to identity2
    let result = registry.transfer_quid_domain(
        &domain_name,
        &identity1,
        &identity2,
        &user1_key.verifying_key(),
        &user2_key.verifying_key(),
    );
    assert!(result.is_ok());
    
    // Verify transfer
    assert_eq!(registry.get_quid_domain_for_identity(&identity1), None);
    assert_eq!(registry.get_quid_domain_for_identity(&identity2), Some(&domain_name));
}

#[test]
fn test_governance_system() {
    let admin_key = AxonSigningKey::generate();
    let proposer_key = AxonSigningKey::generate();
    let voter_key = AxonSigningKey::generate();
    let config = create_test_config();
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    // Create governance proposal to update pricing
    let new_pricing = QuidDomainPricing {
        base_price: 200_000,  // Double the price
        renewal_price: 100_000,
        transfer_fee: 20_000,
        identity_verification_bonus: 0.15,
    };
    
    let proposal_id = registry.create_governance_proposal(
        proposer_key.verifying_key(),
        ProposalType::PricingUpdate,
        "Increase .quid domain pricing".to_string(),
        ProposalChanges::UpdateQuidPricing(new_pricing),
        604800, // 7 days voting period
    ).unwrap();
    
    // Vote on proposal
    let result = registry.vote_on_proposal(
        &proposal_id,
        voter_key.verifying_key(),
        VoteType::For,
        1000, // Voting power
    );
    assert!(result.is_ok());
    
    // Try to vote again (should fail)
    let result = registry.vote_on_proposal(
        &proposal_id,
        voter_key.verifying_key(),
        VoteType::Against,
        1000,
    );
    assert!(result.is_err());
}

#[test]
fn test_revenue_tracking() {
    let admin_key = AxonSigningKey::generate();
    let user_key = AxonSigningKey::generate();
    let config = create_test_config();
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    let quid_identity = NymIdentity::from_bytes(&[1; 32]).unwrap();
    
    // Register some domains to generate revenue
    let quid_domain = DomainName::new("alice.quid".to_string()).unwrap();
    let _ = registry.register_quid_domain(
        quid_domain,
        quid_identity.clone(),
        user_key.verifying_key(),
        vec![0u8; 32],
    );
    
    let axon_domain = DomainName::new("myblog.axon".to_string()).unwrap();
    let metadata = DomainMetadata {
        description: Some("Blog domain".to_string()),
        keywords: vec!["blog".to_string()],
        contact_info: None,
        category: Some("personal".to_string()),
        reputation_score: 0,
        verification_status: VerificationStatus::Unverified,
    };
    
    let _ = registry.register_axon_domain(
        axon_domain,
        quid_identity,
        user_key.verifying_key(),
        vec![0u8; 32],
        metadata,
    );
    
    // Check registry statistics
    let stats = registry.get_registry_statistics();
    assert_eq!(stats.quid_domains, 1);
    assert_eq!(stats.axon_domains, 1);
    assert!(stats.total_revenue > 0);
    assert!(stats.burned_tokens > 0);
    assert!(stats.revenue_distribution.contains_key("development"));
    assert!(stats.revenue_distribution.contains_key("ecosystem"));
    assert!(stats.revenue_distribution.contains_key("validators"));
    assert!(stats.revenue_distribution.contains_key("creators"));
}

#[test]
fn test_domain_validation() {
    let admin_key = AxonSigningKey::generate();
    let user_key = AxonSigningKey::generate();
    let config = create_test_config();
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    let quid_identity = NymIdentity::from_bytes(&[1; 32]).unwrap();
    
    // Try to register invalid .quid domain (too short)
    let invalid_domain = DomainName::new("a.quid".to_string()).unwrap();
    let result = registry.register_quid_domain(
        invalid_domain,
        quid_identity.clone(),
        user_key.verifying_key(),
        vec![0u8; 32],
    );
    assert!(result.is_err());
    
    // Try to register invalid .quid domain (too long)
    let long_name = "a".repeat(33);
    let invalid_domain = DomainName::new(format!("{}.quid", long_name)).unwrap();
    let result = registry.register_quid_domain(
        invalid_domain,
        quid_identity,
        user_key.verifying_key(),
        vec![0u8; 32],
    );
    assert!(result.is_err());
}

#[test]
fn test_one_to_one_quid_mapping() {
    let admin_key = AxonSigningKey::generate();
    let user_key = AxonSigningKey::generate();
    let config = create_test_config();
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    let quid_identity = NymIdentity::from_bytes(&[1; 32]).unwrap();
    
    // Register first .quid domain
    let domain1 = DomainName::new("alice.quid".to_string()).unwrap();
    let result = registry.register_quid_domain(
        domain1,
        quid_identity.clone(),
        user_key.verifying_key(),
        vec![0u8; 32],
    );
    assert!(result.is_ok());
    
    // Try to register second .quid domain for same identity (should fail)
    let domain2 = DomainName::new("bob.quid".to_string()).unwrap();
    let result = registry.register_quid_domain(
        domain2,
        quid_identity,
        user_key.verifying_key(),
        vec![0u8; 32],
    );
    assert!(result.is_err());
}

#[test]
fn test_one_to_many_axon_mapping() {
    let admin_key = AxonSigningKey::generate();
    let user_key = AxonSigningKey::generate();
    let config = create_test_config();
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    let quid_identity = NymIdentity::from_bytes(&[1; 32]).unwrap();
    
    // Register multiple .axon domains for same identity (should succeed)
    let domains = vec![
        "blog.axon",
        "portfolio.axon",
        "shop.axon",
        "gallery.axon",
        "docs.axon",
    ];
    
    for domain_str in domains {
        let domain_name = DomainName::new(domain_str.to_string()).unwrap();
        let metadata = DomainMetadata {
            description: Some(format!("Test domain: {}", domain_str)),
            keywords: vec!["test".to_string()],
            contact_info: None,
            category: Some("personal".to_string()),
            reputation_score: 0,
            verification_status: VerificationStatus::Unverified,
        };
        
        let result = registry.register_axon_domain(
            domain_name,
            quid_identity.clone(),
            user_key.verifying_key(),
            vec![0u8; 32],
            metadata,
        );
        
        assert!(result.is_ok());
    }
    
    // Verify all domains are registered
    let registered_domains = registry.get_axon_domains_for_identity(&quid_identity);
    assert_eq!(registered_domains.len(), 5);
}