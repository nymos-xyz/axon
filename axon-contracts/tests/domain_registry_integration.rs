//! Integration tests for domain registry smart contracts

use axon_contracts::{
    QuidAxonRegistry, NymverseDomainConfig, QuidDomainPricing, AxonDomainPricing,
    RevenueDistribution, GovernanceSettings, AdaptivePricingContract, MarketIndicators
};
use axon_core::{
    crypto::AxonSigningKey,
    domain::{DomainPricing, DomainTypePricing, DomainMetadata, VerificationStatus},
    types::{DomainName, DomainType, Timestamp},
};
use nym_core::NymIdentity;
use std::collections::HashMap;

#[tokio::test]
async fn test_quid_axon_domain_registry_integration() {
    // Setup test environment
    let admin_key = AxonSigningKey::generate();
    let user1_key = AxonSigningKey::generate();
    let user2_key = AxonSigningKey::generate();
    
    // Create QuID identities
    let quid_identity1 = NymIdentity::new("user1_identity_hash".to_string());
    let quid_identity2 = NymIdentity::new("user2_identity_hash".to_string());
    
    // Configure registry
    let config = NymverseDomainConfig {
        quid_pricing: QuidDomainPricing {
            base_price: 10_000,  // 10k NYM tokens
            renewal_price: 5_000,
            transfer_fee: 1_000,
            identity_verification_bonus: 0.1, // 10% discount for verified
        },
        axon_pricing: AxonDomainPricing {
            base_price: 5_000,   // 5k NYM tokens
            renewal_price: 2_500,
            transfer_fee: 500,
            length_multiplier: {
                let mut map = HashMap::new();
                map.insert(2, 3.0);  // 2-char domains cost 3x
                map.insert(3, 2.0);  // 3-char domains cost 2x
                map.insert(4, 1.5);  // 4-char domains cost 1.5x
                map
            },
            premium_keywords: {
                let mut map = HashMap::new();
                map.insert("crypto".to_string(), 2.0);
                map.insert("nft".to_string(), 1.8);
                map.insert("defi".to_string(), 1.6);
                map
            },
        },
        revenue_distribution: RevenueDistribution {
            development_fund: 0.25,   // 25%
            ecosystem_fund: 0.20,     // 20%
            token_burn: 0.15,         // 15% (deflationary)
            validator_rewards: 0.30,  // 30%
            creator_rewards: 0.10,    // 10%
        },
        governance_settings: GovernanceSettings {
            voting_threshold: 0.67,   // 67% participation required
            proposal_delay: 7 * 24 * 3600, // 7 days
            emergency_admin: Some(admin_key.verifying_key()),
        },
    };
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config);
    
    // Test 1: Register .quid domain (social profile)
    let quid_domain = DomainName::new("alice".to_string()).unwrap();
    let payment_proof = vec![1, 2, 3, 4]; // Mock payment proof
    
    let result = registry.register_quid_domain(
        quid_domain.clone(),
        quid_identity1.clone(),
        user1_key.verifying_key(),
        payment_proof.clone(),
    );
    
    assert!(result.is_ok(), "Failed to register .quid domain: {:?}", result);
    
    // Verify 1:1 mapping
    let mapped_domain = registry.get_quid_domain_for_identity(&quid_identity1);
    assert_eq!(mapped_domain, Some(&quid_domain));
    
    // Test 2: Try to register another .quid domain for same identity (should fail)
    let quid_domain2 = DomainName::new("alice2".to_string()).unwrap();
    let result = registry.register_quid_domain(
        quid_domain2,
        quid_identity1.clone(),
        user1_key.verifying_key(),
        payment_proof.clone(),
    );
    
    assert!(result.is_err(), "Should not allow second .quid domain for same identity");
    
    // Test 3: Register .axon domains (generic content) - multiple allowed
    let axon_domain1 = DomainName::new("myblog".to_string()).unwrap();
    let axon_domain2 = DomainName::new("portfolio".to_string()).unwrap();
    let axon_domain3 = DomainName::new("crypto".to_string()).unwrap(); // Premium keyword
    
    let metadata = DomainMetadata {
        description: Some("Personal blog".to_string()),
        keywords: vec!["blog".to_string(), "personal".to_string()],
        contact_info: None,
        category: Some("personal".to_string()),
        reputation_score: 0,
        verification_status: VerificationStatus::Unverified,
    };
    
    // Register first .axon domain
    let result = registry.register_axon_domain(
        axon_domain1.clone(),
        quid_identity1.clone(),
        user1_key.verifying_key(),
        payment_proof.clone(),
        metadata.clone(),
    );
    assert!(result.is_ok(), "Failed to register first .axon domain: {:?}", result);
    
    // Register second .axon domain for same identity
    let result = registry.register_axon_domain(
        axon_domain2.clone(),
        quid_identity1.clone(),
        user1_key.verifying_key(),
        payment_proof.clone(),
        metadata.clone(),
    );
    assert!(result.is_ok(), "Failed to register second .axon domain: {:?}", result);
    
    // Register premium keyword domain (should cost more)
    let result = registry.register_axon_domain(
        axon_domain3.clone(),
        quid_identity1.clone(),
        user1_key.verifying_key(),
        payment_proof.clone(),
        metadata,
    );
    assert!(result.is_ok(), "Failed to register premium .axon domain: {:?}", result);
    
    // Verify 1:many mapping
    let axon_domains = registry.get_axon_domains_for_identity(&quid_identity1);
    assert_eq!(axon_domains.len(), 3);
    assert!(axon_domains.contains(&&axon_domain1));
    assert!(axon_domains.contains(&&axon_domain2));
    assert!(axon_domains.contains(&&axon_domain3));
    
    // Test 4: Domain transfer for .quid (changes identity mapping)
    let result = registry.transfer_quid_domain(
        &quid_domain,
        &quid_identity1,
        &quid_identity2,
        &user1_key.verifying_key(),
        &user2_key.verifying_key(),
    );
    assert!(result.is_ok(), "Failed to transfer .quid domain: {:?}", result);
    
    // Verify mapping updated
    assert_eq!(registry.get_quid_domain_for_identity(&quid_identity1), None);
    assert_eq!(registry.get_quid_domain_for_identity(&quid_identity2), Some(&quid_domain));
    
    // Test 5: Check registry statistics
    let stats = registry.get_registry_statistics();
    assert_eq!(stats.quid_domains, 1);
    assert_eq!(stats.axon_domains, 3);
    assert!(stats.total_revenue > 0);
    assert!(stats.burned_tokens > 0);
    assert_eq!(stats.quid_identity_mapping_size, 1);
    
    println!("Domain registry integration test completed successfully!");
    println!("Stats: {:?}", stats);
}

#[tokio::test]
async fn test_adaptive_pricing_integration() {
    let admin_key = AxonSigningKey::generate();
    
    let initial_pricing = DomainPricing {
        base_prices: DomainTypePricing {
            standard: 1000,
            premium: 5000,
            vanity: 2000,
            organization: 3000,
            community: 1500,
        },
        network_health_multiplier: 1.0,
        demand_multiplier: 1.0,
        updated_at: Timestamp::now(),
    };
    
    let mut pricing_contract = AdaptivePricingContract::new(
        admin_key.verifying_key(),
        initial_pricing,
    );
    
    // Test 1: Network health impact on pricing
    let initial_price = pricing_contract.calculate_domain_price(&DomainType::Standard, 1);
    assert_eq!(initial_price, 1000);
    
    // Simulate poor network health (low consensus participation)
    let result = pricing_contract.update_network_metrics(
        0.5,    // 50% consensus participation (unhealthy)
        2000,   // High zk-proof cost
        0.9,    // High utilization
        0.7,    // Low node availability
    );
    assert!(result.is_ok());
    
    let adjusted_price = pricing_contract.calculate_domain_price(&DomainType::Standard, 1);
    println!("Price adjustment due to poor network health: {} -> {}", initial_price, adjusted_price);
    
    // Test 2: Demand impact on pricing
    let mut registrations = HashMap::new();
    registrations.insert(DomainType::Standard, 100); // High demand
    registrations.insert(DomainType::Premium, 20);
    
    let result = pricing_contract.update_demand_metrics(
        registrations,
        10000, // Total domains
        0.05,  // 5% expiration rate
        0.95,  // 95% renewal rate
    );
    assert!(result.is_ok());
    
    let demand_adjusted_price = pricing_contract.calculate_domain_price(&DomainType::Standard, 1);
    println!("Price after demand adjustment: {}", demand_adjusted_price);
    
    // Test 3: Market-based pricing with trending keywords
    let market_indicators = MarketIndicators {
        trending_keywords: vec!["ai".to_string(), "crypto".to_string()],
        competition_factor: 0.3,
        time_factor: 1.2, // Peak hours
        market_volatility: 0.1,
        external_demand_signals: vec![],
    };
    
    let market_price_normal = pricing_contract.calculate_market_price(
        &DomainType::Standard,
        "testdomain",
        1,
        &market_indicators,
    );
    
    let market_price_trending = pricing_contract.calculate_market_price(
        &DomainType::Standard,
        "aidomain",
        1,
        &market_indicators,
    );
    
    assert!(market_price_trending > market_price_normal, 
           "Trending keyword domain should cost more");
    
    println!("Market pricing - Normal: {}, Trending: {}", 
             market_price_normal, market_price_trending);
    
    // Test 4: Price trend prediction
    let prediction = pricing_contract.predict_price_trend(&DomainType::Standard, 30);
    println!("Price trend prediction: {:?}", prediction);
    
    assert!(prediction.confidence > 0.0);
    assert!(prediction.days_ahead == 30);
    
    // Test 5: Manual pricing adjustment by admin
    let result = pricing_contract.manual_adjustment(
        &admin_key.verifying_key(),
        None,
        Some(1.5), // Increase network health multiplier
        Some(0.8), // Decrease demand multiplier
    );
    assert!(result.is_ok());
    
    let manually_adjusted_price = pricing_contract.calculate_domain_price(&DomainType::Standard, 1);
    println!("Manually adjusted price: {}", manually_adjusted_price);
    
    println!("Adaptive pricing integration test completed successfully!");
}

#[tokio::test]
async fn test_revenue_distribution_and_governance() {
    let admin_key = AxonSigningKey::generate();
    let voter1_key = AxonSigningKey::generate();
    let voter2_key = AxonSigningKey::generate();
    
    let config = NymverseDomainConfig {
        quid_pricing: QuidDomainPricing {
            base_price: 10_000,
            renewal_price: 5_000,
            transfer_fee: 1_000,
            identity_verification_bonus: 0.1,
        },
        axon_pricing: AxonDomainPricing {
            base_price: 5_000,
            renewal_price: 2_500,
            transfer_fee: 500,
            length_multiplier: HashMap::new(),
            premium_keywords: HashMap::new(),
        },
        revenue_distribution: RevenueDistribution {
            development_fund: 0.25,
            ecosystem_fund: 0.20,
            token_burn: 0.15,
            validator_rewards: 0.30,
            creator_rewards: 0.10,
        },
        governance_settings: GovernanceSettings {
            voting_threshold: 0.67,
            proposal_delay: 3600, // 1 hour for testing
            emergency_admin: Some(admin_key.verifying_key()),
        },
    };
    
    let mut registry = QuidAxonRegistry::new(admin_key.verifying_key(), config.clone());
    
    // Test governance proposal creation
    use axon_contracts::{ProposalType, ProposalChanges};
    
    let new_revenue_distribution = RevenueDistribution {
        development_fund: 0.30, // Increase development funding
        ecosystem_fund: 0.20,
        token_burn: 0.10,       // Reduce burn rate
        validator_rewards: 0.30,
        creator_rewards: 0.10,
    };
    
    let proposal_id = registry.create_governance_proposal(
        admin_key.verifying_key(),
        ProposalType::RevenueDistributionChange,
        "Increase development funding".to_string(),
        ProposalChanges::UpdateRevenueDistribution(new_revenue_distribution.clone()),
        86400, // 24 hours voting period
    ).unwrap();
    
    println!("Created governance proposal: {}", proposal_id);
    
    // Test voting
    use axon_contracts::VoteType;
    
    let result = registry.vote_on_proposal(
        &proposal_id,
        voter1_key.verifying_key(),
        VoteType::For,
        1000, // Voting power
    );
    assert!(result.is_ok());
    
    let result = registry.vote_on_proposal(
        &proposal_id,
        voter2_key.verifying_key(),
        VoteType::For,
        500, // Voting power
    );
    assert!(result.is_ok());
    
    // Note: In a real test, we would need to wait for the voting period to end
    // and have sufficient votes to execute the proposal
    
    println!("Revenue distribution and governance test completed!");
}