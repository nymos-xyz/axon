use crate::domain_registry::{DomainRegistryContract, DomainRegistryEvent};
use axon_core::{
    domain::{DomainRecord, DomainRegistrationRequest, DomainPricing},
    types::{DomainName, DomainType, Timestamp},
    crypto::AxonVerifyingKey,
    Result, AxonError,
};
use nym_core::NymIdentity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NymverseDomainType {
    Quid,  // .quid domains - social profile domains (1:1 with QuID identity)
    Axon,  // .axon domains - generic content domains (1:many with QuID identity)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NymverseDomainConfig {
    pub quid_pricing: QuidDomainPricing,
    pub axon_pricing: AxonDomainPricing,
    pub revenue_distribution: RevenueDistribution,
    pub governance_settings: GovernanceSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuidDomainPricing {
    pub base_price: u64,           // Base price in NYM tokens
    pub renewal_price: u64,        // Annual renewal price
    pub transfer_fee: u64,         // Fee for domain transfers
    pub identity_verification_bonus: f64, // Discount for verified identities
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxonDomainPricing {
    pub base_price: u64,
    pub renewal_price: u64,
    pub transfer_fee: u64,
    pub length_multiplier: HashMap<usize, f64>, // Price multiplier by domain length
    pub premium_keywords: HashMap<String, f64>, // Premium pricing for certain keywords
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevenueDistribution {
    pub development_fund: f64,     // Percentage to development fund
    pub ecosystem_fund: f64,       // Percentage to ecosystem growth
    pub token_burn: f64,          // Percentage to burn (deflationary)
    pub validator_rewards: f64,    // Percentage to validators
    pub creator_rewards: f64,      // Percentage to creators
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceSettings {
    pub voting_threshold: f64,     // Minimum participation for votes
    pub proposal_delay: u64,       // Time before proposal takes effect
    pub emergency_admin: Option<AxonVerifyingKey>, // Emergency admin key
}

pub struct QuidAxonRegistry {
    quid_registry: DomainRegistryContract,
    axon_registry: DomainRegistryContract,
    config: NymverseDomainConfig,
    quid_identity_mapping: HashMap<NymIdentity, DomainName>, // 1:1 mapping
    axon_identity_mapping: HashMap<NymIdentity, Vec<DomainName>>, // 1:many mapping
    revenue_tracker: RevenueTracker,
    governance: GovernanceManager,
}

#[derive(Debug, Clone)]
struct RevenueTracker {
    total_revenue: u64,
    revenue_by_type: HashMap<NymverseDomainType, u64>,
    distributed_funds: HashMap<String, u64>,
    burned_tokens: u64,
}

#[derive(Debug)]
struct GovernanceManager {
    active_proposals: HashMap<String, GovernanceProposal>,
    voting_records: HashMap<String, VotingRecord>,
    proposal_counter: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GovernanceProposal {
    proposal_id: String,
    proposer: AxonVerifyingKey,
    proposal_type: ProposalType,
    description: String,
    proposed_changes: ProposalChanges,
    created_at: Timestamp,
    voting_ends_at: Timestamp,
    execution_delay: u64,
    minimum_votes_required: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ProposalType {
    PricingUpdate,
    RevenueDistributionChange,
    GovernanceParameterChange,
    EmergencyAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ProposalChanges {
    UpdateQuidPricing(QuidDomainPricing),
    UpdateAxonPricing(AxonDomainPricing),
    UpdateRevenueDistribution(RevenueDistribution),
    UpdateGovernanceSettings(GovernanceSettings),
    EmergencyFreeze { reason: String },
}

#[derive(Debug)]
struct VotingRecord {
    proposal_id: String,
    votes_for: u64,
    votes_against: u64,
    voters: HashMap<AxonVerifyingKey, Vote>,
    voting_power_snapshot: HashMap<AxonVerifyingKey, u64>,
}

#[derive(Debug, Clone)]
struct Vote {
    voter: AxonVerifyingKey,
    vote_type: VoteType,
    voting_power: u64,
    timestamp: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum VoteType {
    For,
    Against,
    Abstain,
}

impl QuidAxonRegistry {
    pub fn new(
        admin: AxonVerifyingKey,
        config: NymverseDomainConfig,
    ) -> Self {
        info!("Initializing Quid/Axon domain registry with governance");

        // Create specialized pricing for each domain type
        let quid_pricing = DomainPricing {
            base_prices: axon_core::domain::DomainTypePricing {
                standard: config.quid_pricing.base_price,
                premium: config.quid_pricing.base_price * 2,
                vanity: config.quid_pricing.base_price * 3,
                organization: config.quid_pricing.base_price * 5,
                community: config.quid_pricing.base_price,
            },
            network_health_multiplier: 1.0,
            demand_multiplier: 1.0,
            updated_at: Timestamp::now(),
        };

        let axon_pricing = DomainPricing {
            base_prices: axon_core::domain::DomainTypePricing {
                standard: config.axon_pricing.base_price,
                premium: config.axon_pricing.base_price * 3,
                vanity: config.axon_pricing.base_price * 2,
                organization: config.axon_pricing.base_price * 4,
                community: config.axon_pricing.base_price,
            },
            network_health_multiplier: 1.0,
            demand_multiplier: 1.0,
            updated_at: Timestamp::now(),
        };

        Self {
            quid_registry: DomainRegistryContract::new(admin.clone(), quid_pricing),
            axon_registry: DomainRegistryContract::new(admin.clone(), axon_pricing),
            config,
            quid_identity_mapping: HashMap::new(),
            axon_identity_mapping: HashMap::new(),
            revenue_tracker: RevenueTracker {
                total_revenue: 0,
                revenue_by_type: HashMap::new(),
                distributed_funds: HashMap::new(),
                burned_tokens: 0,
            },
            governance: GovernanceManager {
                active_proposals: HashMap::new(),
                voting_records: HashMap::new(),
                proposal_counter: 0,
            },
        }
    }

    pub fn register_quid_domain(
        &mut self,
        domain_name: DomainName,
        quid_identity: NymIdentity,
        owner_key: AxonVerifyingKey,
        payment_proof: Vec<u8>,
    ) -> Result<DomainRegistryEvent> {
        info!("Registering .quid domain: {} for identity: {}", 
              domain_name.as_str(), quid_identity.to_string());

        // Enforce 1:1 mapping between QuID identity and .quid domain
        if self.quid_identity_mapping.contains_key(&quid_identity) {
            return Err(AxonError::InvalidDomain(
                "QuID identity already has a .quid domain registered".to_string()
            ));
        }

        // Verify that the domain name matches QuID identity requirements
        self.validate_quid_domain_name(&domain_name, &quid_identity)?;

        // Create registration request
        let request = self.create_quid_registration_request(
            domain_name.clone(),
            quid_identity.clone(),
            owner_key,
            payment_proof,
        )?;

        // Register the domain
        let event = self.quid_registry.register_domain(request)?;

        // Update identity mapping
        self.quid_identity_mapping.insert(quid_identity, domain_name.clone());

        // Track revenue
        self.track_revenue(NymverseDomainType::Quid, self.config.quid_pricing.base_price);

        info!("Successfully registered .quid domain: {}", domain_name.as_str());
        Ok(event)
    }

    pub fn register_axon_domain(
        &mut self,
        domain_name: DomainName,
        quid_identity: NymIdentity,
        owner_key: AxonVerifyingKey,
        payment_proof: Vec<u8>,
        domain_metadata: axon_core::domain::DomainMetadata,
    ) -> Result<DomainRegistryEvent> {
        info!("Registering .axon domain: {} for identity: {}", 
              domain_name.as_str(), quid_identity.to_string());

        // Validate domain name for .axon requirements
        self.validate_axon_domain_name(&domain_name)?;

        // Calculate pricing based on domain characteristics
        let pricing = self.calculate_axon_pricing(&domain_name);

        // Create registration request
        let request = self.create_axon_registration_request(
            domain_name.clone(),
            quid_identity.clone(),
            owner_key,
            payment_proof,
            domain_metadata,
            pricing,
        )?;

        // Register the domain
        let event = self.axon_registry.register_domain(request)?;

        // Update identity mapping (1:many relationship)
        self.axon_identity_mapping.entry(quid_identity)
            .or_insert_with(Vec::new)
            .push(domain_name.clone());

        // Track revenue
        self.track_revenue(NymverseDomainType::Axon, pricing);

        info!("Successfully registered .axon domain: {}", domain_name.as_str());
        Ok(event)
    }

    pub fn transfer_quid_domain(
        &mut self,
        domain_name: &DomainName,
        from_identity: &NymIdentity,
        to_identity: &NymIdentity,
        from_key: &AxonVerifyingKey,
        to_key: &AxonVerifyingKey,
    ) -> Result<DomainRegistryEvent> {
        info!("Transferring .quid domain: {} from {} to {}", 
              domain_name.as_str(), from_identity.to_string(), to_identity.to_string());

        // Verify current ownership
        if self.quid_identity_mapping.get(from_identity) != Some(domain_name) {
            return Err(AxonError::PermissionDenied);
        }

        // Ensure target identity doesn't already have a .quid domain
        if self.quid_identity_mapping.contains_key(to_identity) {
            return Err(AxonError::InvalidDomain(
                "Target identity already has a .quid domain".to_string()
            ));
        }

        // Perform the transfer
        let event = self.quid_registry.transfer_domain(domain_name, from_key, to_key)?;

        // Update mappings
        self.quid_identity_mapping.remove(from_identity);
        self.quid_identity_mapping.insert(to_identity.clone(), domain_name.clone());

        // Track transfer fee revenue
        self.track_revenue(NymverseDomainType::Quid, self.config.quid_pricing.transfer_fee);

        Ok(event)
    }

    pub fn get_quid_domain_for_identity(&self, identity: &NymIdentity) -> Option<&DomainName> {
        self.quid_identity_mapping.get(identity)
    }

    pub fn get_axon_domains_for_identity(&self, identity: &NymIdentity) -> Vec<&DomainName> {
        self.axon_identity_mapping.get(identity)
            .map(|domains| domains.iter().collect())
            .unwrap_or_default()
    }

    pub fn create_governance_proposal(
        &mut self,
        proposer: AxonVerifyingKey,
        proposal_type: ProposalType,
        description: String,
        proposed_changes: ProposalChanges,
        voting_duration: u64,
    ) -> Result<String> {
        let proposal_id = format!("prop_{}", self.governance.proposal_counter);
        self.governance.proposal_counter += 1;

        let proposal = GovernanceProposal {
            proposal_id: proposal_id.clone(),
            proposer,
            proposal_type,
            description,
            proposed_changes,
            created_at: Timestamp::now(),
            voting_ends_at: Timestamp(Timestamp::now().0 + voting_duration),
            execution_delay: self.config.governance_settings.proposal_delay,
            minimum_votes_required: 1000, // Would be calculated based on total supply
        };

        let voting_record = VotingRecord {
            proposal_id: proposal_id.clone(),
            votes_for: 0,
            votes_against: 0,
            voters: HashMap::new(),
            voting_power_snapshot: HashMap::new(),
        };

        self.governance.active_proposals.insert(proposal_id.clone(), proposal);
        self.governance.voting_records.insert(proposal_id.clone(), voting_record);

        info!("Created governance proposal: {}", proposal_id);
        Ok(proposal_id)
    }

    pub fn vote_on_proposal(
        &mut self,
        proposal_id: &str,
        voter: AxonVerifyingKey,
        vote_type: VoteType,
        voting_power: u64,
    ) -> Result<()> {
        let proposal = self.governance.active_proposals.get(proposal_id)
            .ok_or_else(|| AxonError::InvalidDomain("Proposal not found".to_string()))?;

        if Timestamp::now().0 > proposal.voting_ends_at.0 {
            return Err(AxonError::InvalidDomain("Voting period has ended".to_string()));
        }

        let voting_record = self.governance.voting_records.get_mut(proposal_id)
            .ok_or_else(|| AxonError::Internal("Voting record not found".to_string()))?;

        // Check if already voted
        if voting_record.voters.contains_key(&voter) {
            return Err(AxonError::InvalidDomain("Already voted on this proposal".to_string()));
        }

        let vote = Vote {
            voter: voter.clone(),
            vote_type: vote_type.clone(),
            voting_power,
            timestamp: Timestamp::now(),
        };

        match vote_type {
            VoteType::For => voting_record.votes_for += voting_power,
            VoteType::Against => voting_record.votes_against += voting_power,
            VoteType::Abstain => {}, // Abstain doesn't add to either side
        }

        voting_record.voters.insert(voter.clone(), vote);
        voting_record.voting_power_snapshot.insert(voter, voting_power);

        Ok(())
    }

    pub fn execute_proposal(&mut self, proposal_id: &str) -> Result<()> {
        let proposal = self.governance.active_proposals.get(proposal_id)
            .ok_or_else(|| AxonError::InvalidDomain("Proposal not found".to_string()))?
            .clone();

        let voting_record = self.governance.voting_records.get(proposal_id)
            .ok_or_else(|| AxonError::Internal("Voting record not found".to_string()))?;

        // Check if voting period has ended
        if Timestamp::now().0 <= proposal.voting_ends_at.0 {
            return Err(AxonError::InvalidDomain("Voting period still active".to_string()));
        }

        // Check if proposal passed
        let total_votes = voting_record.votes_for + voting_record.votes_against;
        if total_votes < proposal.minimum_votes_required {
            return Err(AxonError::InvalidDomain("Insufficient votes".to_string()));
        }

        if voting_record.votes_for <= voting_record.votes_against {
            return Err(AxonError::InvalidDomain("Proposal did not pass".to_string()));
        }

        // Execute the proposal
        match proposal.proposed_changes {
            ProposalChanges::UpdateQuidPricing(new_pricing) => {
                self.config.quid_pricing = new_pricing;
                info!("Updated .quid pricing via governance");
            }
            ProposalChanges::UpdateAxonPricing(new_pricing) => {
                self.config.axon_pricing = new_pricing;
                info!("Updated .axon pricing via governance");
            }
            ProposalChanges::UpdateRevenueDistribution(new_distribution) => {
                self.config.revenue_distribution = new_distribution;
                info!("Updated revenue distribution via governance");
            }
            ProposalChanges::UpdateGovernanceSettings(new_settings) => {
                self.config.governance_settings = new_settings;
                info!("Updated governance settings via governance");
            }
            ProposalChanges::EmergencyFreeze { reason } => {
                warn!("Emergency freeze activated via governance: {}", reason);
                // Implementation would freeze domain registrations
            }
        }

        // Remove from active proposals
        self.governance.active_proposals.remove(proposal_id);

        Ok(())
    }

    fn validate_quid_domain_name(&self, domain_name: &DomainName, quid_identity: &NymIdentity) -> Result<()> {
        let name = domain_name.as_str();
        
        // Basic validation rules for .quid domains
        if name.len() < 3 || name.len() > 32 {
            return Err(AxonError::InvalidDomain(
                ".quid domains must be 3-32 characters".to_string()
            ));
        }

        // Must be alphanumeric and hyphens only
        if !name.chars().all(|c| c.is_alphanumeric() || c == '-') {
            return Err(AxonError::InvalidDomain(
                ".quid domains must be alphanumeric with hyphens only".to_string()
            ));
        }

        // Cannot start or end with hyphen
        if name.starts_with('-') || name.ends_with('-') {
            return Err(AxonError::InvalidDomain(
                ".quid domains cannot start or end with hyphen".to_string()
            ));
        }

        // Additional validation could include checking against identity fingerprint
        
        Ok(())
    }

    fn validate_axon_domain_name(&self, domain_name: &DomainName) -> Result<()> {
        let name = domain_name.as_str();
        
        // More flexible rules for .axon domains
        if name.len() < 2 || name.len() > 64 {
            return Err(AxonError::InvalidDomain(
                ".axon domains must be 2-64 characters".to_string()
            ));
        }

        // Allow alphanumeric, hyphens, and underscores
        if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            return Err(AxonError::InvalidDomain(
                ".axon domains must be alphanumeric with hyphens and underscores only".to_string()
            ));
        }

        Ok(())
    }

    fn calculate_axon_pricing(&self, domain_name: &DomainName) -> u64 {
        let name = domain_name.as_str();
        let mut price = self.config.axon_pricing.base_price;

        // Apply length multiplier
        if let Some(&multiplier) = self.config.axon_pricing.length_multiplier.get(&name.len()) {
            price = (price as f64 * multiplier) as u64;
        }

        // Check for premium keywords
        for (keyword, multiplier) in &self.config.axon_pricing.premium_keywords {
            if name.contains(keyword) {
                price = (price as f64 * multiplier) as u64;
                break;
            }
        }

        price
    }

    fn track_revenue(&mut self, domain_type: NymverseDomainType, amount: u64) {
        self.revenue_tracker.total_revenue += amount;
        *self.revenue_tracker.revenue_by_type.entry(domain_type).or_insert(0) += amount;

        // Distribute revenue according to configuration
        let distribution = &self.config.revenue_distribution;
        
        let dev_fund = (amount as f64 * distribution.development_fund) as u64;
        let ecosystem_fund = (amount as f64 * distribution.ecosystem_fund) as u64;
        let burn_amount = (amount as f64 * distribution.token_burn) as u64;
        let validator_rewards = (amount as f64 * distribution.validator_rewards) as u64;
        let creator_rewards = (amount as f64 * distribution.creator_rewards) as u64;

        *self.revenue_tracker.distributed_funds.entry("development".to_string()).or_insert(0) += dev_fund;
        *self.revenue_tracker.distributed_funds.entry("ecosystem".to_string()).or_insert(0) += ecosystem_fund;
        *self.revenue_tracker.distributed_funds.entry("validators".to_string()).or_insert(0) += validator_rewards;
        *self.revenue_tracker.distributed_funds.entry("creators".to_string()).or_insert(0) += creator_rewards;
        
        self.revenue_tracker.burned_tokens += burn_amount;

        debug!("Tracked revenue: {} NYM, burned: {} NYM", amount, burn_amount);
    }

    fn create_quid_registration_request(
        &self,
        domain_name: DomainName,
        quid_identity: NymIdentity,
        owner_key: AxonVerifyingKey,
        payment_proof: Vec<u8>,
    ) -> Result<DomainRegistrationRequest> {
        let temp_signing_key = axon_core::crypto::AxonSigningKey::generate();
        
        Ok(DomainRegistrationRequest {
            domain_name,
            domain_type: DomainType::Standard,
            duration_years: 1,
            owner: owner_key,
            initial_content_hash: axon_core::crypto::hash_content(b"quid_profile"),
            auto_renewal: Some(axon_core::domain::AutoRenewalConfig {
                enabled: true,
                renewal_duration_years: 1,
                max_renewal_price: 100_000, // Max price willing to pay for renewal
                funding_source: axon_core::domain::FundingSource::Account(owner_key),
            }),
            metadata: axon_core::domain::DomainMetadata {
                description: Some("QuID social profile domain".to_string()),
                keywords: vec!["quid".to_string(), "social".to_string(), "profile".to_string()],
                contact_info: None,
                category: Some("social".to_string()),
                reputation_score: 0,
                verification_status: axon_core::domain::VerificationStatus::Pending,
            },
            payment_proof,
            signature: temp_signing_key.sign(&[]), // Placeholder
        })
    }

    fn create_axon_registration_request(
        &self,
        domain_name: DomainName,
        quid_identity: NymIdentity,
        owner_key: AxonVerifyingKey,
        payment_proof: Vec<u8>,
        metadata: axon_core::domain::DomainMetadata,
        pricing: u64,
    ) -> Result<DomainRegistrationRequest> {
        let temp_signing_key = axon_core::crypto::AxonSigningKey::generate();
        
        Ok(DomainRegistrationRequest {
            domain_name,
            domain_type: DomainType::Standard,
            duration_years: 1,
            owner: owner_key,
            initial_content_hash: axon_core::crypto::hash_content(b"axon_content"),
            auto_renewal: Some(axon_core::domain::AutoRenewalConfig {
                enabled: false,
                renewal_duration_years: 1,
                max_renewal_price: 50_000, // Max price willing to pay for renewal
                funding_source: axon_core::domain::FundingSource::Account(owner_key),
            }),
            metadata,
            payment_proof,
            signature: temp_signing_key.sign(&[]), // Placeholder
        })
    }

    pub fn get_registry_statistics(&self) -> NymverseRegistryStats {
        let quid_stats = self.quid_registry.get_stats();
        let axon_stats = self.axon_registry.get_stats();

        NymverseRegistryStats {
            quid_domains: quid_stats.total_registered,
            axon_domains: axon_stats.total_registered,
            total_revenue: self.revenue_tracker.total_revenue,
            burned_tokens: self.revenue_tracker.burned_tokens,
            active_proposals: self.governance.active_proposals.len(),
            quid_identity_mapping_size: self.quid_identity_mapping.len(),
            revenue_distribution: self.revenue_tracker.distributed_funds.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NymverseRegistryStats {
    pub quid_domains: u64,
    pub axon_domains: u64,
    pub total_revenue: u64,
    pub burned_tokens: u64,
    pub active_proposals: usize,
    pub quid_identity_mapping_size: usize,
    pub revenue_distribution: HashMap<String, u64>,
}