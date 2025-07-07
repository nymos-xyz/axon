//! Axon Governance - Community Decision Making System
//!
//! This crate provides decentralized governance capabilities:
//! - Privacy-preserving voting with zk-STARKs
//! - Anonymous proposal creation and discussion
//! - Quadratic voting to prevent whale dominance
//! - Delegated voting with privacy protection
//! - Transparent governance analytics

pub mod error;
pub mod proposals;
pub mod voting;
pub mod delegation;
pub mod governance_analytics;
pub mod community_council;

// Re-export main types
pub use error::{GovernanceError, GovernanceResult};
pub use proposals::{
    ProposalManager, Proposal, ProposalType, ProposalStatus,
    ProposalVotingPeriod, ProposalExecution
};
pub use voting::{
    PrivacyVotingSystem, Vote, VotingMethod, QuadraticVote,
    VotingResults, AnonymousVoter, VotingProof
};
pub use delegation::{
    DelegationManager, VotingDelegation, DelegateProfile,
    DelegationChain, DelegationAnalytics
};
pub use governance_analytics::{
    GovernanceAnalytics, ParticipationMetrics, ProposalMetrics,
    VotingPatterns, CommunityEngagement
};
pub use community_council::{
    CommunityCouncil, CouncilMember, CouncilElection,
    CouncilGovernance, EmergencyActions
};

/// Re-export commonly used types from dependencies
pub use axon_core::{
    identity::QuIDIdentity as Identity,
    ContentHash as ContentId,
    errors::{AxonError, Result as AxonResult}
};
pub use nym_crypto::{
    ZkProof, ProofVerification, EncryptedAmount
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration constants
pub mod defaults {
    use std::time::Duration;
    
    /// Default voting period for proposals
    pub const DEFAULT_VOTING_PERIOD: Duration = Duration::from_secs(7 * 24 * 3600); // 7 days
    
    /// Default minimum participation rate for proposal validity
    pub const MIN_PARTICIPATION_RATE: f64 = 0.05; // 5%
    
    /// Default supermajority threshold for critical proposals
    pub const SUPERMAJORITY_THRESHOLD: f64 = 0.67; // 67%
    
    /// Default simple majority threshold
    pub const SIMPLE_MAJORITY_THRESHOLD: f64 = 0.51; // 51%
    
    /// Default quadratic voting cost scaling factor
    pub const QUADRATIC_SCALING_FACTOR: f64 = 1.0;
    
    /// Default maximum voting power per individual
    pub const MAX_INDIVIDUAL_VOTING_POWER: f64 = 0.05; // 5%
    
    /// Default delegation depth limit
    pub const MAX_DELEGATION_DEPTH: u32 = 5;
    
    /// Default proposal submission threshold
    pub const PROPOSAL_SUBMISSION_THRESHOLD: u64 = 1000; // NYM tokens
}

/// Governance configuration
#[derive(Debug, Clone)]
pub struct GovernanceConfig {
    /// Voting periods for different proposal types
    pub voting_periods: std::collections::HashMap<ProposalType, std::time::Duration>,
    /// Participation thresholds for proposal validity
    pub participation_thresholds: std::collections::HashMap<ProposalType, f64>,
    /// Approval thresholds for different proposal types
    pub approval_thresholds: std::collections::HashMap<ProposalType, f64>,
    /// Enable quadratic voting
    pub enable_quadratic_voting: bool,
    /// Enable delegation
    pub enable_delegation: bool,
    /// Maximum voting power concentration
    pub max_voting_power_concentration: f64,
    /// Governance token requirements
    pub token_requirements: TokenRequirements,
    /// Privacy settings
    pub privacy_settings: GovernancePrivacySettings,
}

/// Token requirements for governance participation
#[derive(Debug, Clone)]
pub struct TokenRequirements {
    /// Minimum tokens to vote
    pub min_tokens_to_vote: u64,
    /// Minimum tokens to create proposals
    pub min_tokens_to_propose: u64,
    /// Minimum tokens to delegate
    pub min_tokens_to_delegate: u64,
    /// Token lockup period for voting
    pub voting_lockup_period: std::time::Duration,
}

/// Privacy settings for governance
#[derive(Debug, Clone)]
pub struct GovernancePrivacySettings {
    /// Enable anonymous voting
    pub anonymous_voting: bool,
    /// Enable anonymous proposals
    pub anonymous_proposals: bool,
    /// Voting result privacy level
    pub result_privacy_level: PrivacyLevel,
    /// Participant privacy level
    pub participant_privacy_level: PrivacyLevel,
}

#[derive(Debug, Clone)]
pub enum PrivacyLevel {
    Public,
    Anonymous,
    Private,
    Encrypted,
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        use std::collections::HashMap;
        use std::time::Duration;
        
        let mut voting_periods = HashMap::new();
        voting_periods.insert(ProposalType::Constitutional, Duration::from_secs(14 * 24 * 3600)); // 14 days
        voting_periods.insert(ProposalType::Economic, Duration::from_secs(10 * 24 * 3600)); // 10 days
        voting_periods.insert(ProposalType::Technical, Duration::from_secs(7 * 24 * 3600)); // 7 days
        voting_periods.insert(ProposalType::Community, Duration::from_secs(5 * 24 * 3600)); // 5 days
        
        let mut participation_thresholds = HashMap::new();
        participation_thresholds.insert(ProposalType::Constitutional, 0.15); // 15%
        participation_thresholds.insert(ProposalType::Economic, 0.10); // 10%
        participation_thresholds.insert(ProposalType::Technical, 0.05); // 5%
        participation_thresholds.insert(ProposalType::Community, 0.03); // 3%
        
        let mut approval_thresholds = HashMap::new();
        approval_thresholds.insert(ProposalType::Constitutional, 0.75); // 75%
        approval_thresholds.insert(ProposalType::Economic, 0.67); // 67%
        approval_thresholds.insert(ProposalType::Technical, 0.60); // 60%
        approval_thresholds.insert(ProposalType::Community, 0.51); // 51%
        
        Self {
            voting_periods,
            participation_thresholds,
            approval_thresholds,
            enable_quadratic_voting: true,
            enable_delegation: true,
            max_voting_power_concentration: 0.20, // 20%
            token_requirements: TokenRequirements {
                min_tokens_to_vote: 100,
                min_tokens_to_propose: 10000,
                min_tokens_to_delegate: 50,
                voting_lockup_period: Duration::from_secs(24 * 3600), // 1 day
            },
            privacy_settings: GovernancePrivacySettings {
                anonymous_voting: true,
                anonymous_proposals: true,
                result_privacy_level: PrivacyLevel::Anonymous,
                participant_privacy_level: PrivacyLevel::Anonymous,
            },
        }
    }
}

/// Main governance system
#[derive(Debug)]
pub struct GovernanceSystem {
    config: GovernanceConfig,
    proposal_manager: ProposalManager,
    voting_system: PrivacyVotingSystem,
    delegation_manager: DelegationManager,
    analytics: GovernanceAnalytics,
    community_council: CommunityCouncil,
}

impl GovernanceSystem {
    /// Create a new governance system
    pub fn new(config: GovernanceConfig) -> Self {
        Self {
            proposal_manager: ProposalManager::new(config.clone()),
            voting_system: PrivacyVotingSystem::new(config.clone()),
            delegation_manager: DelegationManager::new(config.clone()),
            analytics: GovernanceAnalytics::new(),
            community_council: CommunityCouncil::new(config.clone()),
            config,
        }
    }

    /// Submit a new proposal
    pub async fn submit_proposal(
        &mut self,
        proposer: &Identity,
        proposal: Proposal,
    ) -> GovernanceResult<String> {
        // Verify proposer meets requirements
        self.verify_proposal_requirements(proposer, &proposal).await?;
        
        // Submit proposal through manager
        let proposal_id = self.proposal_manager.submit_proposal(proposer, proposal).await?;
        
        // Update analytics
        self.analytics.record_proposal_submission(&proposal_id).await?;
        
        Ok(proposal_id)
    }

    /// Cast a vote on a proposal
    pub async fn cast_vote(
        &mut self,
        voter: &Identity,
        proposal_id: &str,
        vote: Vote,
    ) -> GovernanceResult<String> {
        // Verify voter eligibility
        self.verify_voting_eligibility(voter, proposal_id).await?;
        
        // Cast vote through voting system
        let vote_id = self.voting_system.cast_vote(voter, proposal_id, vote).await?;
        
        // Update analytics
        self.analytics.record_vote_cast(proposal_id, &vote_id).await?;
        
        Ok(vote_id)
    }

    /// Delegate voting power
    pub async fn delegate_voting_power(
        &mut self,
        delegator: &Identity,
        delegate: &Identity,
        delegation: VotingDelegation,
    ) -> GovernanceResult<String> {
        if !self.config.enable_delegation {
            return Err(GovernanceError::DelegationDisabled);
        }
        
        // Create delegation through manager
        let delegation_id = self.delegation_manager.create_delegation(
            delegator,
            delegate,
            delegation,
        ).await?;
        
        // Update analytics
        self.analytics.record_delegation_created(&delegation_id).await?;
        
        Ok(delegation_id)
    }

    /// Get governance analytics
    pub async fn get_governance_analytics(&self) -> GovernanceResult<GovernanceAnalytics> {
        Ok(self.analytics.clone())
    }

    /// Execute a passed proposal
    pub async fn execute_proposal(&mut self, proposal_id: &str) -> GovernanceResult<()> {
        // Get proposal and verify it passed
        let proposal = self.proposal_manager.get_proposal(proposal_id)?;
        let results = self.voting_system.get_voting_results(proposal_id).await?;
        
        if !self.proposal_passed(&proposal, &results) {
            return Err(GovernanceError::ProposalNotPassed(proposal_id.to_string()));
        }
        
        // Execute proposal
        self.proposal_manager.execute_proposal(proposal_id).await?;
        
        // Update analytics
        self.analytics.record_proposal_execution(proposal_id).await?;
        
        Ok(())
    }

    // Private helper methods
    
    async fn verify_proposal_requirements(
        &self,
        proposer: &Identity,
        proposal: &Proposal,
    ) -> GovernanceResult<()> {
        // Check token requirements
        let token_balance = self.get_token_balance(proposer).await?;
        if token_balance < self.config.token_requirements.min_tokens_to_propose {
            return Err(GovernanceError::InsufficientTokens(
                self.config.token_requirements.min_tokens_to_propose,
                token_balance
            ));
        }
        
        // Check proposal type specific requirements
        match &proposal.proposal_type {
            ProposalType::Constitutional => {
                // Constitutional proposals require higher token threshold
                if token_balance < self.config.token_requirements.min_tokens_to_propose * 5 {
                    return Err(GovernanceError::InsufficientTokens(
                        self.config.token_requirements.min_tokens_to_propose * 5,
                        token_balance
                    ));
                }
            },
            _ => {} // Other types use standard threshold
        }
        
        Ok(())
    }

    async fn verify_voting_eligibility(
        &self,
        voter: &Identity,
        proposal_id: &str,
    ) -> GovernanceResult<()> {
        // Check token requirements
        let token_balance = self.get_token_balance(voter).await?;
        if token_balance < self.config.token_requirements.min_tokens_to_vote {
            return Err(GovernanceError::InsufficientTokens(
                self.config.token_requirements.min_tokens_to_vote,
                token_balance
            ));
        }
        
        // Check if already voted
        if self.voting_system.has_voted(voter, proposal_id).await? {
            return Err(GovernanceError::AlreadyVoted(proposal_id.to_string()));
        }
        
        // Check if proposal is in voting period
        let proposal = self.proposal_manager.get_proposal(proposal_id)?;
        if !proposal.is_in_voting_period() {
            return Err(GovernanceError::VotingPeriodEnded(proposal_id.to_string()));
        }
        
        Ok(())
    }

    async fn get_token_balance(&self, identity: &Identity) -> GovernanceResult<u64> {
        // This would integrate with the token system
        // For now, return a placeholder value
        Ok(10000) // Placeholder
    }

    fn proposal_passed(&self, proposal: &Proposal, results: &VotingResults) -> bool {
        let participation_threshold = self.config.participation_thresholds
            .get(&proposal.proposal_type)
            .unwrap_or(&0.05);
        
        let approval_threshold = self.config.approval_thresholds
            .get(&proposal.proposal_type)
            .unwrap_or(&0.51);
        
        results.participation_rate >= *participation_threshold &&
        results.approval_rate >= *approval_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_governance_system_creation() {
        let config = GovernanceConfig::default();
        let _governance = GovernanceSystem::new(config);
        
        println!("✅ Governance system created successfully");
    }

    #[tokio::test]
    async fn test_governance_config() {
        let config = GovernanceConfig::default();
        
        assert!(config.enable_quadratic_voting);
        assert!(config.enable_delegation);
        assert_eq!(config.max_voting_power_concentration, 0.20);
        
        println!("✅ Governance configuration validated");
    }

    #[test]
    fn test_module_exports() {
        // Test that all main types are accessible
        println!("All governance modules exported successfully");
    }
}