//! Advanced governance smart contracts for Axon protocol
//! 
//! Provides decentralized governance with:
//! - Proposal creation and voting
//! - Quadratic voting to prevent whale dominance
//! - Time-locked execution for security
//! - Emergency governance for critical issues
//! - Delegation and proxy voting

use axon_core::{
    crypto::AxonVerifyingKey,
    types::Timestamp,
    Result, AxonError,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{info, warn, error};

/// Advanced governance smart contract
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernanceContract {
    /// Active proposals
    pub proposals: HashMap<u64, Proposal>,
    /// Voting records
    pub votes: HashMap<u64, VotingRecord>,
    /// Governance configuration
    pub config: GovernanceConfig,
    /// Delegation mappings
    pub delegations: HashMap<AxonVerifyingKey, Delegation>,
    /// Execution queue for passed proposals
    pub execution_queue: Vec<ExecutionItem>,
    /// Proposal counter
    pub next_proposal_id: u64,
    /// Contract creation timestamp
    pub created_at: Timestamp,
    /// Emergency multisig members
    pub emergency_council: HashSet<AxonVerifyingKey>,
}

/// Governance configuration parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernanceConfig {
    /// Minimum tokens required to create proposal
    pub proposal_threshold: u64,
    /// Minimum participation rate for valid vote
    pub quorum_threshold: f64,
    /// Voting period duration in seconds
    pub voting_period: u64,
    /// Time lock period before execution
    pub timelock_period: u64,
    /// Maximum proposal execution window
    pub execution_window: u64,
    /// Enable quadratic voting
    pub quadratic_voting: bool,
    /// Maximum voting power per user (prevents whale dominance)
    pub max_voting_power: u64,
    /// Delegation fee percentage
    pub delegation_fee: f64,
}

/// Individual proposal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub id: u64,
    pub proposer: AxonVerifyingKey,
    pub title: String,
    pub description: String,
    pub proposal_type: ProposalType,
    pub actions: Vec<ProposalAction>,
    pub created_at: Timestamp,
    pub voting_starts_at: Timestamp,
    pub voting_ends_at: Timestamp,
    pub execution_eta: Option<Timestamp>,
    pub status: ProposalStatus,
    pub metadata: ProposalMetadata,
}

/// Types of proposals
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProposalType {
    /// Parameter changes (pricing, fees, etc.)
    ParameterUpdate,
    /// Protocol upgrades
    ProtocolUpgrade,
    /// Treasury spending
    TreasurySpending,
    /// Emergency actions
    Emergency,
    /// Governance rule changes
    GovernanceUpdate,
    /// Token economics changes
    TokenomicsUpdate,
}

/// Specific actions to execute if proposal passes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProposalAction {
    UpdatePricing {
        contract_address: String,
        new_parameters: HashMap<String, String>,
    },
    TransferFunds {
        to: AxonVerifyingKey,
        amount: u64,
        reason: String,
    },
    UpdateContract {
        contract_address: String,
        new_code_hash: String,
    },
    BurnTokens {
        amount: u64,
        reason: String,
    },
    EmergencyPause {
        contracts: Vec<String>,
        duration: u64,
    },
}

/// Proposal execution status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ProposalStatus {
    Pending,
    Active,
    Passed,
    Failed,
    Executed,
    Cancelled,
    Expired,
}

/// Additional proposal metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalMetadata {
    pub category: String,
    pub urgency: UrgencyLevel,
    pub estimated_impact: ImpactLevel,
    pub required_expertise: Vec<String>,
    pub external_links: Vec<String>,
    pub discussion_forum: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Systemic,
}

/// Voting record for a proposal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VotingRecord {
    pub proposal_id: u64,
    pub votes_for: u64,
    pub votes_against: u64,
    pub votes_abstain: u64,
    pub total_voting_power: u64,
    pub unique_voters: u32,
    pub voter_details: HashMap<AxonVerifyingKey, Vote>,
    pub quadratic_adjusted_for: u64,
    pub quadratic_adjusted_against: u64,
}

/// Individual vote
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vote {
    pub voter: AxonVerifyingKey,
    pub choice: VoteChoice,
    pub voting_power: u64,
    pub quadratic_power: u64,
    pub timestamp: Timestamp,
    pub is_delegated: bool,
    pub delegate: Option<AxonVerifyingKey>,
    pub reasoning: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}

/// Vote delegation system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Delegation {
    pub delegator: AxonVerifyingKey,
    pub delegate: AxonVerifyingKey,
    pub voting_power: u64,
    pub categories: Vec<ProposalType>, // Can specify delegation by category
    pub created_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub fee_paid: u64,
}

/// Item in execution queue
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionItem {
    pub proposal_id: u64,
    pub scheduled_execution: Timestamp,
    pub actions: Vec<ProposalAction>,
    pub executor: Option<AxonVerifyingKey>,
}

/// Governance events
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GovernanceEvent {
    ProposalCreated {
        proposal_id: u64,
        proposer: AxonVerifyingKey,
        title: String,
        proposal_type: ProposalType,
    },
    VoteCast {
        proposal_id: u64,
        voter: AxonVerifyingKey,
        choice: VoteChoice,
        voting_power: u64,
    },
    ProposalPassed {
        proposal_id: u64,
        votes_for: u64,
        votes_against: u64,
        execution_eta: Timestamp,
    },
    ProposalExecuted {
        proposal_id: u64,
        executor: AxonVerifyingKey,
        timestamp: Timestamp,
    },
    DelegationCreated {
        delegator: AxonVerifyingKey,
        delegate: AxonVerifyingKey,
        voting_power: u64,
    },
    EmergencyAction {
        action: String,
        executor: AxonVerifyingKey,
        reason: String,
    },
}

impl GovernanceContract {
    /// Create new governance contract
    pub fn new(config: GovernanceConfig, emergency_council: HashSet<AxonVerifyingKey>) -> Self {
        Self {
            proposals: HashMap::new(),
            votes: HashMap::new(),
            config,
            delegations: HashMap::new(),
            execution_queue: Vec::new(),
            next_proposal_id: 1,
            created_at: Timestamp::now(),
            emergency_council,
        }
    }

    /// Create a new proposal
    pub fn create_proposal(
        &mut self,
        proposer: AxonVerifyingKey,
        title: String,
        description: String,
        proposal_type: ProposalType,
        actions: Vec<ProposalAction>,
        metadata: ProposalMetadata,
        proposer_voting_power: u64,
    ) -> Result<GovernanceEvent> {
        // Check if proposer has enough voting power
        if proposer_voting_power < self.config.proposal_threshold {
            return Err(AxonError::PermissionDenied);
        }

        let proposal_id = self.next_proposal_id;
        self.next_proposal_id += 1;

        let now = Timestamp::now();
        let voting_delay = if matches!(proposal_type, ProposalType::Emergency) {
            3600 // 1 hour for emergency proposals
        } else {
            86400 // 24 hours for regular proposals
        };

        let proposal = Proposal {
            id: proposal_id,
            proposer: proposer.clone(),
            title: title.clone(),
            description,
            proposal_type: proposal_type.clone(),
            actions,
            created_at: now,
            voting_starts_at: Timestamp(now.0 + voting_delay),
            voting_ends_at: Timestamp(now.0 + voting_delay + self.config.voting_period),
            execution_eta: None,
            status: ProposalStatus::Pending,
            metadata,
        };

        // Initialize voting record
        let voting_record = VotingRecord {
            proposal_id,
            votes_for: 0,
            votes_against: 0,
            votes_abstain: 0,
            total_voting_power: 0,
            unique_voters: 0,
            voter_details: HashMap::new(),
            quadratic_adjusted_for: 0,
            quadratic_adjusted_against: 0,
        };

        self.proposals.insert(proposal_id, proposal);
        self.votes.insert(proposal_id, voting_record);

        info!("Created proposal {} by {}", proposal_id, proposer.to_string());

        Ok(GovernanceEvent::ProposalCreated {
            proposal_id,
            proposer,
            title,
            proposal_type,
        })
    }

    /// Cast a vote on a proposal
    pub fn vote_on_proposal(
        &mut self,
        proposal_id: u64,
        voter: AxonVerifyingKey,
        choice: VoteChoice,
        voting_power: u64,
        reasoning: Option<String>,
    ) -> Result<GovernanceEvent> {
        let proposal = self.proposals.get_mut(&proposal_id)
            .ok_or(AxonError::InvalidDomain("Proposal not found".to_string()))?;

        let now = Timestamp::now();

        // Check voting period
        if now.0 < proposal.voting_starts_at.0 {
            return Err(AxonError::InvalidDomain("Voting not yet started".to_string()));
        }
        if now.0 > proposal.voting_ends_at.0 {
            return Err(AxonError::InvalidDomain("Voting period ended".to_string()));
        }

        // Update proposal status if needed
        if proposal.status == ProposalStatus::Pending {
            proposal.status = ProposalStatus::Active;
        }

        let voting_record = self.votes.get_mut(&proposal_id)
            .ok_or(AxonError::Internal("Voting record not found".to_string()))?;

        // Check if already voted
        if voting_record.voter_details.contains_key(&voter) {
            return Err(AxonError::InvalidDomain("Already voted".to_string()));
        }

        // Apply maximum voting power limit
        let effective_voting_power = voting_power.min(self.config.max_voting_power);

        // Calculate quadratic voting power if enabled
        let quadratic_power = if self.config.quadratic_voting {
            (effective_voting_power as f64).sqrt() as u64
        } else {
            effective_voting_power
        };

        // Check for delegation
        let (is_delegated, delegate) = if let Some(delegation) = self.delegations.get(&voter) {
            if delegation.categories.is_empty() || delegation.categories.contains(&proposal.proposal_type) {
                (true, Some(delegation.delegate.clone()))
            } else {
                (false, None)
            }
        } else {
            (false, None)
        };

        let vote = Vote {
            voter: voter.clone(),
            choice: choice.clone(),
            voting_power: effective_voting_power,
            quadratic_power,
            timestamp: now,
            is_delegated,
            delegate,
            reasoning,
        };

        // Update vote tallies
        match choice {
            VoteChoice::For => {
                voting_record.votes_for += effective_voting_power;
                if self.config.quadratic_voting {
                    voting_record.quadratic_adjusted_for += quadratic_power;
                }
            }
            VoteChoice::Against => {
                voting_record.votes_against += effective_voting_power;
                if self.config.quadratic_voting {
                    voting_record.quadratic_adjusted_against += quadratic_power;
                }
            }
            VoteChoice::Abstain => {
                voting_record.votes_abstain += effective_voting_power;
            }
        }

        voting_record.total_voting_power += effective_voting_power;
        voting_record.unique_voters += 1;
        voting_record.voter_details.insert(voter.clone(), vote);

        info!("Vote cast on proposal {} by {}", proposal_id, voter.to_string());

        Ok(GovernanceEvent::VoteCast {
            proposal_id,
            voter,
            choice,
            voting_power: effective_voting_power,
        })
    }

    /// Finalize proposal after voting period ends
    pub fn finalize_proposal(&mut self, proposal_id: u64) -> Result<GovernanceEvent> {
        let proposal = self.proposals.get_mut(&proposal_id)
            .ok_or(AxonError::InvalidDomain("Proposal not found".to_string()))?;

        let now = Timestamp::now();
        if now.0 <= proposal.voting_ends_at.0 {
            return Err(AxonError::InvalidDomain("Voting period still active".to_string()));
        }

        let voting_record = self.votes.get(&proposal_id)
            .ok_or(AxonError::Internal("Voting record not found".to_string()))?;

        // Check quorum
        let total_supply = 1_000_000_000u64; // Would be fetched from token contract
        let participation_rate = voting_record.total_voting_power as f64 / total_supply as f64;
        
        if participation_rate < self.config.quorum_threshold {
            proposal.status = ProposalStatus::Failed;
            warn!("Proposal {} failed due to insufficient quorum", proposal_id);
            return Ok(GovernanceEvent::ProposalExecuted {
                proposal_id,
                executor: AxonVerifyingKey::from([0u8; 32]), // System
                timestamp: now,
            });
        }

        // Determine result based on voting method
        let (for_votes, against_votes) = if self.config.quadratic_voting {
            (voting_record.quadratic_adjusted_for, voting_record.quadratic_adjusted_against)
        } else {
            (voting_record.votes_for, voting_record.votes_against)
        };

        if for_votes > against_votes {
            proposal.status = ProposalStatus::Passed;
            let execution_eta = Timestamp(now.0 + self.config.timelock_period);
            proposal.execution_eta = Some(execution_eta);

            // Add to execution queue
            self.execution_queue.push(ExecutionItem {
                proposal_id,
                scheduled_execution: execution_eta,
                actions: proposal.actions.clone(),
                executor: None,
            });

            info!("Proposal {} passed and scheduled for execution", proposal_id);

            Ok(GovernanceEvent::ProposalPassed {
                proposal_id,
                votes_for: for_votes,
                votes_against: against_votes,
                execution_eta,
            })
        } else {
            proposal.status = ProposalStatus::Failed;
            info!("Proposal {} failed", proposal_id);
            Ok(GovernanceEvent::ProposalExecuted {
                proposal_id,
                executor: AxonVerifyingKey::from([0u8; 32]), // System
                timestamp: now,
            })
        }
    }

    /// Execute a passed proposal
    pub fn execute_proposal(
        &mut self,
        proposal_id: u64,
        executor: AxonVerifyingKey,
    ) -> Result<GovernanceEvent> {
        let proposal = self.proposals.get_mut(&proposal_id)
            .ok_or(AxonError::InvalidDomain("Proposal not found".to_string()))?;

        if proposal.status != ProposalStatus::Passed {
            return Err(AxonError::InvalidDomain("Proposal not in passed state".to_string()));
        }

        let now = Timestamp::now();
        if let Some(eta) = proposal.execution_eta {
            if now.0 < eta.0 {
                return Err(AxonError::InvalidDomain("Timelock period not yet passed".to_string()));
            }
            if now.0 > eta.0 + self.config.execution_window {
                proposal.status = ProposalStatus::Expired;
                return Err(AxonError::InvalidDomain("Execution window expired".to_string()));
            }
        }

        // Execute actions (placeholder - would integrate with actual contracts)
        for action in &proposal.actions {
            self.execute_action(action)?;
        }

        proposal.status = ProposalStatus::Executed;

        // Remove from execution queue
        self.execution_queue.retain(|item| item.proposal_id != proposal_id);

        info!("Executed proposal {} by {}", proposal_id, executor.to_string());

        Ok(GovernanceEvent::ProposalExecuted {
            proposal_id,
            executor,
            timestamp: now,
        })
    }

    /// Delegate voting power
    pub fn delegate_voting_power(
        &mut self,
        delegator: AxonVerifyingKey,
        delegate: AxonVerifyingKey,
        voting_power: u64,
        categories: Vec<ProposalType>,
        expires_at: Option<Timestamp>,
    ) -> Result<GovernanceEvent> {
        let fee = (voting_power as f64 * self.config.delegation_fee) as u64;
        
        let delegation = Delegation {
            delegator: delegator.clone(),
            delegate: delegate.clone(),
            voting_power,
            categories,
            created_at: Timestamp::now(),
            expires_at,
            fee_paid: fee,
        };

        self.delegations.insert(delegator.clone(), delegation);

        Ok(GovernanceEvent::DelegationCreated {
            delegator,
            delegate,
            voting_power,
        })
    }

    /// Emergency action by council
    pub fn emergency_action(
        &mut self,
        executor: AxonVerifyingKey,
        action: ProposalAction,
        reason: String,
    ) -> Result<GovernanceEvent> {
        if !self.emergency_council.contains(&executor) {
            return Err(AxonError::PermissionDenied);
        }

        // Execute emergency action immediately
        self.execute_action(&action)?;

        warn!("Emergency action executed by {}: {}", executor.to_string(), reason);

        Ok(GovernanceEvent::EmergencyAction {
            action: format!("{:?}", action),
            executor,
            reason,
        })
    }

    /// Execute a specific action
    fn execute_action(&self, action: &ProposalAction) -> Result<()> {
        match action {
            ProposalAction::UpdatePricing { contract_address, new_parameters } => {
                info!("Would update pricing for contract {} with params {:?}", contract_address, new_parameters);
                // Placeholder - would call actual contract
            }
            ProposalAction::TransferFunds { to, amount, reason } => {
                info!("Would transfer {} tokens to {} for: {}", amount, to.to_string(), reason);
                // Placeholder - would execute transfer
            }
            ProposalAction::UpdateContract { contract_address, new_code_hash } => {
                info!("Would update contract {} to code hash {}", contract_address, new_code_hash);
                // Placeholder - would update contract
            }
            ProposalAction::BurnTokens { amount, reason } => {
                info!("Would burn {} tokens for: {}", amount, reason);
                // Placeholder - would burn tokens
            }
            ProposalAction::EmergencyPause { contracts, duration } => {
                info!("Would pause contracts {:?} for {} seconds", contracts, duration);
                // Placeholder - would pause contracts
            }
        }
        Ok(())
    }

    /// Get proposal information
    pub fn get_proposal(&self, proposal_id: u64) -> Option<&Proposal> {
        self.proposals.get(&proposal_id)
    }

    /// Get voting record
    pub fn get_voting_record(&self, proposal_id: u64) -> Option<&VotingRecord> {
        self.votes.get(&proposal_id)
    }

    /// Get active proposals
    pub fn get_active_proposals(&self) -> Vec<&Proposal> {
        self.proposals.values()
            .filter(|p| matches!(p.status, ProposalStatus::Active | ProposalStatus::Pending))
            .collect()
    }

    /// Get governance statistics
    pub fn get_governance_stats(&self) -> GovernanceStats {
        let mut status_counts = HashMap::new();
        for proposal in self.proposals.values() {
            *status_counts.entry(proposal.status.clone()).or_insert(0) += 1;
        }

        GovernanceStats {
            total_proposals: self.proposals.len(),
            status_breakdown: status_counts,
            total_delegations: self.delegations.len(),
            emergency_council_size: self.emergency_council.len(),
            pending_executions: self.execution_queue.len(),
            config: self.config.clone(),
        }
    }
}

/// Governance statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernanceStats {
    pub total_proposals: usize,
    pub status_breakdown: HashMap<ProposalStatus, u32>,
    pub total_delegations: usize,
    pub emergency_council_size: usize,
    pub pending_executions: usize,
    pub config: GovernanceConfig,
}

impl Default for GovernanceContract {
    fn default() -> Self {
        Self::new(
            GovernanceConfig {
                proposal_threshold: 10_000,
                quorum_threshold: 0.1, // 10% participation
                voting_period: 604800,  // 1 week
                timelock_period: 172800, // 48 hours
                execution_window: 259200, // 72 hours
                quadratic_voting: true,
                max_voting_power: 100_000,
                delegation_fee: 0.01, // 1%
            },
            HashSet::new(),
        )
    }
}