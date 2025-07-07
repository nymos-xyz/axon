//! Proposal Management System

use crate::error::{GovernanceError, GovernanceResult};
use crate::{Identity, GovernanceConfig};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug)]
pub struct ProposalManager {
    proposals: HashMap<String, Proposal>,
    config: GovernanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub proposal_id: String,
    pub title: String,
    pub description: String,
    pub proposal_type: ProposalType,
    pub proposer_id: Option<String>, // None for anonymous
    pub voting_period: ProposalVotingPeriod,
    pub status: ProposalStatus,
    pub execution_details: Option<ProposalExecution>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProposalType {
    Constitutional,
    Economic,
    Technical,
    Community,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalVotingPeriod {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub extended_until: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalStatus {
    Draft,
    Active,
    Passed,
    Failed,
    Executed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalExecution {
    pub execution_type: ExecutionType,
    pub execution_data: Vec<u8>,
    pub executed_at: Option<DateTime<Utc>>,
    pub execution_result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionType {
    ParameterChange,
    ContractUpgrade,
    FundTransfer,
    PolicyUpdate,
    ConfigurationChange,
}

impl ProposalManager {
    pub fn new(config: GovernanceConfig) -> Self {
        Self {
            proposals: HashMap::new(),
            config,
        }
    }

    pub async fn submit_proposal(&mut self, proposer: &Identity, mut proposal: Proposal) -> GovernanceResult<String> {
        // Set proposer (or keep anonymous)
        if !self.config.privacy_settings.anonymous_proposals {
            proposal.proposer_id = Some(proposer.get_id());
        }

        // Set voting period based on proposal type
        let voting_duration = self.config.voting_periods.get(&proposal.proposal_type)
            .cloned()
            .unwrap_or(std::time::Duration::from_secs(7 * 24 * 3600));

        proposal.voting_period = ProposalVotingPeriod {
            start_time: Utc::now(),
            end_time: Utc::now() + chrono::Duration::from_std(voting_duration).unwrap(),
            extended_until: None,
        };

        proposal.status = ProposalStatus::Active;
        let proposal_id = proposal.proposal_id.clone();
        
        self.proposals.insert(proposal_id.clone(), proposal);
        Ok(proposal_id)
    }

    pub fn get_proposal(&self, proposal_id: &str) -> GovernanceResult<&Proposal> {
        self.proposals.get(proposal_id)
            .ok_or_else(|| GovernanceError::ProposalNotFound(proposal_id.to_string()))
    }

    pub async fn execute_proposal(&mut self, proposal_id: &str) -> GovernanceResult<()> {
        let proposal = self.proposals.get_mut(proposal_id)
            .ok_or_else(|| GovernanceError::ProposalNotFound(proposal_id.to_string()))?;

        proposal.status = ProposalStatus::Executed;
        if let Some(ref mut execution) = proposal.execution_details {
            execution.executed_at = Some(Utc::now());
            execution.execution_result = Some("Executed successfully".to_string());
        }

        Ok(())
    }
}

impl Proposal {
    pub fn is_in_voting_period(&self) -> bool {
        let now = Utc::now();
        let end_time = self.voting_period.extended_until
            .unwrap_or(self.voting_period.end_time);
        
        now >= self.voting_period.start_time && now <= end_time
    }
}