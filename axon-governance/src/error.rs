//! Governance error types

use std::fmt;

#[derive(Debug, Clone)]
pub enum GovernanceError {
    ProposalNotFound(String),
    VotingPeriodEnded(String),
    InsufficientTokens(u64, u64), // required, actual
    AlreadyVoted(String),
    ProposalNotPassed(String),
    DelegationDisabled,
    InvalidProposal(String),
    VotingError(String),
    DelegationError(String),
    ConfigurationError(String),
    Other(String),
}

impl fmt::Display for GovernanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GovernanceError::ProposalNotFound(id) => write!(f, "Proposal not found: {}", id),
            GovernanceError::VotingPeriodEnded(id) => write!(f, "Voting period ended for proposal: {}", id),
            GovernanceError::InsufficientTokens(required, actual) => {
                write!(f, "Insufficient tokens: required {}, actual {}", required, actual)
            }
            GovernanceError::AlreadyVoted(id) => write!(f, "Already voted on proposal: {}", id),
            GovernanceError::ProposalNotPassed(id) => write!(f, "Proposal did not pass: {}", id),
            GovernanceError::DelegationDisabled => write!(f, "Delegation is disabled"),
            GovernanceError::InvalidProposal(reason) => write!(f, "Invalid proposal: {}", reason),
            GovernanceError::VotingError(msg) => write!(f, "Voting error: {}", msg),
            GovernanceError::DelegationError(msg) => write!(f, "Delegation error: {}", msg),
            GovernanceError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            GovernanceError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for GovernanceError {}

pub type GovernanceResult<T> = Result<T, GovernanceError>;