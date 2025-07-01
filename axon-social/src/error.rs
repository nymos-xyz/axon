//! Error types for Axon social networking features

use thiserror::Error;

/// Social networking error types
#[derive(Error, Debug)]
pub enum SocialError {
    #[error("Social graph error: {0}")]
    SocialGraph(String),
    
    #[error("Privacy violation: {0}")]
    PrivacyViolation(String),
    
    #[error("Interaction failed: {0}")]
    InteractionFailed(String),
    
    #[error("Feed generation failed: {0}")]
    FeedGenerationFailed(String),
    
    #[error("Analytics error: {0}")]
    Analytics(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("User not found: {0}")]
    UserNotFound(String),
    
    #[error("Content not found: {0}")]
    ContentNotFound(String),
    
    #[error("Connection limit exceeded: max {max}, current {current}")]
    ConnectionLimitExceeded { max: usize, current: usize },
    
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error("Proof verification failed: {0}")]
    ProofVerificationFailed(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Authentication error: {0}")]
    AuthenticationError(String),
    
    #[error("Core error: {0}")]
    CoreError(#[from] axon_core::AxonError),
    
    #[error("Identity error: {0}")]
    IdentityError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for social operations
pub type SocialResult<T> = Result<T, SocialError>;

impl From<serde_json::Error> for SocialError {
    fn from(error: serde_json::Error) -> Self {
        SocialError::SerializationError(error.to_string())
    }
}

impl From<bincode::Error> for SocialError {
    fn from(error: bincode::Error) -> Self {
        SocialError::SerializationError(error.to_string())
    }
}