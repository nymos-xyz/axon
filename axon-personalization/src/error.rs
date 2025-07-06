//! Personalization system error types

use thiserror::Error;

/// Result type for personalization operations
pub type PersonalizationResult<T> = Result<T, PersonalizationError>;

/// Personalization system errors
#[derive(Error, Debug, Clone)]
pub enum PersonalizationError {
    #[error("Invalid user profile: {0}")]
    InvalidUserProfile(String),
    
    #[error("Profile too large: {dimensions} dimensions (max: {max})")]
    ProfileTooLarge { dimensions: usize, max: usize },
    
    #[error("Insufficient interaction data: minimum {min_interactions} required")]
    InsufficientData { min_interactions: usize },
    
    #[error("Privacy violation: {0}")]
    PrivacyViolation(String),
    
    #[error("Privacy budget exhausted: used {used}/{total}")]
    PrivacyBudgetExhausted { used: f64, total: f64 },
    
    #[error("Model training failed: {0}")]
    ModelTrainingFailed(String),
    
    #[error("Federation error: {0}")]
    FederationError(String),
    
    #[error("Recommendation generation failed: {0}")]
    RecommendationFailed(String),
    
    #[error("Content filtering failed: {0}")]
    FilteringFailed(String),
    
    #[error("Cryptographic error: {0}")]
    CryptographicError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}