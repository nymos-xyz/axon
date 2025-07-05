use thiserror::Error;

pub type Result<T> = std::result::Result<T, DiscoveryError>;

#[derive(Error, Debug)]
pub enum DiscoveryError {
    #[error("Privacy violation detected: {0}")]
    PrivacyViolation(String),

    #[error("Anonymous discovery failed: {0}")]
    AnonymousDiscoveryFailed(String),

    #[error("Interest matching error: {0}")]
    InterestMatchingError(String),

    #[error("NymCompute integration error: {0}")]
    NymComputeError(String),

    #[error("Recommendation system error: {0}")]
    RecommendationError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}