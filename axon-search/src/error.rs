//! Search system error types

use thiserror::Error;

/// Result type for search operations
pub type SearchResult<T> = Result<T, SearchError>;

/// Search system errors
#[derive(Error, Debug, Clone)]
pub enum SearchError {
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    
    #[error("Query too long: {length} characters (max: {max})")]
    QueryTooLong { length: usize, max: usize },
    
    #[error("Search timeout: operation took longer than {timeout_seconds}s")]
    SearchTimeout { timeout_seconds: u64 },
    
    #[error("Index error: {0}")]
    IndexError(String),
    
    #[error("Privacy violation: {0}")]
    PrivacyViolation(String),
    
    #[error("Insufficient permissions: {0}")]
    InsufficientPermissions(String),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    
    #[error("Index not found: {index_name}")]
    IndexNotFound { index_name: String },
    
    #[error("Shard unavailable: {shard_id}")]
    ShardUnavailable { shard_id: String },
    
    #[error("Cryptographic error: {0}")]
    CryptographicError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}