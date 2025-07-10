//! Error types for Axon protocol

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AxonError {
    #[error("Invalid domain: {0}")]
    InvalidDomain(String),
    
    #[error("Invalid content: {0}")]
    InvalidContent(String),
    
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    
    #[error("Identity error: {0}")]
    IdentityError(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Authentication failed")]
    AuthenticationFailed,
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("Content not found")]
    ContentNotFound,
    
    #[error("Domain not found")]
    DomainNotFound,
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Invalid signature")]
    InvalidSignature,
    
    #[error("Protocol version mismatch")]
    VersionMismatch,
    
    #[error("Privacy error: {0}")]
    Privacy(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, AxonError>;