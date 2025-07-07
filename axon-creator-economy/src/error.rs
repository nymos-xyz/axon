//! Error types for the Creator Economy module

use std::fmt;

/// Creator Economy specific errors
#[derive(Debug, Clone)]
pub enum CreatorEconomyError {
    /// Payment processing failed
    PaymentFailed(String),
    /// Subscription management error
    SubscriptionError(String),
    /// Revenue distribution error
    RevenueError(String),
    /// Insufficient funds for operation
    InsufficientFunds(u64, u64), // required, available
    /// Invalid payment method
    InvalidPaymentMethod(String),
    /// Subscription not found
    SubscriptionNotFound(String),
    /// Creator not found
    CreatorNotFound(String),
    /// Invalid subscription tier
    InvalidSubscriptionTier(String),
    /// Payment verification failed
    PaymentVerificationFailed(String),
    /// Privacy proof verification failed
    PrivacyProofFailed(String),
    /// Campaign funding error
    FundingError(String),
    /// Analytics calculation error
    AnalyticsError(String),
    /// Configuration error
    ConfigError(String),
    /// Database operation failed
    DatabaseError(String),
    /// Network communication error
    NetworkError(String),
    /// Serialization/deserialization error
    SerializationError(String),
    /// Generic error with message
    Other(String),
}

impl fmt::Display for CreatorEconomyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CreatorEconomyError::PaymentFailed(msg) => write!(f, "Payment failed: {}", msg),
            CreatorEconomyError::SubscriptionError(msg) => write!(f, "Subscription error: {}", msg),
            CreatorEconomyError::RevenueError(msg) => write!(f, "Revenue error: {}", msg),
            CreatorEconomyError::InsufficientFunds(required, available) => {
                write!(f, "Insufficient funds: required {}, available {}", required, available)
            },
            CreatorEconomyError::InvalidPaymentMethod(method) => {
                write!(f, "Invalid payment method: {}", method)
            },
            CreatorEconomyError::SubscriptionNotFound(id) => {
                write!(f, "Subscription not found: {}", id)
            },
            CreatorEconomyError::CreatorNotFound(id) => {
                write!(f, "Creator not found: {}", id)
            },
            CreatorEconomyError::InvalidSubscriptionTier(tier) => {
                write!(f, "Invalid subscription tier: {}", tier)
            },
            CreatorEconomyError::PaymentVerificationFailed(reason) => {
                write!(f, "Payment verification failed: {}", reason)
            },
            CreatorEconomyError::PrivacyProofFailed(reason) => {
                write!(f, "Privacy proof verification failed: {}", reason)
            },
            CreatorEconomyError::FundingError(msg) => write!(f, "Funding error: {}", msg),
            CreatorEconomyError::AnalyticsError(msg) => write!(f, "Analytics error: {}", msg),
            CreatorEconomyError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            CreatorEconomyError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            CreatorEconomyError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            CreatorEconomyError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            CreatorEconomyError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for CreatorEconomyError {}

/// Result type for Creator Economy operations
pub type CreatorEconomyResult<T> = Result<T, CreatorEconomyError>;

/// Convert from other error types
impl From<std::io::Error> for CreatorEconomyError {
    fn from(err: std::io::Error) -> Self {
        CreatorEconomyError::Other(err.to_string())
    }
}

impl From<serde_json::Error> for CreatorEconomyError {
    fn from(err: serde_json::Error) -> Self {
        CreatorEconomyError::SerializationError(err.to_string())
    }
}