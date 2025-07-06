//! AI system error types

use thiserror::Error;

/// Result type for AI operations
pub type AIResult<T> = Result<T, AIError>;

/// AI system errors
#[derive(Error, Debug, Clone)]
pub enum AIError {
    #[error("Model not found: {model_id}")]
    ModelNotFound { model_id: String },
    
    #[error("Invalid model format: {format}")]
    InvalidModelFormat { format: String },
    
    #[error("Training failed: {reason}")]
    TrainingFailed { reason: String },
    
    #[error("Inference failed: {reason}")]
    InferenceFailed { reason: String },
    
    #[error("Privacy budget exhausted: {used}/{total}")]
    PrivacyBudgetExhausted { used: f64, total: f64 },
    
    #[error("Insufficient anonymity set: {current}/{required}")]
    InsufficientAnonymitySet { current: usize, required: usize },
    
    #[error("Model too large: {size_mb}MB (max: {max_mb}MB)")]
    ModelTooLarge { size_mb: usize, max_mb: usize },
    
    #[error("Federated learning error: {0}")]
    FederatedLearningError(String),
    
    #[error("Privacy violation detected: {0}")]
    PrivacyViolation(String),
    
    #[error("Content moderation error: {0}")]
    ModerationError(String),
    
    #[error("Matching algorithm error: {0}")]
    MatchingError(String),
    
    #[error("Prediction error: {0}")]
    PredictionError(String),
    
    #[error("Data preprocessing error: {0}")]
    PreprocessingError(String),
    
    #[error("Model convergence failed: {0}")]
    ConvergenceError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    #[error("Internal AI error: {0}")]
    Internal(String),
}