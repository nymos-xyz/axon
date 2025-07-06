//! Media system error types

use thiserror::Error;

/// Result type for media operations
pub type MediaResult<T> = Result<T, MediaError>;

/// Media system errors
#[derive(Error, Debug, Clone)]
pub enum MediaError {
    #[error("Stream not found: {stream_id}")]
    StreamNotFound { stream_id: String },
    
    #[error("Invalid media format: {format}")]
    InvalidFormat { format: String },
    
    #[error("Transcoding failed: {reason}")]
    TranscodingFailed { reason: String },
    
    #[error("Stream capacity exceeded: {current}/{max}")]
    CapacityExceeded { current: usize, max: usize },
    
    #[error("Bitrate too high: {bitrate} bps (max: {max_bitrate} bps)")]
    BitrateExceeded { bitrate: u64, max_bitrate: u64 },
    
    #[error("Recording error: {0}")]
    RecordingError(String),
    
    #[error("Playback error: {0}")]
    PlaybackError(String),
    
    #[error("Authentication required for stream: {stream_id}")]
    AuthenticationRequired { stream_id: String },
    
    #[error("Privacy violation: {0}")]
    PrivacyViolation(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Codec error: {codec}: {reason}")]
    CodecError { codec: String, reason: String },
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Protocol error: {protocol}: {reason}")]
    ProtocolError { protocol: String, reason: String },
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    #[error("Internal error: {0}")]
    Internal(String),
}