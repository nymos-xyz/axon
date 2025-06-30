//! # Axon Core
//! 
//! Core protocol types, traits, and constants for the Axon decentralized social network.
//! This crate provides the fundamental data structures and cryptographic primitives.

pub mod types;
pub mod crypto;
pub mod domain;
pub mod content;
pub mod identity;
pub mod errors;

pub use types::*;
pub use errors::{AxonError, Result};

/// Protocol version for compatibility checks
pub const PROTOCOL_VERSION: u32 = 1;

/// Maximum content size in bytes (16MB)
pub const MAX_CONTENT_SIZE: usize = 16 * 1024 * 1024;

/// Maximum domain name length
pub const MAX_DOMAIN_LENGTH: usize = 253;

/// Minimum domain name length  
pub const MIN_DOMAIN_LENGTH: usize = 3;

/// Default content cache duration in seconds (1 hour)
pub const DEFAULT_CACHE_DURATION: u64 = 3600;