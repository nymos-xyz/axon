//! # Axon Identity
//! 
//! QuID identity integration and management for the Axon protocol.
//! Provides identity verification, authentication, and privacy-preserving
//! identity operations.

pub mod quid_integration;
pub mod auth_service;
pub mod identity_manager;
pub mod verification;

pub use quid_integration::QuIDIntegration;
pub use auth_service::AuthenticationService;
pub use identity_manager::IdentityManager;
pub use verification::IdentityVerificationService;