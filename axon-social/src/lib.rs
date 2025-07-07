//! Axon Social - Privacy-First Social Networking Features
//!
//! This crate provides social networking functionality for the Axon platform:
//! - Following/followers system with privacy preservation
//! - Content interaction system (replies, comments, likes)
//! - Feed generation with chronological and algorithmic ranking
//! - Privacy controls for all social features
//! - Anonymous engagement tracking
//!
//! All social features are designed with privacy-first principles:
//! - Zero-knowledge proofs for social connections
//! - Anonymous interactions with content
//! - Privacy-preserving analytics
//! - Optional content revelation mechanisms

pub mod error;
pub mod social_graph;
pub mod interactions;
pub mod feed;
pub mod privacy;
pub mod analytics;
pub mod advanced_search;

// Re-export main types
pub use error::{SocialError, SocialResult};
pub use social_graph::{SocialGraph, Connection, ConnectionType, FollowRequest};
pub use interactions::{Interaction, InteractionType, Reply, Like, Share, InteractionManager};
pub use feed::{Feed, FeedItem, FeedGenerator, FeedAlgorithm, RankingStrategy};
pub use privacy::{PrivacyLevel, PrivacyController, AnonymousProof, ProofType, SocialPrivacyManager};
pub use analytics::{EngagementMetrics, UserMetrics, ContentMetrics, AnonymousAnalytics};
pub use advanced_search::{AdvancedSearchEngine, SearchConfig, SearchResults, SearchResult, DiscoveryType};

/// Re-export commonly used types from dependencies
pub use axon_core::{
    identity::QuIDIdentity as Identity,
    content::{Post as Content, PostContent},
    ContentHash as ContentId,
    ContentType,
    DomainName,
    errors::{AxonError, Result as AxonResult}
};
pub use axon_identity::{
    auth_service::AuthenticationService as AuthService,
    IdentityManager
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration constants
pub mod defaults {
    /// Default maximum followers per user
    pub const MAX_FOLLOWERS: usize = 100_000;
    
    /// Default maximum following per user  
    pub const MAX_FOLLOWING: usize = 10_000;
    
    /// Default feed items per request
    pub const FEED_PAGE_SIZE: usize = 50;
    
    /// Default privacy level for new users
    pub const DEFAULT_PRIVACY_LEVEL: super::PrivacyLevel = super::PrivacyLevel::Anonymous;
    
    /// Default interaction retention period (days)
    pub const INTERACTION_RETENTION_DAYS: u32 = 365;
}