pub mod discovery_engine;
pub mod privacy_preserving;
pub mod recommendations;
pub mod interest_matching;
pub mod nymcompute_integration;
pub mod social_discovery;
pub mod error;
pub mod types;

pub use discovery_engine::DiscoveryEngine;
pub use privacy_preserving::PrivacyPreservingDiscovery;
pub use recommendations::RecommendationSystem;
pub use interest_matching::InterestMatcher;
pub use nymcompute_integration::NymComputeDiscovery;
pub use social_discovery::{
    SocialDiscoveryEngine, SocialDiscoveryConfig, UserRecommendation, 
    CommunityRecommendation, SocialDiscoveryRequest, SocialDiscoveryType
};
pub use error::{DiscoveryError, Result};
pub use types::*;