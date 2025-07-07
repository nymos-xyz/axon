//! Axon Creator Economy - Privacy-Preserving Monetization Platform
//!
//! This crate provides anonymous monetization capabilities for content creators:
//! - Anonymous subscription management with zk-STARKs
//! - Privacy-preserving payment processing
//! - Creator revenue distribution without tracking
//! - Anonymous tipping and donations
//! - Privacy-first creator analytics
//! - Community funding and crowdfunding mechanisms

pub mod error;
pub mod subscriptions;
pub mod payments;
pub mod revenue;
pub mod analytics;
pub mod funding;
pub mod creator_tools;
pub mod anonymous_economy;

// Re-export main types
pub use error::{CreatorEconomyError, CreatorEconomyResult};
pub use subscriptions::{
    AnonymousSubscriptionManager, SubscriptionTier, SubscriptionPayment,
    SubscriptionAnalytics, SubscriptionContract
};
pub use payments::{
    PrivacyPaymentProcessor, PaymentMethod, AnonymousPayment,
    PaymentAnalytics, PaymentVerification
};
pub use revenue::{
    RevenueDistributor, RevenueShare, RevenueStream,
    AnonymousRevenue, RevenueAnalytics
};
pub use analytics::{
    CreatorAnalytics, AnonymousCreatorMetrics, EngagementMetrics,
    MonetizationMetrics, PrivacyPreservingAnalytics
};
pub use funding::{
    CommunityFunding, CrowdfundingCampaign, AnonymousBacker,
    FundingGoal, CampaignAnalytics
};
pub use creator_tools::{
    CreatorDashboard, ContentMonetization, AudienceInsights,
    CreatorSupport, PrivacySettings
};
pub use anonymous_economy::{
    AnonymousCreatorEconomyEngine, AnonymousEconomyConfig, AnonymousCreatorId,
    AnonymousSubscription, AnonymousTip, RevenueDistribution, CreatorTier,
    RevenuePrivacyLevel, CreatorProfile, SubscriberProfile, EconomyReport
};

/// Re-export commonly used types from dependencies
pub use axon_core::{
    identity::QuIDIdentity as Identity,
    content::{Post as Content, PostContent},
    ContentHash as ContentId,
    ContentType,
    DomainName,
    errors::{AxonError, Result as AxonResult}
};
pub use nym_crypto::{
    EncryptedAmount, PrivateKey, PublicKey,
    ZkProof, ProofVerification
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration constants
pub mod defaults {
    /// Default subscription tier price in NYM tokens
    pub const DEFAULT_SUBSCRIPTION_PRICE: u64 = 1000;
    
    /// Default creator revenue share percentage
    pub const DEFAULT_CREATOR_SHARE: f64 = 0.85;
    
    /// Default platform fee percentage
    pub const DEFAULT_PLATFORM_FEE: f64 = 0.10;
    
    /// Default network fee percentage
    pub const DEFAULT_NETWORK_FEE: f64 = 0.05;
    
    /// Default minimum tip amount in NYM tokens
    pub const MIN_TIP_AMOUNT: u64 = 10;
    
    /// Default maximum subscription duration in days
    pub const MAX_SUBSCRIPTION_DURATION: u32 = 365;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that all main types are accessible
        println!("Creator economy modules loaded successfully");
    }
}