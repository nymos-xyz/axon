//! # Axon Personalization Engine
//! 
//! Privacy-preserving personalization system for the Axon social network.
//! Provides personalized content recommendations while maintaining user anonymity
//! through federated learning, differential privacy, and secure computation.

pub mod error;
pub mod privacy_personalization;
pub mod federated_learning;
pub mod preference_learning;
pub mod recommendation_engine;
pub mod content_filtering;
pub mod personalization_analytics;

pub use error::{PersonalizationError, PersonalizationResult};
pub use privacy_personalization::{
    PrivacyPersonalizationEngine, PersonalizationConfig, UserProfile,
    PrivatePreferences, PersonalizationRequest, PersonalizationResponse
};
pub use federated_learning::{
    FederatedLearningClient, FederatedModel, ModelUpdate,
    AggregationStrategy, FederatedTrainingConfig
};
pub use preference_learning::{
    PreferenceLearner, PreferenceVector, InteractionHistory,
    PreferencePrediction, LearningStrategy
};
pub use recommendation_engine::{
    RecommendationEngine, RecommendationAlgorithm, ContentRecommendation,
    RecommendationScore, RecommendationMetrics
};
pub use content_filtering::{
    ContentFilter, FilteringCriteria, FilteredContent,
    ContentCategory, FilteringStrategy
};
pub use personalization_analytics::{
    PersonalizationAnalytics, PersonalizationMetrics, EngagementMetrics,
    PrivacyMetrics as PersonalizationPrivacyMetrics, RecommendationEffectiveness
};

/// Personalization protocol version
pub const PERSONALIZATION_PROTOCOL_VERSION: u32 = 1;

/// Maximum user profile dimensions for privacy
pub const MAX_PROFILE_DIMENSIONS: usize = 1000;

/// Default recommendation batch size
pub const DEFAULT_RECOMMENDATION_BATCH: usize = 50;

/// Privacy budget for differential privacy
pub const DEFAULT_PRIVACY_BUDGET: f64 = 10.0;