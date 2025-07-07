//! # Axon AI-Powered Privacy Features
//! 
//! Advanced AI-driven privacy protection and enhancement features for the Axon social network.
//! Provides intelligent content moderation, privacy recommendations, anonymous matching,
//! and predictive privacy controls while maintaining zero-knowledge about users.

pub mod error;
pub mod ai_moderation;
pub mod privacy_recommendations;
pub mod anonymous_matching;
pub mod predictive_privacy;
pub mod privacy_ml;
pub mod federated_ai;
pub mod ai_analytics;
pub mod social_computing;

pub use error::{AIError, AIResult};
pub use ai_moderation::{
    AIContentModerator, ModerationResult, ContentAnalysis,
    ModerationAction, ModerationConfig, ThreatDetection
};
pub use privacy_recommendations::{
    PrivacyRecommendationEngine, PrivacyRecommendation, RecommendationType,
    PrivacyInsight, RecommendationConfig, PrivacyScore
};
pub use anonymous_matching::{
    AnonymousMatchingEngine, MatchingAlgorithm, MatchingResult,
    CompatibilityScore, MatchingConfig, AnonymousProfile
};
pub use predictive_privacy::{
    PredictivePrivacyEngine, PrivacyPrediction, PrivacyRiskAssessment,
    PredictiveConfig, PrivacyTrend, RiskLevel
};
pub use privacy_ml::{
    PrivacyMLEngine, FederatedModel, ModelUpdate,
    PrivacyPreservingAlgorithm, MLConfig, TrainingJob
};
pub use federated_ai::{
    FederatedAICoordinator, FederatedLearningSession, ModelAggregation,
    ParticipantMetrics, FederatedConfig, AggregationStrategy
};
pub use ai_analytics::{
    AIAnalytics, IntelligentInsights, ContentTrends,
    UserBehaviorPatterns, PredictiveAnalytics, AnalyticsConfig
};
pub use social_computing::{
    SocialComputingEngine, SocialComputingConfig, ModerationRequest, AIModerationResult,
    SocialAnalysisRequest, SocialAnalysisResult, ContentProcessingRequest, ContentProcessingResult,
    RealtimeRecommendationRequest, RealtimeRecommendationResult, ModerationLevel, ViolationType,
    ModerationAction, SocialAnalysisType, ContentProcessingType, RecommendationType
};

/// AI privacy protocol version
pub const AI_PRIVACY_PROTOCOL_VERSION: u32 = 1;

/// Maximum model size for privacy preservation
pub const MAX_MODEL_SIZE_MB: usize = 100;

/// Default privacy budget for AI operations
pub const AI_PRIVACY_BUDGET: f64 = 2.0;

/// Minimum anonymity set for AI processing
pub const MIN_AI_ANONYMITY_SET: usize = 50;