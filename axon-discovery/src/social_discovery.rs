use crate::{
    error::{DiscoveryError, Result},
    types::*,
    privacy_preserving::PrivacyPreservingDiscovery,
    nymcompute_integration::NymComputeDiscovery,
};
use axon_core::types::ContentHash;
use axon_social::{
    social_graph::SocialGraph, 
    privacy::PrivacyManager,
    analytics::AnalyticsEngine,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn};
use chrono::Utc;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct SocialDiscoveryConfig {
    pub max_recommendations: usize,
    pub min_mutual_connections: usize,
    pub interest_similarity_threshold: f64,
    pub privacy_level: PrivacyLevel,
    pub enable_community_discovery: bool,
    pub social_proof_threshold: usize,
    pub recommendation_refresh_interval: chrono::Duration,
}

impl Default for SocialDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_recommendations: 50,
            min_mutual_connections: 2,
            interest_similarity_threshold: 0.7,
            privacy_level: PrivacyLevel::Anonymous,
            enable_community_discovery: true,
            social_proof_threshold: 3,
            recommendation_refresh_interval: chrono::Duration::hours(6),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRecommendation {
    pub user_id: String,
    pub relevance_score: f64,
    pub mutual_connections: usize,
    pub shared_interests: Vec<String>,
    pub social_proof: Vec<SocialProof>,
    pub privacy_preserved: bool,
    pub recommendation_reason: RecommendationReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialProof {
    pub proof_type: SocialProofType,
    pub evidence: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SocialProofType {
    MutualFollower,
    SharedCommunity,
    SimilarInterests,
    EngagementPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationReason {
    MutualConnections,
    SharedInterests,
    CommunityBased,
    EngagementSimilarity,
    LocationProximity,
    ActivityPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityRecommendation {
    pub community_id: String,
    pub community_name: String,
    pub member_count: usize,
    pub activity_level: f64,
    pub interest_alignment: f64,
    pub privacy_level: PrivacyLevel,
    pub joining_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialDiscoveryRequest {
    pub request_id: Uuid,
    pub user_id: String,
    pub discovery_type: SocialDiscoveryType,
    pub user_interests: Vec<Interest>,
    pub current_connections: Vec<String>,
    pub privacy_preferences: SocialPrivacyPreferences,
    pub max_results: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SocialDiscoveryType {
    UserRecommendations,
    CommunityRecommendations,
    MutualConnectionSuggestions,
    InterestBasedMatching,
    ActivityBasedSuggestions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialPrivacyPreferences {
    pub allow_mutual_connection_discovery: bool,
    pub allow_interest_based_matching: bool,
    pub allow_activity_analysis: bool,
    pub minimum_anonymity_set_size: usize,
    pub enable_differential_privacy: bool,
}

pub struct SocialDiscoveryEngine {
    config: SocialDiscoveryConfig,
    privacy_discovery: Arc<PrivacyPreservingDiscovery>,
    nymcompute_discovery: Arc<NymComputeDiscovery>,
    social_graph: Arc<SocialGraph>,
    privacy_manager: Arc<PrivacyManager>,
    analytics: Arc<AnalyticsEngine>,
    user_profiles: Arc<RwLock<HashMap<String, AnonymousUserProfile>>>,
    community_registry: Arc<RwLock<HashMap<String, CommunityInfo>>>,
    recommendation_cache: Arc<RwLock<HashMap<String, CachedRecommendations>>>,
}

#[derive(Debug, Clone)]
struct AnonymousUserProfile {
    user_id: String,
    interest_vector: Vec<f64>,
    activity_pattern: ActivityPattern,
    connection_preferences: ConnectionPreferences,
    last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
struct ActivityPattern {
    posting_frequency: f64,
    engagement_ratio: f64,
    active_hours: Vec<u8>,
    content_types: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
struct ConnectionPreferences {
    preferred_connection_types: Vec<String>,
    interaction_style: InteractionStyle,
    privacy_level: PrivacyLevel,
}

#[derive(Debug, Clone)]
enum InteractionStyle {
    PublicEngagement,
    PrivateMessaging,
    CommunityFocused,
    ContentCreation,
}

#[derive(Debug, Clone)]
struct CommunityInfo {
    community_id: String,
    name: String,
    description: String,
    member_count: usize,
    activity_metrics: CommunityActivityMetrics,
    interest_tags: Vec<String>,
    privacy_settings: CommunityPrivacySettings,
}

#[derive(Debug, Clone)]
struct CommunityActivityMetrics {
    posts_per_day: f64,
    active_members_ratio: f64,
    engagement_score: f64,
    growth_rate: f64,
}

#[derive(Debug, Clone)]
struct CommunityPrivacySettings {
    visibility: CommunityVisibility,
    joining_requirements: Vec<JoiningRequirement>,
    member_privacy_level: PrivacyLevel,
}

#[derive(Debug, Clone)]
enum CommunityVisibility {
    Public,
    PrivateInviteOnly,
    AnonymousOpen,
}

#[derive(Debug, Clone)]
enum JoiningRequirement {
    MinimumReputation(f64),
    InvitationRequired,
    AnswerQuestions(Vec<String>),
    StakeTokens(u64),
}

#[derive(Debug, Clone)]
struct CachedRecommendations {
    user_id: String,
    recommendations: Vec<UserRecommendation>,
    generated_at: chrono::DateTime<chrono::Utc>,
    expires_at: chrono::DateTime<chrono::Utc>,
}

impl SocialDiscoveryEngine {
    pub async fn new(
        config: SocialDiscoveryConfig,
        privacy_discovery: Arc<PrivacyPreservingDiscovery>,
        nymcompute_discovery: Arc<NymComputeDiscovery>,
        social_graph: Arc<SocialGraph>,
        privacy_manager: Arc<PrivacyManager>,
        analytics: Arc<AnalyticsEngine>,
    ) -> Result<Self> {
        info!("Initializing Social Discovery Engine with privacy-first user and community recommendations");

        Ok(Self {
            config,
            privacy_discovery,
            nymcompute_discovery,
            social_graph,
            privacy_manager,
            analytics,
            user_profiles: Arc::new(RwLock::new(HashMap::new())),
            community_registry: Arc::new(RwLock::new(HashMap::new())),
            recommendation_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn discover_users(&self, request: SocialDiscoveryRequest) -> Result<Vec<UserRecommendation>> {
        info!("Processing social user discovery request for user: {}", request.user_id);

        // Check cache first
        if let Some(cached) = self.get_cached_recommendations(&request.user_id).await? {
            if cached.expires_at > Utc::now() {
                debug!("Returning cached recommendations for user: {}", request.user_id);
                return Ok(cached.recommendations);
            }
        }

        let recommendations = match request.discovery_type {
            SocialDiscoveryType::UserRecommendations => {
                self.generate_user_recommendations(&request).await?
            },
            SocialDiscoveryType::MutualConnectionSuggestions => {
                self.find_mutual_connection_suggestions(&request).await?
            },
            SocialDiscoveryType::InterestBasedMatching => {
                self.match_by_interests(&request).await?
            },
            SocialDiscoveryType::ActivityBasedSuggestions => {
                self.suggest_by_activity_patterns(&request).await?
            },
            _ => {
                return Err(DiscoveryError::Internal(
                    "Unsupported discovery type for user recommendations".to_string()
                ));
            }
        };

        // Cache the results
        self.cache_recommendations(&request.user_id, &recommendations).await?;

        Ok(recommendations)
    }

    pub async fn discover_communities(&self, request: SocialDiscoveryRequest) -> Result<Vec<CommunityRecommendation>> {
        info!("Processing community discovery request for user: {}", request.user_id);

        match request.discovery_type {
            SocialDiscoveryType::CommunityRecommendations => {
                self.recommend_communities(&request).await
            },
            _ => {
                Err(DiscoveryError::Internal(
                    "Invalid discovery type for community recommendations".to_string()
                ))
            }
        }
    }

    async fn generate_user_recommendations(&self, request: &SocialDiscoveryRequest) -> Result<Vec<UserRecommendation>> {
        debug!("Generating comprehensive user recommendations");

        let mut all_recommendations = Vec::new();

        // Get mutual connection suggestions
        let mutual_suggestions = self.find_mutual_connection_suggestions(request).await?;
        all_recommendations.extend(mutual_suggestions);

        // Get interest-based matches
        let interest_matches = self.match_by_interests(request).await?;
        all_recommendations.extend(interest_matches);

        // Get activity-based suggestions
        let activity_suggestions = self.suggest_by_activity_patterns(request).await?;
        all_recommendations.extend(activity_suggestions);

        // Deduplicate and rank
        let deduplicated = self.deduplicate_and_rank_recommendations(all_recommendations).await?;
        
        // Apply privacy filters
        let privacy_filtered = self.apply_privacy_filters(&deduplicated, &request.privacy_preferences).await?;

        // Limit results
        Ok(privacy_filtered.into_iter().take(request.max_results).collect())
    }

    async fn find_mutual_connection_suggestions(&self, request: &SocialDiscoveryRequest) -> Result<Vec<UserRecommendation>> {
        if !request.privacy_preferences.allow_mutual_connection_discovery {
            return Ok(Vec::new());
        }

        debug!("Finding mutual connection suggestions");

        let mutual_connections = self.social_graph
            .find_mutual_connections(&request.user_id, self.config.min_mutual_connections).await
            .map_err(|e| DiscoveryError::Internal(format!("Social graph error: {}", e)))?;

        let mut recommendations = Vec::new();
        
        for (candidate_user, mutual_count) in mutual_connections {
            if request.current_connections.contains(&candidate_user) {
                continue;
            }

            let social_proof = vec![SocialProof {
                proof_type: SocialProofType::MutualFollower,
                evidence: format!("{} mutual connections", mutual_count),
                confidence: (mutual_count as f64 / 10.0).min(1.0),
            }];

            let recommendation = UserRecommendation {
                user_id: candidate_user,
                relevance_score: self.calculate_mutual_connection_score(mutual_count),
                mutual_connections: mutual_count,
                shared_interests: Vec::new(), // Would be populated from user profiles
                social_proof,
                privacy_preserved: true,
                recommendation_reason: RecommendationReason::MutualConnections,
            };

            recommendations.push(recommendation);
        }

        Ok(recommendations)
    }

    async fn match_by_interests(&self, request: &SocialDiscoveryRequest) -> Result<Vec<UserRecommendation>> {
        if !request.privacy_preferences.allow_interest_based_matching {
            return Ok(Vec::new());
        }

        debug!("Matching users by interests with privacy preservation");

        // Use NymCompute for privacy-preserving interest matching
        let interest_matching_job = self.create_interest_matching_job(request).await?;
        let matching_results = self.nymcompute_discovery
            .process_privacy_preserving_matching(interest_matching_job).await?;

        let mut recommendations = Vec::new();
        
        for match_result in matching_results {
            if request.current_connections.contains(&match_result.user_id) {
                continue;
            }

            let social_proof = vec![SocialProof {
                proof_type: SocialProofType::SimilarInterests,
                evidence: format!("Shared interest categories: {}", match_result.shared_interests.len()),
                confidence: match_result.similarity_score,
            }];

            let recommendation = UserRecommendation {
                user_id: match_result.user_id,
                relevance_score: match_result.similarity_score,
                mutual_connections: 0, // Would be calculated separately
                shared_interests: match_result.shared_interests,
                social_proof,
                privacy_preserved: true,
                recommendation_reason: RecommendationReason::SharedInterests,
            };

            recommendations.push(recommendation);
        }

        Ok(recommendations)
    }

    async fn suggest_by_activity_patterns(&self, request: &SocialDiscoveryRequest) -> Result<Vec<UserRecommendation>> {
        if !request.privacy_preferences.allow_activity_analysis {
            return Ok(Vec::new());
        }

        debug!("Suggesting users based on activity patterns");

        let user_activity_pattern = self.get_user_activity_pattern(&request.user_id).await?;
        
        let similar_users = self.find_users_with_similar_activity(&user_activity_pattern).await?;

        let mut recommendations = Vec::new();
        
        for (user_id, similarity_score) in similar_users {
            if request.current_connections.contains(&user_id) {
                continue;
            }

            let social_proof = vec![SocialProof {
                proof_type: SocialProofType::EngagementPattern,
                evidence: "Similar activity patterns".to_string(),
                confidence: similarity_score,
            }];

            let recommendation = UserRecommendation {
                user_id,
                relevance_score: similarity_score,
                mutual_connections: 0,
                shared_interests: Vec::new(),
                social_proof,
                privacy_preserved: true,
                recommendation_reason: RecommendationReason::EngagementSimilarity,
            };

            recommendations.push(recommendation);
        }

        Ok(recommendations)
    }

    async fn recommend_communities(&self, request: &SocialDiscoveryRequest) -> Result<Vec<CommunityRecommendation>> {
        debug!("Recommending communities based on user interests and activity");

        let communities = self.community_registry.read().await;
        let mut recommendations = Vec::new();

        for community in communities.values() {
            let interest_alignment = self.calculate_community_interest_alignment(
                &request.user_interests,
                &community.interest_tags,
            ).await?;

            if interest_alignment >= self.config.interest_similarity_threshold {
                let recommendation = CommunityRecommendation {
                    community_id: community.community_id.clone(),
                    community_name: community.name.clone(),
                    member_count: community.member_count,
                    activity_level: community.activity_metrics.engagement_score,
                    interest_alignment,
                    privacy_level: community.privacy_settings.member_privacy_level.clone(),
                    joining_requirements: community.privacy_settings.joining_requirements.iter()
                        .map(|req| format!("{:?}", req))
                        .collect(),
                };

                recommendations.push(recommendation);
            }
        }

        // Sort by relevance
        recommendations.sort_by(|a, b| b.interest_alignment.partial_cmp(&a.interest_alignment).unwrap());
        recommendations.truncate(request.max_results);

        Ok(recommendations)
    }

    async fn create_interest_matching_job(&self, request: &SocialDiscoveryRequest) -> Result<InterestMatchingJob> {
        Ok(InterestMatchingJob {
            job_id: Uuid::new_v4().to_string(),
            user_interests: request.user_interests.clone(),
            similarity_threshold: self.config.interest_similarity_threshold,
            max_results: request.max_results,
            privacy_level: request.privacy_preferences.clone(),
        })
    }

    async fn calculate_mutual_connection_score(&self, mutual_count: usize) -> f64 {
        // Logarithmic scoring to prevent over-weighting highly connected users
        (mutual_count as f64 + 1.0).ln() / 10.0_f64.ln()
    }

    async fn get_user_activity_pattern(&self, user_id: &str) -> Result<ActivityPattern> {
        let profiles = self.user_profiles.read().await;
        
        if let Some(profile) = profiles.get(user_id) {
            Ok(profile.activity_pattern.clone())
        } else {
            // Generate default activity pattern
            Ok(ActivityPattern {
                posting_frequency: 0.5,
                engagement_ratio: 0.3,
                active_hours: vec![9, 12, 15, 18, 21], // Default active hours
                content_types: HashMap::new(),
            })
        }
    }

    async fn find_users_with_similar_activity(&self, pattern: &ActivityPattern) -> Result<Vec<(String, f64)>> {
        let profiles = self.user_profiles.read().await;
        let mut similar_users = Vec::new();

        for (user_id, profile) in profiles.iter() {
            let similarity = self.calculate_activity_similarity(pattern, &profile.activity_pattern);
            
            if similarity > 0.6 { // Similarity threshold
                similar_users.push((user_id.clone(), similarity));
            }
        }

        similar_users.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similar_users.truncate(50); // Limit results
        
        Ok(similar_users)
    }

    fn calculate_activity_similarity(&self, pattern1: &ActivityPattern, pattern2: &ActivityPattern) -> f64 {
        let frequency_sim = 1.0 - (pattern1.posting_frequency - pattern2.posting_frequency).abs();
        let engagement_sim = 1.0 - (pattern1.engagement_ratio - pattern2.engagement_ratio).abs();
        
        // Calculate hour overlap
        let hours1: HashSet<_> = pattern1.active_hours.iter().collect();
        let hours2: HashSet<_> = pattern2.active_hours.iter().collect();
        let overlap = hours1.intersection(&hours2).count();
        let union = hours1.union(&hours2).count();
        let hour_sim = if union > 0 { overlap as f64 / union as f64 } else { 0.0 };

        (frequency_sim + engagement_sim + hour_sim) / 3.0
    }

    async fn calculate_community_interest_alignment(
        &self,
        user_interests: &[Interest],
        community_tags: &[String],
    ) -> Result<f64> {
        if user_interests.is_empty() || community_tags.is_empty() {
            return Ok(0.0);
        }

        let user_categories: HashSet<_> = user_interests.iter()
            .map(|i| &i.category)
            .collect();
        
        let community_categories: HashSet<_> = community_tags.iter().collect();
        
        let intersection = user_categories.intersection(&community_categories).count();
        let union = user_categories.union(&community_categories).count();

        if union > 0 {
            Ok(intersection as f64 / union as f64)
        } else {
            Ok(0.0)
        }
    }

    async fn deduplicate_and_rank_recommendations(&self, mut recommendations: Vec<UserRecommendation>) -> Result<Vec<UserRecommendation>> {
        let mut seen_users = HashSet::new();
        let mut deduplicated = Vec::new();

        // Sort by relevance score first
        recommendations.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        for rec in recommendations {
            if !seen_users.contains(&rec.user_id) {
                seen_users.insert(rec.user_id.clone());
                deduplicated.push(rec);
            }
        }

        Ok(deduplicated)
    }

    async fn apply_privacy_filters(
        &self,
        recommendations: &[UserRecommendation],
        privacy_prefs: &SocialPrivacyPreferences,
    ) -> Result<Vec<UserRecommendation>> {
        let mut filtered = Vec::new();

        for rec in recommendations {
            // Check anonymity set size
            if privacy_prefs.minimum_anonymity_set_size > 0 {
                // This would check if the recommendation maintains k-anonymity
                // For now, we'll assume all recommendations meet the requirement
            }

            // Apply differential privacy if enabled
            let mut filtered_rec = rec.clone();
            if privacy_prefs.enable_differential_privacy {
                filtered_rec.relevance_score = self.add_differential_privacy_noise(rec.relevance_score);
            }

            filtered.push(filtered_rec);
        }

        Ok(filtered)
    }

    fn add_differential_privacy_noise(&self, score: f64) -> f64 {
        use rand::Rng;
        let noise = rand::thread_rng().gen_range(-0.05..0.05);
        (score + noise).clamp(0.0, 1.0)
    }

    async fn get_cached_recommendations(&self, user_id: &str) -> Result<Option<CachedRecommendations>> {
        let cache = self.recommendation_cache.read().await;
        Ok(cache.get(user_id).cloned())
    }

    async fn cache_recommendations(&self, user_id: &str, recommendations: &[UserRecommendation]) -> Result<()> {
        let mut cache = self.recommendation_cache.write().await;
        
        let cached = CachedRecommendations {
            user_id: user_id.to_string(),
            recommendations: recommendations.to_vec(),
            generated_at: Utc::now(),
            expires_at: Utc::now() + self.config.recommendation_refresh_interval,
        };

        cache.insert(user_id.to_string(), cached);
        Ok(())
    }

    pub async fn register_community(&self, community_info: CommunityInfo) -> Result<()> {
        info!("Registering community: {}", community_info.name);
        
        self.community_registry.write().await.insert(
            community_info.community_id.clone(),
            community_info,
        );
        
        Ok(())
    }

    pub async fn update_user_profile(&self, user_id: String, interests: Vec<Interest>) -> Result<()> {
        debug!("Updating user profile for social discovery: {}", user_id);

        let interest_vector = self.create_interest_vector(&interests).await?;
        
        let profile = AnonymousUserProfile {
            user_id: user_id.clone(),
            interest_vector,
            activity_pattern: ActivityPattern {
                posting_frequency: 0.5,
                engagement_ratio: 0.3,
                active_hours: vec![9, 12, 15, 18, 21],
                content_types: HashMap::new(),
            },
            connection_preferences: ConnectionPreferences {
                preferred_connection_types: vec!["following".to_string()],
                interaction_style: InteractionStyle::PublicEngagement,
                privacy_level: PrivacyLevel::Anonymous,
            },
            last_updated: Utc::now(),
        };

        self.user_profiles.write().await.insert(user_id, profile);
        Ok(())
    }

    async fn create_interest_vector(&self, interests: &[Interest]) -> Result<Vec<f64>> {
        let mut vector = vec![0.0; 128]; // Fixed dimension vector
        
        for (i, interest) in interests.iter().enumerate().take(128) {
            vector[i] = interest.weight;
        }

        Ok(vector)
    }
}

#[derive(Debug, Clone)]
struct InterestMatchingJob {
    job_id: String,
    user_interests: Vec<Interest>,
    similarity_threshold: f64,
    max_results: usize,
    privacy_level: SocialPrivacyPreferences,
}

#[derive(Debug, Clone)]
struct InterestMatchingResult {
    user_id: String,
    similarity_score: f64,
    shared_interests: Vec<String>,
}