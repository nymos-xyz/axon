//! Feed Generation - Privacy-preserving content feed system
//!
//! This module implements algorithmic and chronological feed generation:
//! - Chronological timeline with privacy controls
//! - Algorithmic ranking based on engagement patterns
//! - Anonymous content recommendation
//! - Privacy-aware trend detection
//! - User-customizable feed algorithms

use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use axon_core::{
    ContentHash as ContentId,
    content::Post as Content,
    identity::QuIDIdentity as Identity
};
use crate::{SocialError, SocialResult, PrivacyLevel, Interaction};
use crate::social_graph::UserId;

/// Feed generator for creating personalized content feeds
#[derive(Debug)]
pub struct FeedGenerator {
    /// Feed algorithms available
    algorithms: HashMap<String, FeedAlgorithm>,
    /// User feed preferences
    user_preferences: HashMap<UserId, UserFeedPreferences>,
    /// Feed cache
    feed_cache: HashMap<String, CachedFeed>,
    /// Content rankings
    content_rankings: HashMap<ContentId, ContentRanking>,
}

/// A user's personalized feed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feed {
    /// Feed identifier
    pub id: String,
    /// User ID (may be anonymous)
    pub user_id: Option<String>,
    /// Feed items in order
    pub items: Vec<FeedItem>,
    /// Feed generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Algorithm used to generate feed
    pub algorithm: String,
    /// Privacy level of feed
    pub privacy_level: PrivacyLevel,
}

/// Individual item in a feed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedItem {
    /// Item identifier
    pub id: String,
    /// Content reference
    pub content_id: ContentId,
    /// Content summary (privacy-aware)
    pub content_summary: Option<String>,
    /// Item position in feed
    pub position: u32,
    /// Ranking score
    pub score: f64,
    /// Reason for inclusion (privacy-preserving)
    pub inclusion_reason: InclusionReason,
    /// Timestamp when added to feed
    pub added_at: DateTime<Utc>,
}

/// Feed generation algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedAlgorithm {
    /// Chronological timeline
    Chronological,
    /// Engagement-based ranking
    Engagement,
    /// Interest-based recommendations
    InterestBased,
    /// Trending content
    Trending,
    /// Custom algorithm
    Custom(String),
}

/// Content ranking strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RankingStrategy {
    /// Time-based ranking (newest first)
    TimeDescending,
    /// Engagement score ranking
    EngagementScore,
    /// Relevance to user interests
    RelevanceScore,
    /// Trending momentum
    TrendingScore,
    /// Combined ranking factors
    Combined(Vec<RankingFactor>),
}

/// Factors for ranking calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingFactor {
    /// Factor name
    pub name: String,
    /// Weight in final score (0.0-1.0)
    pub weight: f64,
}

/// Reason why content was included in feed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InclusionReason {
    /// Following the content creator
    Following,
    /// High engagement content
    PopularContent,
    /// Trending in user's interests
    Trending,
    /// Similar to previously liked content
    SimilarContent,
    /// Recent from followed users
    RecentFromFollowed,
    /// Suggested based on anonymous patterns
    Suggested,
}

/// User preferences for feed generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedPreferences {
    /// Preferred algorithm
    pub algorithm: FeedAlgorithm,
    /// Content types to include
    pub content_types: Vec<String>,
    /// Content types to exclude
    pub excluded_types: Vec<String>,
    /// Privacy level for feed
    pub privacy_level: PrivacyLevel,
    /// Show content from anonymous users
    pub include_anonymous: bool,
    /// Maximum age of content (hours)
    pub max_content_age_hours: u32,
    /// Feed refresh frequency (minutes)
    pub refresh_frequency_minutes: u32,
}

/// Cached feed data
#[derive(Debug, Clone)]
struct CachedFeed {
    /// The cached feed
    feed: Feed,
    /// Cache expiry time
    expires_at: DateTime<Utc>,
    /// Cache hit count
    hit_count: u32,
}

/// Content ranking information
#[derive(Debug, Clone)]
struct ContentRanking {
    /// Overall ranking score
    score: f64,
    /// Engagement metrics
    engagement_score: f64,
    /// Trending score
    trending_score: f64,
    /// Recency score
    recency_score: f64,
    /// Last updated
    updated_at: DateTime<Utc>,
}

impl Default for UserFeedPreferences {
    fn default() -> Self {
        Self {
            algorithm: FeedAlgorithm::Chronological,
            content_types: vec!["text".to_string(), "image".to_string()],
            excluded_types: vec![],
            privacy_level: PrivacyLevel::Anonymous,
            include_anonymous: true,
            max_content_age_hours: 24,
            refresh_frequency_minutes: 15,
        }
    }
}

impl FeedGenerator {
    /// Create a new feed generator
    pub fn new() -> Self {
        Self {
            algorithms: Self::default_algorithms(),
            user_preferences: HashMap::new(),
            feed_cache: HashMap::new(),
            content_rankings: HashMap::new(),
        }
    }

    /// Generate feed for user
    pub async fn generate_feed(
        &mut self,
        user_identity: &Identity,
        algorithm: Option<FeedAlgorithm>,
        limit: Option<usize>,
    ) -> SocialResult<Feed> {
        let user_id = user_identity.get_id();
        let preferences = self.get_user_preferences(&user_id);
        let selected_algorithm = algorithm.unwrap_or(preferences.algorithm.clone());

        // Check cache first
        let cache_key = format!("{}_{:?}", user_id, selected_algorithm);
        if let Some(cached) = self.feed_cache.get(&cache_key) {
            if cached.expires_at > Utc::now() {
                return Ok(cached.feed.clone());
            }
        }

        // Generate new feed
        let feed = match selected_algorithm {
            FeedAlgorithm::Chronological => self.generate_chronological_feed(&user_id, limit).await?,
            FeedAlgorithm::Engagement => self.generate_engagement_feed(&user_id, limit).await?,
            FeedAlgorithm::Trending => self.generate_trending_feed(&user_id, limit).await?,
            _ => self.generate_chronological_feed(&user_id, limit).await?, // Fallback
        };

        // Cache the feed
        self.cache_feed(cache_key, feed.clone());

        Ok(feed)
    }

    /// Set user feed preferences
    pub fn set_user_preferences(&mut self, user_id: &str, preferences: UserFeedPreferences) {
        self.user_preferences.insert(user_id.to_string(), preferences);
    }

    /// Update content ranking
    pub fn update_content_ranking(&mut self, content_id: &ContentId, interactions: &[Interaction]) {
        let ranking = self.calculate_content_ranking(content_id, interactions);
        self.content_rankings.insert(content_id.clone(), ranking);
    }

    // Private implementation methods

    fn default_algorithms() -> HashMap<String, FeedAlgorithm> {
        let mut algorithms = HashMap::new();
        algorithms.insert("chronological".to_string(), FeedAlgorithm::Chronological);
        algorithms.insert("engagement".to_string(), FeedAlgorithm::Engagement);
        algorithms.insert("trending".to_string(), FeedAlgorithm::Trending);
        algorithms
    }

    fn get_user_preferences(&self, user_id: &str) -> UserFeedPreferences {
        self.user_preferences
            .get(user_id)
            .cloned()
            .unwrap_or_default()
    }

    async fn generate_chronological_feed(&self, user_id: &str, limit: Option<usize>) -> SocialResult<Feed> {
        // Placeholder implementation for chronological feed
        let feed_id = format!("chrono_{}_{}", user_id, Utc::now().timestamp());
        
        Ok(Feed {
            id: feed_id,
            user_id: Some(user_id.to_string()),
            items: vec![], // Would be populated with actual content
            generated_at: Utc::now(),
            algorithm: "chronological".to_string(),
            privacy_level: PrivacyLevel::Anonymous,
        })
    }

    async fn generate_engagement_feed(&self, user_id: &str, limit: Option<usize>) -> SocialResult<Feed> {
        // Placeholder implementation for engagement-based feed
        let feed_id = format!("engagement_{}_{}", user_id, Utc::now().timestamp());
        
        Ok(Feed {
            id: feed_id,
            user_id: Some(user_id.to_string()),
            items: vec![], // Would be populated based on engagement scores
            generated_at: Utc::now(),
            algorithm: "engagement".to_string(),
            privacy_level: PrivacyLevel::Anonymous,
        })
    }

    async fn generate_trending_feed(&self, user_id: &str, limit: Option<usize>) -> SocialResult<Feed> {
        // Placeholder implementation for trending feed
        let feed_id = format!("trending_{}_{}", user_id, Utc::now().timestamp());
        
        Ok(Feed {
            id: feed_id,
            user_id: Some(user_id.to_string()),
            items: vec![], // Would be populated with trending content
            generated_at: Utc::now(),
            algorithm: "trending".to_string(),
            privacy_level: PrivacyLevel::Anonymous,
        })
    }

    fn calculate_content_ranking(&self, content_id: &ContentId, interactions: &[Interaction]) -> ContentRanking {
        // Calculate various ranking scores
        let engagement_score = self.calculate_engagement_score(interactions);
        let trending_score = self.calculate_trending_score(interactions);
        let recency_score = self.calculate_recency_score(interactions);

        // Combined score (weighted average)
        let score = (engagement_score * 0.4) + (trending_score * 0.3) + (recency_score * 0.3);

        ContentRanking {
            score,
            engagement_score,
            trending_score,
            recency_score,
            updated_at: Utc::now(),
        }
    }

    fn calculate_engagement_score(&self, interactions: &[Interaction]) -> f64 {
        // Simple engagement calculation based on interaction count and types
        let mut score = 0.0;
        
        for interaction in interactions {
            match &interaction.interaction_type {
                crate::InteractionType::Like => score += 1.0,
                crate::InteractionType::Reply { .. } => score += 3.0,
                crate::InteractionType::Share { .. } => score += 5.0,
                crate::InteractionType::Comment { .. } => score += 2.0,
                _ => {}
            }
        }

        score / (interactions.len() as f64 + 1.0) // Normalize by interaction count
    }

    fn calculate_trending_score(&self, interactions: &[Interaction]) -> f64 {
        // Calculate trending based on recent interaction velocity
        let now = Utc::now();
        let hour_ago = now - chrono::Duration::hours(1);
        
        let recent_interactions = interactions.iter()
            .filter(|i| i.created_at > hour_ago)
            .count();

        (recent_interactions as f64).min(10.0) / 10.0 // Normalize to 0-1
    }

    fn calculate_recency_score(&self, interactions: &[Interaction]) -> f64 {
        if interactions.is_empty() {
            return 0.0;
        }

        let now = Utc::now();
        let most_recent = interactions.iter()
            .map(|i| i.created_at)
            .max()
            .unwrap_or(now);

        let hours_since = now.signed_duration_since(most_recent).num_hours() as f64;
        
        // Decay score over time (1.0 for recent, 0.0 for very old)
        (1.0 / (1.0 + hours_since / 24.0)).max(0.0)
    }

    fn cache_feed(&mut self, cache_key: String, feed: Feed) {
        let cached = CachedFeed {
            feed,
            expires_at: Utc::now() + chrono::Duration::minutes(15), // 15 minute cache
            hit_count: 0,
        };
        
        self.feed_cache.insert(cache_key, cached);
        
        // Clean old cache entries (simple cleanup)
        let now = Utc::now();
        self.feed_cache.retain(|_, cached| cached.expires_at > now);
    }
}

impl Default for FeedGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::Identity;

    #[tokio::test]
    async fn test_feed_generation() {
        let mut generator = FeedGenerator::new();
        let identity = Identity::new_for_test("user1");

        let feed = generator.generate_feed(&identity, None, Some(10)).await.unwrap();
        
        assert_eq!(feed.algorithm, "chronological");
        assert!(feed.items.is_empty()); // Placeholder implementation
    }

    #[test]
    fn test_user_preferences() {
        let mut generator = FeedGenerator::new();
        let user_id = "test_user";
        
        let preferences = UserFeedPreferences {
            algorithm: FeedAlgorithm::Engagement,
            ..Default::default()
        };
        
        generator.set_user_preferences(user_id, preferences.clone());
        
        let retrieved = generator.get_user_preferences(user_id);
        assert!(matches!(retrieved.algorithm, FeedAlgorithm::Engagement));
    }

    #[test]
    fn test_ranking_calculation() {
        let generator = FeedGenerator::new();
        let content_id = ContentId::new("test_content");
        let interactions = vec![];
        
        let ranking = generator.calculate_content_ranking(&content_id, &interactions);
        assert!(ranking.score >= 0.0);
        assert!(ranking.score <= 1.0);
    }
}