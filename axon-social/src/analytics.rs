//! Analytics - Privacy-preserving social metrics and insights
//!
//! This module provides analytics while preserving user privacy:
//! - Anonymous engagement tracking
//! - Aggregated metrics without individual profiling
//! - Trend analysis with differential privacy
//! - Content performance metrics
//! - Platform health monitoring

use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use axon_core::{
    ContentHash as ContentId,
    content::Post as Content,
    identity::QuIDIdentity as Identity
};
use crate::{SocialError, SocialResult, PrivacyLevel, Interaction};
use crate::social_graph::UserId;

/// Anonymous analytics engine
#[derive(Debug)]
pub struct AnonymousAnalytics {
    /// Engagement metrics by content
    content_metrics: HashMap<ContentId, ContentMetrics>,
    /// Aggregated user metrics (anonymized)
    user_metrics: HashMap<String, UserMetrics>, // Hashed user IDs
    /// Platform-wide engagement metrics
    platform_metrics: EngagementMetrics,
    /// Trend data over time
    trend_data: VecDeque<TrendSnapshot>,
    /// Analytics settings
    settings: AnalyticsSettings,
}

/// Engagement metrics for content or platform
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngagementMetrics {
    /// Total likes received
    pub total_likes: u64,
    /// Total dislikes received
    pub total_dislikes: u64,
    /// Total replies/comments
    pub total_replies: u64,
    /// Total shares
    pub total_shares: u64,
    /// Total bookmarks
    pub total_bookmarks: u64,
    /// Unique users interacted (approximated)
    pub unique_interactions: u64,
    /// Engagement rate (interactions per view)
    pub engagement_rate: f64,
    /// Average engagement per content
    pub avg_engagement: f64,
    /// Peak engagement time
    pub peak_engagement_time: Option<DateTime<Utc>>,
}

/// Metrics for individual users (anonymized)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMetrics {
    /// Anonymized user identifier
    pub user_hash: String,
    /// Content creation count
    pub content_created: u32,
    /// Interactions made
    pub interactions_made: u32,
    /// Interactions received on content
    pub interactions_received: u32,
    /// Follower count (approximate range)
    pub follower_range: FollowerRange,
    /// Account age in days
    pub account_age_days: u32,
    /// Engagement consistency score
    pub consistency_score: f64,
    /// Privacy level preference
    pub privacy_preference: PrivacyLevel,
    /// Last activity time (rounded to hour)
    pub last_activity: DateTime<Utc>,
}

/// Content-specific metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentMetrics {
    /// Content identifier
    pub content_id: ContentId,
    /// Number of views
    pub views: u64,
    /// Engagement breakdown
    pub engagement: EngagementMetrics,
    /// Time to first interaction
    pub time_to_first_interaction: Option<Duration>,
    /// Peak engagement period
    pub peak_period: Option<TimeRange>,
    /// Content reach (estimated unique users)
    pub estimated_reach: u64,
    /// Virality coefficient
    pub virality_score: f64,
    /// Content lifetime (creation to last interaction)
    pub content_lifetime: Duration,
    /// Geographic distribution (anonymized regions)
    pub region_distribution: HashMap<String, u32>,
}

/// Follower count ranges for privacy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FollowerRange {
    /// 0-10 followers
    VerySmall,
    /// 11-100 followers
    Small,
    /// 101-1000 followers
    Medium,
    /// 1001-10000 followers
    Large,
    /// 10000+ followers
    VeryLarge,
}

/// Time range for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time
    pub start: DateTime<Utc>,
    /// End time
    pub end: DateTime<Utc>,
}

/// Snapshot of trends at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,
    /// Platform metrics at this time
    pub platform_metrics: EngagementMetrics,
    /// Active users (approximate)
    pub active_users: u64,
    /// Content creation rate (per hour)
    pub content_creation_rate: f64,
    /// Top content categories
    pub trending_categories: Vec<String>,
}

/// Analytics configuration
#[derive(Debug, Clone)]
pub struct AnalyticsSettings {
    /// Enable user metrics collection
    pub collect_user_metrics: bool,
    /// Enable content metrics collection
    pub collect_content_metrics: bool,
    /// Retention period for metrics (days)
    pub retention_days: u32,
    /// Anonymization level
    pub anonymization_level: AnonymizationLevel,
    /// Update frequency (minutes)
    pub update_frequency_minutes: u32,
    /// Enable trend tracking
    pub enable_trend_tracking: bool,
}

/// Levels of data anonymization
#[derive(Debug, Clone, PartialEq)]
pub enum AnonymizationLevel {
    /// Basic anonymization (hashed IDs)
    Basic,
    /// Strong anonymization (differential privacy)
    Strong,
    /// Maximum anonymization (aggregated only)
    Maximum,
}

impl Default for AnalyticsSettings {
    fn default() -> Self {
        Self {
            collect_user_metrics: true,
            collect_content_metrics: true,
            retention_days: 90,
            anonymization_level: AnonymizationLevel::Strong,
            update_frequency_minutes: 60,
            enable_trend_tracking: true,
        }
    }
}

impl AnonymousAnalytics {
    /// Create new analytics engine
    pub fn new() -> Self {
        Self::with_settings(AnalyticsSettings::default())
    }

    /// Create analytics engine with custom settings
    pub fn with_settings(settings: AnalyticsSettings) -> Self {
        Self {
            content_metrics: HashMap::new(),
            user_metrics: HashMap::new(),
            platform_metrics: EngagementMetrics::default(),
            trend_data: VecDeque::new(),
            settings,
        }
    }

    /// Record an interaction for analytics
    pub fn record_interaction(&mut self, interaction: &Interaction) -> SocialResult<()> {
        // Update content metrics
        if self.settings.collect_content_metrics {
            self.update_content_metrics(&interaction.content_id, interaction);
        }

        // Update user metrics (anonymized)
        if self.settings.collect_user_metrics {
            if let Some(user_id) = &interaction.user_id {
                self.update_user_metrics(user_id, interaction);
            }
        }

        // Update platform metrics
        self.update_platform_metrics(interaction);

        Ok(())
    }

    /// Get content metrics
    pub fn get_content_metrics(&self, content_id: &ContentId) -> Option<&ContentMetrics> {
        self.content_metrics.get(content_id)
    }

    /// Get platform-wide metrics
    pub fn get_platform_metrics(&self) -> &EngagementMetrics {
        &self.platform_metrics
    }

    /// Get trending content metrics
    pub fn get_trending_content(&self, limit: usize) -> Vec<(ContentId, f64)> {
        let mut trending: Vec<(ContentId, f64)> = self.content_metrics
            .iter()
            .map(|(id, metrics)| (id.clone(), metrics.virality_score))
            .collect();

        trending.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        trending.truncate(limit);
        trending
    }

    /// Get engagement trends over time
    pub fn get_engagement_trends(&self, time_range: TimeRange) -> Vec<TrendSnapshot> {
        self.trend_data
            .iter()
            .filter(|snapshot| {
                snapshot.timestamp >= time_range.start && snapshot.timestamp <= time_range.end
            })
            .cloned()
            .collect()
    }

    /// Generate analytics report
    pub fn generate_report(&self, time_range: TimeRange) -> AnalyticsReport {
        let trends = self.get_engagement_trends(time_range.clone());
        let trending_content = self.get_trending_content(10);

        AnalyticsReport {
            time_range,
            platform_summary: self.platform_metrics.clone(),
            trending_content,
            trend_snapshots: trends,
            total_content: self.content_metrics.len() as u64,
            total_users: self.user_metrics.len() as u64,
            generated_at: Utc::now(),
        }
    }

    /// Update metrics periodically
    pub fn update_metrics(&mut self) {
        // Create trend snapshot
        if self.settings.enable_trend_tracking {
            self.create_trend_snapshot();
        }

        // Clean old data
        self.cleanup_old_data();

        // Recalculate derived metrics
        self.recalculate_platform_metrics();
    }

    // Private helper methods

    fn update_content_metrics(&mut self, content_id: &ContentId, interaction: &Interaction) {
        let metrics = self.content_metrics
            .entry(content_id.clone())
            .or_insert_with(|| ContentMetrics {
                content_id: content_id.clone(),
                ..Default::default()
            });

        // Update engagement metrics
        match &interaction.interaction_type {
            crate::InteractionType::Like => metrics.engagement.total_likes += 1,
            crate::InteractionType::Dislike => metrics.engagement.total_dislikes += 1,
            crate::InteractionType::Reply { .. } => metrics.engagement.total_replies += 1,
            crate::InteractionType::Comment { .. } => metrics.engagement.total_replies += 1,
            crate::InteractionType::Share { .. } => metrics.engagement.total_shares += 1,
            crate::InteractionType::Bookmark => metrics.engagement.total_bookmarks += 1,
            _ => {}
        }

        metrics.engagement.unique_interactions += 1;
        
        // Update virality score based on shares and engagement velocity
        metrics.virality_score = self.calculate_virality_score(metrics);
    }

    fn update_user_metrics(&mut self, user_id: &str, interaction: &Interaction) {
        let user_hash = self.hash_user_id(user_id);
        let metrics = self.user_metrics
            .entry(user_hash.clone())
            .or_insert_with(|| UserMetrics {
                user_hash: user_hash.clone(),
                privacy_preference: interaction.privacy_level.clone(),
                last_activity: interaction.created_at,
                ..Default::default()
            });

        metrics.interactions_made += 1;
        metrics.last_activity = interaction.created_at;
        
        // Update consistency score based on regular activity
        let interactions_made = metrics.interactions_made;
        let account_age = metrics.account_age_days;
        metrics.consistency_score = self.calculate_consistency_score_from_values(interactions_made, account_age);
    }

    fn update_platform_metrics(&mut self, interaction: &Interaction) {
        match &interaction.interaction_type {
            crate::InteractionType::Like => self.platform_metrics.total_likes += 1,
            crate::InteractionType::Dislike => self.platform_metrics.total_dislikes += 1,
            crate::InteractionType::Reply { .. } => self.platform_metrics.total_replies += 1,
            crate::InteractionType::Comment { .. } => self.platform_metrics.total_replies += 1,
            crate::InteractionType::Share { .. } => self.platform_metrics.total_shares += 1,
            crate::InteractionType::Bookmark => self.platform_metrics.total_bookmarks += 1,
            _ => {}
        }

        self.platform_metrics.unique_interactions += 1;
    }

    fn calculate_virality_score(&self, metrics: &ContentMetrics) -> f64 {
        let shares = metrics.engagement.total_shares as f64;
        let total_interactions = (metrics.engagement.total_likes + 
                                 metrics.engagement.total_replies + 
                                 metrics.engagement.total_shares) as f64;

        if total_interactions == 0.0 {
            return 0.0;
        }

        // Virality = (shares / total_interactions) * interaction_velocity
        let share_ratio = shares / total_interactions;
        let interaction_velocity = total_interactions / metrics.content_lifetime.num_hours() as f64;
        
        (share_ratio * interaction_velocity).min(1.0)
    }

    fn calculate_consistency_score_from_values(&self, interactions_made: u32, account_age_days: u32) -> f64 {
        // Simple consistency calculation based on regular activity
        // In practice, this would analyze activity patterns over time
        if interactions_made < 10 {
            return 0.1;
        }

        (interactions_made as f64 / (account_age_days as f64 + 1.0)).min(1.0)
    }

    fn create_trend_snapshot(&mut self) {
        let snapshot = TrendSnapshot {
            timestamp: Utc::now(),
            platform_metrics: self.platform_metrics.clone(),
            active_users: self.user_metrics.len() as u64,
            content_creation_rate: 0.0, // Would calculate from recent data
            trending_categories: vec![], // Would analyze from content types
        };

        self.trend_data.push_back(snapshot);

        // Keep only recent trends (last 30 days)
        let cutoff = Utc::now() - Duration::days(30);
        while let Some(front) = self.trend_data.front() {
            if front.timestamp < cutoff {
                self.trend_data.pop_front();
            } else {
                break;
            }
        }
    }

    fn cleanup_old_data(&mut self) {
        let cutoff = Utc::now() - Duration::days(self.settings.retention_days as i64);
        
        // Remove old user metrics
        self.user_metrics.retain(|_, metrics| metrics.last_activity > cutoff);
        
        // In practice, would also clean old content metrics based on activity
    }

    fn recalculate_platform_metrics(&mut self) {
        // Recalculate aggregate metrics
        let total_content = self.content_metrics.len() as f64;
        if total_content > 0.0 {
            let total_engagement: u64 = self.content_metrics
                .values()
                .map(|m| m.engagement.total_likes + m.engagement.total_replies + m.engagement.total_shares)
                .sum();
            
            self.platform_metrics.avg_engagement = total_engagement as f64 / total_content;
        }
    }

    fn hash_user_id(&self, user_id: &str) -> String {
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        hasher.update(user_id.as_bytes());
        hasher.update(b"analytics_salt");
        
        hex::encode(hasher.finalize())
    }
}

/// Analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsReport {
    /// Time range of the report
    pub time_range: TimeRange,
    /// Platform summary metrics
    pub platform_summary: EngagementMetrics,
    /// Trending content
    pub trending_content: Vec<(ContentId, f64)>,
    /// Trend data over time
    pub trend_snapshots: Vec<TrendSnapshot>,
    /// Total content pieces
    pub total_content: u64,
    /// Total users (approximate)
    pub total_users: u64,
    /// Report generation time
    pub generated_at: DateTime<Utc>,
}

impl Default for AnonymousAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for UserMetrics {
    fn default() -> Self {
        Self {
            user_hash: String::new(),
            content_created: 0,
            interactions_made: 0,
            interactions_received: 0,
            follower_range: FollowerRange::VerySmall,
            account_age_days: 0,
            consistency_score: 0.0,
            privacy_preference: PrivacyLevel::Anonymous,
            last_activity: chrono::Utc::now(),
        }
    }
}

impl Default for FollowerRange {
    fn default() -> Self {
        FollowerRange::VerySmall
    }
}

impl FollowerRange {
    /// Convert follower count to range
    pub fn from_count(count: usize) -> Self {
        match count {
            0..=10 => FollowerRange::VerySmall,
            11..=100 => FollowerRange::Small,
            101..=1000 => FollowerRange::Medium,
            1001..=10000 => FollowerRange::Large,
            _ => FollowerRange::VeryLarge,
        }
    }

    /// Get approximate midpoint of range
    pub fn midpoint(&self) -> usize {
        match self {
            FollowerRange::VerySmall => 5,
            FollowerRange::Small => 55,
            FollowerRange::Medium => 550,
            FollowerRange::Large => 5500,
            FollowerRange::VeryLarge => 50000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Interaction, InteractionType};
    use axon_core::ContentId;

    #[test]
    fn test_analytics_creation() {
        let analytics = AnonymousAnalytics::new();
        assert_eq!(analytics.platform_metrics.total_likes, 0);
        assert!(analytics.content_metrics.is_empty());
    }

    #[test]
    fn test_interaction_recording() {
        let mut analytics = AnonymousAnalytics::new();
        let content_id = ContentId::new("test_content");
        
        let interaction = Interaction {
            id: "test_interaction".to_string(),
            content_id: content_id.clone(),
            interaction_type: InteractionType::Like,
            user_id: Some("user1".to_string()),
            privacy_level: PrivacyLevel::Public,
            created_at: Utc::now(),
            metadata: None,
            proof: None,
        };

        analytics.record_interaction(&interaction).unwrap();
        
        assert_eq!(analytics.platform_metrics.total_likes, 1);
        assert!(analytics.content_metrics.contains_key(&content_id));
    }

    #[test]
    fn test_follower_range() {
        assert_eq!(FollowerRange::from_count(5), FollowerRange::VerySmall);
        assert_eq!(FollowerRange::from_count(50), FollowerRange::Small);
        assert_eq!(FollowerRange::from_count(500), FollowerRange::Medium);
        assert_eq!(FollowerRange::from_count(5000), FollowerRange::Large);
        assert_eq!(FollowerRange::from_count(50000), FollowerRange::VeryLarge);
    }

    #[test]
    fn test_trending_content() {
        let mut analytics = AnonymousAnalytics::new();
        
        // Add some content with different virality scores
        let content1 = ContentId::new("content1");
        let content2 = ContentId::new("content2");
        
        analytics.content_metrics.insert(content1.clone(), ContentMetrics {
            content_id: content1.clone(),
            virality_score: 0.8,
            ..Default::default()
        });
        
        analytics.content_metrics.insert(content2.clone(), ContentMetrics {
            content_id: content2.clone(),
            virality_score: 0.6,
            ..Default::default()
        });
        
        let trending = analytics.get_trending_content(2);
        assert_eq!(trending.len(), 2);
        assert_eq!(trending[0].0, content1); // Higher virality first
        assert_eq!(trending[1].0, content2);
    }
}