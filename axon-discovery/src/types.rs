use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryRequest {
    pub request_id: Uuid,
    pub anonymous_id: String,
    pub interests: Vec<Interest>,
    pub content_types: Vec<ContentType>,
    pub privacy_level: PrivacyLevel,
    pub max_results: usize,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResponse {
    pub request_id: Uuid,
    pub results: Vec<DiscoveryResult>,
    pub privacy_proofs: Vec<PrivacyProof>,
    pub processing_metadata: ProcessingMetadata,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResult {
    pub content_id: String,
    pub content_type: ContentType,
    pub relevance_score: f64,
    pub privacy_preserving_metadata: HashMap<String, String>,
    pub anonymous_engagement_data: Option<AnonymousEngagementData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interest {
    pub category: String,
    pub subcategory: Option<String>,
    pub weight: f64,
    pub privacy_masked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Post,
    Comment,
    Media,
    Link,
    User,
    Domain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyLevel {
    Anonymous,
    Pseudonymous,
    Private,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyProof {
    pub proof_type: String,
    pub proof_data: Vec<u8>,
    pub verification_key: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub compute_job_id: Option<String>,
    pub processing_time_ms: u64,
    pub privacy_preserved: bool,
    pub distributed_processing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymousEngagementData {
    pub total_interactions: u64,
    pub engagement_score: f64,
    pub trending_factor: f64,
    pub quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterestVector {
    pub vector_id: String,
    pub dimensions: Vec<f64>,
    pub privacy_noise: Vec<f64>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationConfig {
    pub algorithm: RecommendationAlgorithm,
    pub privacy_budget: f64,
    pub max_compute_time_ms: u64,
    pub use_distributed_processing: bool,
    pub anonymity_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationAlgorithm {
    CollaborativeFiltering,
    ContentBased,
    Hybrid,
    PrivacyPreserving,
}