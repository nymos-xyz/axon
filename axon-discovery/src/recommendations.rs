use crate::{
    error::{DiscoveryError, Result},
    types::*,
};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{info, debug};
use chrono::Utc;
use uuid::Uuid;

pub struct RecommendationSystem {
    algorithms: HashMap<RecommendationAlgorithm, Box<dyn RecommendationAlgorithmImpl + Send + Sync>>,
    user_embeddings: RwLock<HashMap<String, UserEmbedding>>,
    content_embeddings: RwLock<HashMap<String, ContentEmbedding>>,
    config: DiscoveryConfig,
}

#[derive(Debug, Clone)]
struct UserEmbedding {
    user_id: String,
    embedding: Vec<f64>,
    privacy_noise: Vec<f64>,
    last_updated: chrono::DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct ContentEmbedding {
    content_id: String,
    content_type: ContentType,
    embedding: Vec<f64>,
    quality_score: f64,
    engagement_data: AnonymousEngagementData,
}

trait RecommendationAlgorithmImpl {
    fn recommend(
        &self,
        user_embedding: &UserEmbedding,
        content_embeddings: &[ContentEmbedding],
        config: &RecommendationConfig,
    ) -> Result<Vec<RecommendationScore>>;
    
    fn supports_privacy_preservation(&self) -> bool;
    fn algorithm_type(&self) -> RecommendationAlgorithm;
}

#[derive(Debug, Clone)]
struct RecommendationScore {
    content_id: String,
    score: f64,
    explanation: Option<String>,
}

impl RecommendationSystem {
    pub async fn new(config: DiscoveryConfig) -> Result<Self> {
        info!("Initializing privacy-preserving recommendation system");

        let mut algorithms: HashMap<RecommendationAlgorithm, Box<dyn RecommendationAlgorithmImpl + Send + Sync>> = HashMap::new();
        
        algorithms.insert(
            RecommendationAlgorithm::CollaborativeFiltering,
            Box::new(PrivacyPreservingCollaborativeFiltering::new()),
        );
        
        algorithms.insert(
            RecommendationAlgorithm::ContentBased,
            Box::new(ContentBasedRecommendation::new()),
        );
        
        algorithms.insert(
            RecommendationAlgorithm::PrivacyPreserving,
            Box::new(DifferentialPrivacyRecommendation::new()),
        );
        
        algorithms.insert(
            RecommendationAlgorithm::Hybrid,
            Box::new(HybridRecommendation::new()),
        );

        Ok(Self {
            algorithms,
            user_embeddings: RwLock::new(HashMap::new()),
            content_embeddings: RwLock::new(HashMap::new()),
            config,
        })
    }

    pub async fn generate_private_recommendations(&self, request: &DiscoveryRequest) -> Result<Vec<DiscoveryResult>> {
        info!("Generating private recommendations for anonymous user: {}", request.anonymous_id);

        let user_embedding = self.get_or_create_user_embedding(&request.anonymous_id, &request.interests).await?;
        
        let content_embeddings = self.get_relevant_content_embeddings(&request.content_types).await?;
        
        let recommendation_config = RecommendationConfig {
            algorithm: RecommendationAlgorithm::PrivacyPreserving,
            privacy_budget: self.config.privacy_budget_per_hour / 10.0,
            max_compute_time_ms: 30000,
            use_distributed_processing: self.config.enable_distributed_processing,
            anonymity_threshold: self.config.anonymity_threshold,
        };

        let algorithm = self.algorithms.get(&recommendation_config.algorithm)
            .ok_or_else(|| DiscoveryError::RecommendationError(
                "Recommendation algorithm not available".to_string()
            ))?;

        let scores = algorithm.recommend(&user_embedding, &content_embeddings, &recommendation_config)?;
        
        let mut results = Vec::new();
        for score in scores.into_iter().take(request.max_results) {
            if let Some(content_embedding) = content_embeddings.iter()
                .find(|c| c.content_id == score.content_id) {
                
                results.push(DiscoveryResult {
                    content_id: score.content_id,
                    content_type: content_embedding.content_type.clone(),
                    relevance_score: score.score,
                    privacy_preserving_metadata: self.create_recommendation_metadata(&score).await?,
                    anonymous_engagement_data: Some(content_embedding.engagement_data.clone()),
                });
            }
        }

        Ok(results)
    }

    async fn get_or_create_user_embedding(
        &self,
        anonymous_id: &str,
        interests: &[Interest],
    ) -> Result<UserEmbedding> {
        let embeddings = self.user_embeddings.read().await;
        
        if let Some(embedding) = embeddings.get(anonymous_id) {
            Ok(embedding.clone())
        } else {
            drop(embeddings);
            
            let new_embedding = self.create_user_embedding_from_interests(anonymous_id, interests).await?;
            
            let mut embeddings = self.user_embeddings.write().await;
            embeddings.insert(anonymous_id.to_string(), new_embedding.clone());
            
            Ok(new_embedding)
        }
    }

    async fn create_user_embedding_from_interests(
        &self,
        anonymous_id: &str,
        interests: &[Interest],
    ) -> Result<UserEmbedding> {
        let embedding_dim = 128;
        let mut embedding = vec![0.0; embedding_dim];
        let mut privacy_noise = vec![0.0; embedding_dim];

        for (i, interest) in interests.iter().enumerate().take(embedding_dim) {
            embedding[i] = interest.weight;
            
            if interest.privacy_masked {
                let noise = rand::random::<f64>() * 0.1 - 0.05;
                privacy_noise[i] = noise;
                embedding[i] += noise;
            }
        }

        Ok(UserEmbedding {
            user_id: anonymous_id.to_string(),
            embedding,
            privacy_noise,
            last_updated: Utc::now(),
        })
    }

    async fn get_relevant_content_embeddings(&self, content_types: &[ContentType]) -> Result<Vec<ContentEmbedding>> {
        let embeddings = self.content_embeddings.read().await;
        
        let relevant_embeddings: Vec<ContentEmbedding> = embeddings.values()
            .filter(|embedding| content_types.contains(&embedding.content_type))
            .cloned()
            .collect();
        
        Ok(relevant_embeddings)
    }

    async fn create_recommendation_metadata(&self, score: &RecommendationScore) -> Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        metadata.insert("recommendation_score".to_string(), score.score.to_string());
        metadata.insert("privacy_preserved".to_string(), "true".to_string());
        metadata.insert("algorithm".to_string(), "privacy_preserving".to_string());
        
        if let Some(explanation) = &score.explanation {
            metadata.insert("explanation".to_string(), explanation.clone());
        }
        
        Ok(metadata)
    }
}

struct PrivacyPreservingCollaborativeFiltering;

impl PrivacyPreservingCollaborativeFiltering {
    fn new() -> Self {
        Self
    }
}

impl RecommendationAlgorithmImpl for PrivacyPreservingCollaborativeFiltering {
    fn recommend(
        &self,
        user_embedding: &UserEmbedding,
        content_embeddings: &[ContentEmbedding],
        config: &RecommendationConfig,
    ) -> Result<Vec<RecommendationScore>> {
        let mut scores = Vec::new();
        
        for content in content_embeddings {
            let similarity = self.compute_privacy_preserving_similarity(
                &user_embedding.embedding,
                &content.embedding,
                config.privacy_budget,
            );
            
            scores.push(RecommendationScore {
                content_id: content.content_id.clone(),
                score: similarity,
                explanation: Some("collaborative filtering with differential privacy".to_string()),
            });
        }
        
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(scores)
    }
    
    fn supports_privacy_preservation(&self) -> bool {
        true
    }
    
    fn algorithm_type(&self) -> RecommendationAlgorithm {
        RecommendationAlgorithm::CollaborativeFiltering
    }
}

impl PrivacyPreservingCollaborativeFiltering {
    fn compute_privacy_preserving_similarity(
        &self,
        user_vec: &[f64],
        content_vec: &[f64],
        privacy_budget: f64,
    ) -> f64 {
        let dot_product: f64 = user_vec.iter()
            .zip(content_vec.iter())
            .map(|(u, c)| u * c)
            .sum();
        
        let user_norm: f64 = user_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        let content_norm: f64 = content_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        let similarity = if user_norm > 0.0 && content_norm > 0.0 {
            dot_product / (user_norm * content_norm)
        } else {
            0.0
        };

        let noise = rand::random::<f64>() * privacy_budget * 0.1;
        (similarity + noise).clamp(0.0, 1.0)
    }
}

struct ContentBasedRecommendation;

impl ContentBasedRecommendation {
    fn new() -> Self {
        Self
    }
}

impl RecommendationAlgorithmImpl for ContentBasedRecommendation {
    fn recommend(
        &self,
        user_embedding: &UserEmbedding,
        content_embeddings: &[ContentEmbedding],
        _config: &RecommendationConfig,
    ) -> Result<Vec<RecommendationScore>> {
        let mut scores = Vec::new();
        
        for content in content_embeddings {
            let similarity = self.compute_content_similarity(
                &user_embedding.embedding,
                &content.embedding,
            );
            
            let adjusted_score = similarity * content.quality_score;
            
            scores.push(RecommendationScore {
                content_id: content.content_id.clone(),
                score: adjusted_score,
                explanation: Some("content-based similarity".to_string()),
            });
        }
        
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(scores)
    }
    
    fn supports_privacy_preservation(&self) -> bool {
        false
    }
    
    fn algorithm_type(&self) -> RecommendationAlgorithm {
        RecommendationAlgorithm::ContentBased
    }
}

impl ContentBasedRecommendation {
    fn compute_content_similarity(&self, user_vec: &[f64], content_vec: &[f64]) -> f64 {
        let dot_product: f64 = user_vec.iter()
            .zip(content_vec.iter())
            .map(|(u, c)| u * c)
            .sum();
        
        let user_norm: f64 = user_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        let content_norm: f64 = content_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if user_norm > 0.0 && content_norm > 0.0 {
            dot_product / (user_norm * content_norm)
        } else {
            0.0
        }
    }
}

struct DifferentialPrivacyRecommendation;

impl DifferentialPrivacyRecommendation {
    fn new() -> Self {
        Self
    }
}

impl RecommendationAlgorithmImpl for DifferentialPrivacyRecommendation {
    fn recommend(
        &self,
        user_embedding: &UserEmbedding,
        content_embeddings: &[ContentEmbedding],
        config: &RecommendationConfig,
    ) -> Result<Vec<RecommendationScore>> {
        let mut scores = Vec::new();
        
        for content in content_embeddings {
            let base_score = self.compute_base_similarity(
                &user_embedding.embedding,
                &content.embedding,
            );
            
            let noisy_score = self.add_differential_privacy_noise(
                base_score,
                config.privacy_budget,
            );
            
            scores.push(RecommendationScore {
                content_id: content.content_id.clone(),
                score: noisy_score,
                explanation: Some("differential privacy recommendation".to_string()),
            });
        }
        
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(scores)
    }
    
    fn supports_privacy_preservation(&self) -> bool {
        true
    }
    
    fn algorithm_type(&self) -> RecommendationAlgorithm {
        RecommendationAlgorithm::PrivacyPreserving
    }
}

impl DifferentialPrivacyRecommendation {
    fn compute_base_similarity(&self, user_vec: &[f64], content_vec: &[f64]) -> f64 {
        user_vec.iter()
            .zip(content_vec.iter())
            .map(|(u, c)| (u - c).abs())
            .sum::<f64>() / user_vec.len() as f64
    }
    
    fn add_differential_privacy_noise(&self, score: f64, privacy_budget: f64) -> f64 {
        let sensitivity = 1.0;
        let scale = sensitivity / privacy_budget;
        let noise = self.sample_laplace_noise(scale);
        (score + noise).clamp(0.0, 1.0)
    }
    
    fn sample_laplace_noise(&self, scale: f64) -> f64 {
        let u: f64 = rand::random::<f64>() - 0.5;
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

struct HybridRecommendation;

impl HybridRecommendation {
    fn new() -> Self {
        Self
    }
}

impl RecommendationAlgorithmImpl for HybridRecommendation {
    fn recommend(
        &self,
        user_embedding: &UserEmbedding,
        content_embeddings: &[ContentEmbedding],
        config: &RecommendationConfig,
    ) -> Result<Vec<RecommendationScore>> {
        let collaborative = PrivacyPreservingCollaborativeFiltering::new();
        let content_based = ContentBasedRecommendation::new();
        
        let collab_scores = collaborative.recommend(user_embedding, content_embeddings, config)?;
        let content_scores = content_based.recommend(user_embedding, content_embeddings, config)?;
        
        let mut hybrid_scores = Vec::new();
        
        for collab_score in &collab_scores {
            if let Some(content_score) = content_scores.iter()
                .find(|s| s.content_id == collab_score.content_id) {
                
                let hybrid_score = 0.6 * collab_score.score + 0.4 * content_score.score;
                
                hybrid_scores.push(RecommendationScore {
                    content_id: collab_score.content_id.clone(),
                    score: hybrid_score,
                    explanation: Some("hybrid collaborative and content-based".to_string()),
                });
            }
        }
        
        hybrid_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(hybrid_scores)
    }
    
    fn supports_privacy_preservation(&self) -> bool {
        true
    }
    
    fn algorithm_type(&self) -> RecommendationAlgorithm {
        RecommendationAlgorithm::Hybrid
    }
}