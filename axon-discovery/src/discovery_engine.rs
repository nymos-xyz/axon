use crate::{
    error::{DiscoveryError, Result},
    types::*,
    privacy_preserving::PrivacyPreservingDiscovery,
    recommendations::RecommendationSystem,
    nymcompute_integration::NymComputeDiscovery,
};
use axon_core::types::ContentHash;
use axon_social::{privacy::PrivacyManager, analytics::AnalyticsEngine};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::Utc;
use uuid::Uuid;

pub struct DiscoveryEngine {
    privacy_discovery: Arc<PrivacyPreservingDiscovery>,
    recommendation_system: Arc<RecommendationSystem>,
    nymcompute_discovery: Arc<NymComputeDiscovery>,
    privacy_manager: Arc<PrivacyManager>,
    analytics: Arc<AnalyticsEngine>,
    config: DiscoveryConfig,
}

#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub max_concurrent_requests: usize,
    pub default_privacy_level: PrivacyLevel,
    pub enable_distributed_processing: bool,
    pub anonymity_threshold: usize,
    pub privacy_budget_per_hour: f64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 100,
            default_privacy_level: PrivacyLevel::Anonymous,
            enable_distributed_processing: true,
            anonymity_threshold: 10,
            privacy_budget_per_hour: 1.0,
        }
    }
}

impl DiscoveryEngine {
    pub async fn new(
        privacy_manager: Arc<PrivacyManager>,
        analytics: Arc<AnalyticsEngine>,
        config: DiscoveryConfig,
    ) -> Result<Self> {
        info!("Initializing Axon Discovery Engine with privacy-first approach");

        let privacy_discovery = Arc::new(
            PrivacyPreservingDiscovery::new(config.clone()).await?
        );
        
        let recommendation_system = Arc::new(
            RecommendationSystem::new(config.clone()).await?
        );
        
        let nymcompute_discovery = Arc::new(
            NymComputeDiscovery::new(config.clone()).await?
        );

        Ok(Self {
            privacy_discovery,
            recommendation_system,
            nymcompute_discovery,
            privacy_manager,
            analytics,
            config,
        })
    }

    pub async fn discover_content(&self, request: DiscoveryRequest) -> Result<DiscoveryResponse> {
        info!("Processing discovery request: {} with privacy level: {:?}", 
              request.request_id, request.privacy_level);

        self.validate_request(&request).await?;

        let start_time = std::time::Instant::now();

        let results = match request.privacy_level {
            PrivacyLevel::Anonymous => {
                self.anonymous_discovery(&request).await?
            },
            PrivacyLevel::Pseudonymous => {
                self.pseudonymous_discovery(&request).await?
            },
            PrivacyLevel::Private => {
                self.private_discovery(&request).await?
            },
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        let privacy_proofs = self.generate_privacy_proofs(&request, &results).await?;

        let processing_metadata = ProcessingMetadata {
            compute_job_id: None,
            processing_time_ms: processing_time,
            privacy_preserved: true,
            distributed_processing: self.config.enable_distributed_processing,
        };

        self.analytics.record_discovery_request(&request, &results).await
            .map_err(|e| DiscoveryError::Internal(format!("Analytics error: {}", e)))?;

        Ok(DiscoveryResponse {
            request_id: request.request_id,
            results,
            privacy_proofs,
            processing_metadata,
            created_at: Utc::now(),
        })
    }

    async fn validate_request(&self, request: &DiscoveryRequest) -> Result<()> {
        if request.interests.is_empty() {
            return Err(DiscoveryError::Internal(
                "Discovery request must include at least one interest".to_string()
            ));
        }

        if request.max_results == 0 || request.max_results > 1000 {
            return Err(DiscoveryError::Internal(
                "Max results must be between 1 and 1000".to_string()
            ));
        }

        if !self.privacy_manager.validate_privacy_level(&request.privacy_level).await {
            return Err(DiscoveryError::PrivacyViolation(
                "Invalid privacy level for current context".to_string()
            ));
        }

        Ok(())
    }

    async fn anonymous_discovery(&self, request: &DiscoveryRequest) -> Result<Vec<DiscoveryResult>> {
        info!("Performing anonymous content discovery");

        if self.config.enable_distributed_processing {
            self.nymcompute_discovery.discover_anonymously(request).await
        } else {
            self.privacy_discovery.discover_anonymously(request).await
        }
    }

    async fn pseudonymous_discovery(&self, request: &DiscoveryRequest) -> Result<Vec<DiscoveryResult>> {
        info!("Performing pseudonymous content discovery");

        let mut results = self.privacy_discovery.discover_with_pseudonym(request).await?;
        
        if self.config.enable_distributed_processing {
            let enhanced_results = self.nymcompute_discovery
                .enhance_discovery_results(&results, request).await?;
            results.extend(enhanced_results);
        }

        Ok(results)
    }

    async fn private_discovery(&self, request: &DiscoveryRequest) -> Result<Vec<DiscoveryResult>> {
        info!("Performing private content discovery");

        self.recommendation_system.generate_private_recommendations(request).await
    }

    async fn generate_privacy_proofs(
        &self,
        request: &DiscoveryRequest,
        results: &[DiscoveryResult],
    ) -> Result<Vec<PrivacyProof>> {
        let mut proofs = Vec::new();

        let anonymity_proof = self.privacy_discovery
            .generate_anonymity_proof(request, results).await?;
        proofs.push(anonymity_proof);

        if self.config.enable_distributed_processing {
            let computation_proof = self.nymcompute_discovery
                .generate_computation_proof(request, results).await?;
            proofs.push(computation_proof);
        }

        Ok(proofs)
    }

    pub async fn discover_users(&self, request: DiscoveryRequest) -> Result<DiscoveryResponse> {
        info!("Discovering users with privacy preservation");

        let mut user_request = request;
        user_request.content_types = vec![ContentType::User];

        self.discover_content(user_request).await
    }

    pub async fn discover_trending(&self, request: DiscoveryRequest) -> Result<DiscoveryResponse> {
        info!("Discovering trending content anonymously");

        let trending_results = self.analytics
            .get_anonymous_trending_content(&request.content_types, request.max_results).await
            .map_err(|e| DiscoveryError::Internal(format!("Trending discovery error: {}", e)))?;

        let discovery_results: Vec<DiscoveryResult> = trending_results
            .into_iter()
            .map(|content| DiscoveryResult {
                content_id: content.content_id,
                content_type: content.content_type,
                relevance_score: content.trending_score,
                privacy_preserving_metadata: content.metadata,
                anonymous_engagement_data: Some(content.engagement_data),
            })
            .collect();

        let privacy_proofs = self.generate_privacy_proofs(&request, &discovery_results).await?;

        Ok(DiscoveryResponse {
            request_id: request.request_id,
            results: discovery_results,
            privacy_proofs,
            processing_metadata: ProcessingMetadata {
                compute_job_id: None,
                processing_time_ms: 0,
                privacy_preserved: true,
                distributed_processing: false,
            },
            created_at: Utc::now(),
        })
    }

    pub async fn update_interest_profile(
        &self,
        anonymous_id: &str,
        interests: Vec<Interest>,
    ) -> Result<()> {
        info!("Updating anonymous interest profile");

        self.privacy_discovery
            .update_anonymous_interests(anonymous_id, interests).await?;

        if self.config.enable_distributed_processing {
            self.nymcompute_discovery
                .sync_interest_updates(anonymous_id).await?;
        }

        Ok(())
    }
}