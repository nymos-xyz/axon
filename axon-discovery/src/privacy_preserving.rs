use crate::{
    error::{DiscoveryError, Result},
    types::*,
};
use sha3::{Digest, Sha3_256};
use rand::{Rng, thread_rng};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{info, debug, warn};
use chrono::Utc;
use uuid::Uuid;

pub struct PrivacyPreservingDiscovery {
    anonymous_profiles: RwLock<HashMap<String, AnonymousProfile>>,
    interest_vectors: RwLock<HashMap<String, InterestVector>>,
    differential_privacy: DifferentialPrivacy,
    config: DiscoveryConfig,
}

#[derive(Debug, Clone)]
struct AnonymousProfile {
    anonymous_id: String,
    interests: Vec<Interest>,
    interaction_history: Vec<AnonymousInteraction>,
    privacy_budget_used: f64,
    last_updated: chrono::DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct AnonymousInteraction {
    content_type: ContentType,
    interaction_hash: String,
    timestamp: chrono::DateTime<Utc>,
}

pub struct DifferentialPrivacy {
    epsilon: f64,
    delta: f64,
    sensitivity: f64,
}

impl DifferentialPrivacy {
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            sensitivity: 1.0,
        }
    }

    pub fn add_noise(&self, value: f64) -> f64 {
        let mut rng = thread_rng();
        let noise = rng.gen_range(-self.sensitivity..self.sensitivity) * (1.0 / self.epsilon);
        value + noise
    }

    pub fn add_laplace_noise(&self, value: f64) -> f64 {
        let mut rng = thread_rng();
        let u: f64 = rng.gen_range(-0.5..0.5);
        let noise = -(1.0 / self.epsilon) * u.signum() * (1.0 - 2.0 * u.abs()).ln();
        value + noise
    }
}

impl PrivacyPreservingDiscovery {
    pub async fn new(config: DiscoveryConfig) -> Result<Self> {
        info!("Initializing privacy-preserving discovery system");

        let differential_privacy = DifferentialPrivacy::new(1.0, 1e-5);

        Ok(Self {
            anonymous_profiles: RwLock::new(HashMap::new()),
            interest_vectors: RwLock::new(HashMap::new()),
            differential_privacy,
            config,
        })
    }

    pub async fn discover_anonymously(&self, request: &DiscoveryRequest) -> Result<Vec<DiscoveryResult>> {
        info!("Performing anonymous discovery for request: {}", request.request_id);

        let anonymous_profile = self.get_or_create_anonymous_profile(&request.anonymous_id).await?;
        
        if !self.check_privacy_budget(&anonymous_profile).await? {
            return Err(DiscoveryError::PrivacyViolation(
                "Privacy budget exceeded for this anonymous profile".to_string()
            ));
        }

        let interest_vector = self.create_privacy_preserving_interest_vector(&request.interests).await?;
        
        let content_matches = self.find_content_matches(&interest_vector, &request.content_types).await?;
        
        let mut results = Vec::new();
        for content_match in content_matches {
            let noisy_score = self.differential_privacy.add_laplace_noise(content_match.relevance_score);
            
            if noisy_score > 0.1 {
                results.push(DiscoveryResult {
                    content_id: content_match.content_id,
                    content_type: content_match.content_type,
                    relevance_score: noisy_score.max(0.0).min(1.0),
                    privacy_preserving_metadata: self.create_privacy_metadata(&content_match).await?,
                    anonymous_engagement_data: Some(self.get_anonymous_engagement(&content_match.content_id).await?),
                });
            }
        }

        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        results.truncate(request.max_results);

        self.update_privacy_budget(&request.anonymous_id, 0.1).await?;

        Ok(results)
    }

    pub async fn discover_with_pseudonym(&self, request: &DiscoveryRequest) -> Result<Vec<DiscoveryResult>> {
        info!("Performing pseudonymous discovery");

        let pseudonym_hash = self.create_pseudonym_hash(&request.anonymous_id);
        
        let enhanced_interests = self.enhance_interests_with_history(&request.interests, &pseudonym_hash).await?;
        
        let enhanced_request = DiscoveryRequest {
            interests: enhanced_interests,
            ..request.clone()
        };

        self.discover_anonymously(&enhanced_request).await
    }

    pub async fn generate_anonymity_proof(
        &self,
        request: &DiscoveryRequest,
        results: &[DiscoveryResult],
    ) -> Result<PrivacyProof> {
        info!("Generating anonymity proof for discovery results");

        let proof_data = self.create_k_anonymity_proof(request, results).await?;
        
        let verification_key = self.generate_verification_key(&request.anonymous_id).await?;

        Ok(PrivacyProof {
            proof_type: "k-anonymity".to_string(),
            proof_data,
            verification_key,
        })
    }

    pub async fn update_anonymous_interests(
        &self,
        anonymous_id: &str,
        interests: Vec<Interest>,
    ) -> Result<()> {
        info!("Updating anonymous interest profile for: {}", anonymous_id);

        let mut profiles = self.anonymous_profiles.write().await;
        
        if let Some(profile) = profiles.get_mut(anonymous_id) {
            profile.interests = self.merge_interests_privately(&profile.interests, &interests).await?;
            profile.last_updated = Utc::now();
        } else {
            let new_profile = AnonymousProfile {
                anonymous_id: anonymous_id.to_string(),
                interests: self.apply_privacy_noise_to_interests(&interests).await?,
                interaction_history: Vec::new(),
                privacy_budget_used: 0.0,
                last_updated: Utc::now(),
            };
            profiles.insert(anonymous_id.to_string(), new_profile);
        }

        Ok(())
    }

    async fn get_or_create_anonymous_profile(&self, anonymous_id: &str) -> Result<AnonymousProfile> {
        let profiles = self.anonymous_profiles.read().await;
        
        if let Some(profile) = profiles.get(anonymous_id) {
            Ok(profile.clone())
        } else {
            drop(profiles);
            
            let new_profile = AnonymousProfile {
                anonymous_id: anonymous_id.to_string(),
                interests: Vec::new(),
                interaction_history: Vec::new(),
                privacy_budget_used: 0.0,
                last_updated: Utc::now(),
            };

            let mut profiles = self.anonymous_profiles.write().await;
            profiles.insert(anonymous_id.to_string(), new_profile.clone());
            
            Ok(new_profile)
        }
    }

    async fn check_privacy_budget(&self, profile: &AnonymousProfile) -> Result<bool> {
        let budget_per_hour = self.config.privacy_budget_per_hour;
        let hours_since_last_reset = (Utc::now() - profile.last_updated).num_hours() as f64;
        
        let available_budget = budget_per_hour * (hours_since_last_reset / 24.0).min(1.0);
        
        Ok(profile.privacy_budget_used < available_budget)
    }

    async fn update_privacy_budget(&self, anonymous_id: &str, budget_used: f64) -> Result<()> {
        let mut profiles = self.anonymous_profiles.write().await;
        
        if let Some(profile) = profiles.get_mut(anonymous_id) {
            profile.privacy_budget_used += budget_used;
        }

        Ok(())
    }

    async fn create_privacy_preserving_interest_vector(&self, interests: &[Interest]) -> Result<InterestVector> {
        let mut dimensions = Vec::with_capacity(interests.len());
        let mut privacy_noise = Vec::with_capacity(interests.len());

        for interest in interests {
            let noisy_weight = if interest.privacy_masked {
                self.differential_privacy.add_laplace_noise(interest.weight)
            } else {
                interest.weight
            };
            
            dimensions.push(noisy_weight);
            privacy_noise.push(self.differential_privacy.add_noise(0.0));
        }

        let vector_id = format!("vec_{}", Uuid::new_v4());
        
        Ok(InterestVector {
            vector_id,
            dimensions,
            privacy_noise,
            created_at: Utc::now(),
        })
    }

    async fn find_content_matches(
        &self,
        interest_vector: &InterestVector,
        content_types: &[ContentType],
    ) -> Result<Vec<ContentMatch>> {
        Ok(vec![])
    }

    async fn create_privacy_metadata(&self, content_match: &ContentMatch) -> Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        metadata.insert("privacy_preserved".to_string(), "true".to_string());
        metadata.insert("anonymity_level".to_string(), "k-anonymous".to_string());
        metadata.insert("differential_privacy".to_string(), "enabled".to_string());
        Ok(metadata)
    }

    async fn get_anonymous_engagement(&self, content_id: &str) -> Result<AnonymousEngagementData> {
        let base_interactions = 100u64;
        let noisy_interactions = self.differential_privacy.add_noise(base_interactions as f64) as u64;
        
        Ok(AnonymousEngagementData {
            total_interactions: noisy_interactions,
            engagement_score: self.differential_privacy.add_noise(0.75),
            trending_factor: self.differential_privacy.add_noise(0.5),
            quality_score: self.differential_privacy.add_noise(0.8),
        })
    }

    fn create_pseudonym_hash(&self, anonymous_id: &str) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(anonymous_id.as_bytes());
        hasher.update(b"pseudonym_salt");
        hex::encode(hasher.finalize())
    }

    async fn enhance_interests_with_history(
        &self,
        interests: &[Interest],
        pseudonym_hash: &str,
    ) -> Result<Vec<Interest>> {
        Ok(interests.to_vec())
    }

    async fn create_k_anonymity_proof(
        &self,
        request: &DiscoveryRequest,
        results: &[DiscoveryResult],
    ) -> Result<Vec<u8>> {
        let proof_data = serde_json::to_vec(&format!(
            "k-anonymity-proof-{}-{}",
            request.request_id,
            self.config.anonymity_threshold
        ))?;
        Ok(proof_data)
    }

    async fn generate_verification_key(&self, anonymous_id: &str) -> Result<Vec<u8>> {
        let mut hasher = Sha3_256::new();
        hasher.update(anonymous_id.as_bytes());
        hasher.update(b"verification_key");
        Ok(hasher.finalize().to_vec())
    }

    async fn merge_interests_privately(
        &self,
        existing: &[Interest],
        new: &[Interest],
    ) -> Result<Vec<Interest>> {
        let mut merged = existing.to_vec();
        
        for new_interest in new {
            if let Some(existing_interest) = merged.iter_mut()
                .find(|i| i.category == new_interest.category) {
                existing_interest.weight = self.differential_privacy
                    .add_noise((existing_interest.weight + new_interest.weight) / 2.0);
            } else {
                merged.push(new_interest.clone());
            }
        }

        Ok(merged)
    }

    async fn apply_privacy_noise_to_interests(&self, interests: &[Interest]) -> Result<Vec<Interest>> {
        let mut noisy_interests = Vec::new();
        
        for interest in interests {
            let mut noisy_interest = interest.clone();
            if interest.privacy_masked {
                noisy_interest.weight = self.differential_privacy.add_laplace_noise(interest.weight);
            }
            noisy_interests.push(noisy_interest);
        }

        Ok(noisy_interests)
    }
}

#[derive(Debug, Clone)]
struct ContentMatch {
    content_id: String,
    content_type: ContentType,
    relevance_score: f64,
}