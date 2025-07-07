//! Preference Learning for Privacy-Preserving Personalization
//!
//! This module implements sophisticated preference learning algorithms that
//! learn user preferences from interactions while maintaining complete privacy
//! through differential privacy and federated learning techniques.

use crate::error::{PersonalizationError, PersonalizationResult};
use crate::privacy_personalization::{InteractionData, PrivatePreferences, PersonalizationContext};
use crate::federated_learning::{FederatedLearningClient, FederatedTrainingConfig};

use axon_core::types::Timestamp;
use nym_core::NymIdentity;
use nym_crypto::Hash256;
use nym_compute::{ComputeClient, ComputeJobSpec, PrivacyLevel};

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn, error};
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use rand::{thread_rng, Rng};

/// Preference learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceLearningConfig {
    /// Learning rate for preference updates
    pub learning_rate: f64,
    /// Decay factor for old preferences
    pub decay_factor: f64,
    /// Minimum interactions before learning
    pub min_interactions: usize,
    /// Maximum preference history
    pub max_history: usize,
    /// Enable temporal preference modeling
    pub enable_temporal_modeling: bool,
    /// Enable contextual preference learning
    pub enable_contextual_learning: bool,
    /// Privacy budget for preference updates
    pub privacy_budget: f64,
    /// Enable cross-session preference retention
    pub enable_cross_session_retention: bool,
    /// Preference update frequency (hours)
    pub update_frequency: u64,
}

impl Default for PreferenceLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            decay_factor: 0.95,
            min_interactions: 5,
            max_history: 1000,
            enable_temporal_modeling: true,
            enable_contextual_learning: true,
            privacy_budget: 1.0,
            enable_cross_session_retention: true,
            update_frequency: 24, // Daily updates
        }
    }
}

/// Preference learning strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LearningStrategy {
    /// Simple weighted average
    WeightedAverage,
    /// Exponential moving average
    ExponentialMovingAverage,
    /// Bayesian inference
    BayesianInference,
    /// Neural collaborative filtering
    NeuralCollaborativeFiltering,
    /// Federated learning
    FederatedLearning,
}

/// Preference vector representing user interests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceVector {
    /// Category preferences
    pub category_preferences: HashMap<String, f64>,
    /// Creator preferences
    pub creator_preferences: HashMap<NymIdentity, f64>,
    /// Temporal preferences (time of day, day of week)
    pub temporal_preferences: HashMap<String, f64>,
    /// Content type preferences
    pub content_type_preferences: HashMap<String, f64>,
    /// Interaction type preferences
    pub interaction_preferences: HashMap<String, f64>,
    /// Social context preferences
    pub social_preferences: HashMap<String, f64>,
    /// Confidence scores for each preference
    pub confidence_scores: HashMap<String, f64>,
    /// Last update timestamp
    pub last_updated: Timestamp,
    /// Learning iterations
    pub learning_iterations: u64,
}

impl Default for PreferenceVector {
    fn default() -> Self {
        Self {
            category_preferences: HashMap::new(),
            creator_preferences: HashMap::new(),
            temporal_preferences: HashMap::new(),
            content_type_preferences: HashMap::new(),
            interaction_preferences: HashMap::new(),
            social_preferences: HashMap::new(),
            confidence_scores: HashMap::new(),
            last_updated: Timestamp::now(),
            learning_iterations: 0,
        }
    }
}

/// Interaction history for preference learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionHistory {
    /// Recent interactions
    pub interactions: VecDeque<InteractionData>,
    /// Interaction patterns
    pub patterns: HashMap<String, f64>,
    /// Session boundaries
    pub session_boundaries: Vec<Timestamp>,
    /// Total interactions
    pub total_interactions: u64,
    /// Unique content interacted with
    pub unique_content_count: u64,
}

impl Default for InteractionHistory {
    fn default() -> Self {
        Self {
            interactions: VecDeque::new(),
            patterns: HashMap::new(),
            session_boundaries: Vec::new(),
            total_interactions: 0,
            unique_content_count: 0,
        }
    }
}

/// Preference prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePrediction {
    /// Predicted preference score
    pub preference_score: f64,
    /// Confidence in prediction
    pub confidence: f64,
    /// Contributing factors
    pub factors: HashMap<String, f64>,
    /// Temporal influence
    pub temporal_influence: f64,
    /// Social influence
    pub social_influence: f64,
    /// Content similarity influence
    pub content_similarity: f64,
}

/// Privacy-preserving preference learner
pub struct PreferenceLearner {
    config: PreferenceLearningConfig,
    preference_vector: Arc<RwLock<PreferenceVector>>,
    interaction_history: Arc<RwLock<InteractionHistory>>,
    learning_strategy: LearningStrategy,
    federated_client: Option<FederatedLearningClient>,
    compute_client: Option<ComputeClient>,
    user_id: Hash256,
    learning_analytics: Arc<RwLock<LearningAnalytics>>,
}

/// Learning analytics tracking
#[derive(Debug, Default)]
struct LearningAnalytics {
    total_interactions_processed: u64,
    preference_updates: u64,
    learning_accuracy: f64,
    privacy_budget_used: f64,
    federated_rounds_participated: u32,
    average_confidence: f64,
}

impl PreferenceLearner {
    /// Create new preference learner
    pub async fn new(
        config: PreferenceLearningConfig,
        strategy: LearningStrategy,
    ) -> PersonalizationResult<Self> {
        info!("Initializing preference learner with strategy: {:?}", strategy);
        
        let user_id = Self::generate_user_id().await;
        
        // Initialize federated learning client if enabled
        let federated_client = if strategy == LearningStrategy::FederatedLearning {
            let fed_config = FederatedTrainingConfig {
                privacy_budget_per_round: config.privacy_budget / 10.0,
                learning_rate: config.learning_rate,
                ..Default::default()
            };
            
            match FederatedLearningClient::new(fed_config).await {
                Ok(client) => {
                    info!("Federated learning enabled for preference learning");
                    Some(client)
                }
                Err(e) => {
                    warn!("Failed to initialize federated learning: {:?}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Initialize compute client
        let compute_client = match ComputeClient::new().await {
            Ok(client) => {
                info!("NymCompute integration enabled for preference learning");
                Some(client)
            }
            Err(e) => {
                warn!("NymCompute unavailable for preference learning: {:?}", e);
                None
            }
        };
        
        Ok(Self {
            config,
            preference_vector: Arc::new(RwLock::new(PreferenceVector::default())),
            interaction_history: Arc::new(RwLock::new(InteractionHistory::default())),
            learning_strategy: strategy,
            federated_client,
            compute_client,
            user_id,
            learning_analytics: Arc::new(RwLock::new(LearningAnalytics::default())),
        })
    }
    
    /// Learn from user interaction
    pub async fn learn_from_interaction(
        &self,
        interaction: InteractionData,
    ) -> PersonalizationResult<()> {
        debug!("Learning from interaction: {:?}", interaction.interaction_type);
        
        // Add to interaction history
        self.add_to_history(interaction.clone()).await;
        
        // Update preferences based on learning strategy
        match self.learning_strategy {
            LearningStrategy::WeightedAverage => {
                self.weighted_average_learning(&interaction).await?
            }
            LearningStrategy::ExponentialMovingAverage => {
                self.exponential_moving_average_learning(&interaction).await?
            }
            LearningStrategy::BayesianInference => {
                self.bayesian_inference_learning(&interaction).await?
            }
            LearningStrategy::NeuralCollaborativeFiltering => {
                self.neural_collaborative_filtering(&interaction).await?
            }
            LearningStrategy::FederatedLearning => {
                self.federated_learning(&interaction).await?
            }
        }
        
        // Update analytics
        let mut analytics = self.learning_analytics.write().await;
        analytics.total_interactions_processed += 1;
        analytics.preference_updates += 1;
        
        info!("Preference learning completed for interaction");
        Ok(())
    }
    
    /// Predict user preference for content
    pub async fn predict_preference(
        &self,
        content_features: &HashMap<String, f64>,
        context: &PersonalizationContext,
    ) -> PersonalizationResult<PreferencePrediction> {
        debug!("Predicting preference for content features");
        
        let preference_vector = self.preference_vector.read().await;
        let mut total_score = 0.0;
        let mut confidence = 0.0;
        let mut factors = HashMap::new();
        
        // Category preferences
        if let Some(category) = content_features.get("category") {
            if let Some(pref) = preference_vector.category_preferences.get(&category.to_string()) {
                total_score += pref * 0.3;
                factors.insert("category".to_string(), *pref);
            }
        }
        
        // Temporal preferences
        let temporal_score = self.calculate_temporal_preference(&preference_vector, context).await;
        total_score += temporal_score * 0.2;
        factors.insert("temporal".to_string(), temporal_score);
        
        // Content type preferences
        if let Some(content_type) = content_features.get("content_type") {
            if let Some(pref) = preference_vector.content_type_preferences.get(&content_type.to_string()) {
                total_score += pref * 0.15;
                factors.insert("content_type".to_string(), *pref);
            }
        }
        
        // Social context preferences
        let social_score = self.calculate_social_preference(&preference_vector, context).await;
        total_score += social_score * 0.2;
        factors.insert("social".to_string(), social_score);
        
        // Calculate confidence based on learning iterations and data quality
        confidence = (preference_vector.learning_iterations as f64 / 100.0).min(1.0) * 0.8;
        
        // Add privacy noise
        let noise_scale = 1.0 / self.config.privacy_budget;
        let mut rng = thread_rng();
        let noise: f64 = rng.gen_range(-noise_scale..noise_scale);
        total_score += noise;
        
        // Normalize score
        total_score = total_score.max(0.0).min(1.0);
        
        Ok(PreferencePrediction {
            preference_score: total_score,
            confidence,
            factors,
            temporal_influence: temporal_score,
            social_influence: social_score,
            content_similarity: factors.get("category").unwrap_or(&0.0).clone(),
        })
    }
    
    /// Get current preference vector
    pub async fn get_preference_vector(&self) -> PreferenceVector {
        self.preference_vector.read().await.clone()
    }
    
    /// Get interaction history
    pub async fn get_interaction_history(&self) -> InteractionHistory {
        self.interaction_history.read().await.clone()
    }
    
    /// Add interaction to history
    async fn add_to_history(&self, interaction: InteractionData) {
        let mut history = self.interaction_history.write().await;
        
        history.interactions.push_back(interaction.clone());
        history.total_interactions += 1;
        
        // Limit history size
        while history.interactions.len() > self.config.max_history {
            history.interactions.pop_front();
        }
        
        // Update interaction patterns
        let interaction_type = format!("{:?}", interaction.interaction_type);
        *history.patterns.entry(interaction_type).or_insert(0.0) += 1.0;
        
        // Detect session boundaries (gaps > 30 minutes)
        if let Some(last_interaction) = history.interactions.get(history.interactions.len().saturating_sub(2)) {
            let time_diff = interaction.timestamp.duration_since(&last_interaction.timestamp);
            if time_diff > std::time::Duration::from_secs(1800) { // 30 minutes
                history.session_boundaries.push(interaction.timestamp.clone());
            }
        }
    }
    
    /// Weighted average learning
    async fn weighted_average_learning(
        &self,
        interaction: &InteractionData,
    ) -> PersonalizationResult<()> {
        let mut preferences = self.preference_vector.write().await;
        
        let weight = interaction.engagement_score;
        let learning_rate = self.config.learning_rate;
        
        // Update category preferences
        for category in &interaction.interaction_context.activity_context.content_focus {
            let current = preferences.category_preferences.get(category).unwrap_or(&0.0);
            let updated = current * (1.0 - learning_rate) + weight * learning_rate;
            preferences.category_preferences.insert(category.clone(), updated);
            
            // Update confidence
            let confidence = preferences.confidence_scores.get(category).unwrap_or(&0.0);
            preferences.confidence_scores.insert(category.clone(), confidence + 0.1);
        }
        
        // Update temporal preferences
        let time_key = format!(
            "{}_{}",
            interaction.interaction_context.temporal_context.time_of_day,
            interaction.interaction_context.temporal_context.day_of_week
        );
        let current = preferences.temporal_preferences.get(&time_key).unwrap_or(&0.0);
        let updated = current * (1.0 - learning_rate) + weight * learning_rate;
        preferences.temporal_preferences.insert(time_key, updated);
        
        preferences.learning_iterations += 1;
        preferences.last_updated = Timestamp::now();
        
        Ok(())
    }
    
    /// Exponential moving average learning
    async fn exponential_moving_average_learning(
        &self,
        interaction: &InteractionData,
    ) -> PersonalizationResult<()> {
        let mut preferences = self.preference_vector.write().await;
        
        let alpha = self.config.learning_rate;
        let weight = interaction.engagement_score;
        
        // EMA: new_value = alpha * current_value + (1-alpha) * old_value
        for category in &interaction.interaction_context.activity_context.content_focus {
            let old_value = preferences.category_preferences.get(category).unwrap_or(&0.0);
            let new_value = alpha * weight + (1.0 - alpha) * old_value;
            preferences.category_preferences.insert(category.clone(), new_value);
        }
        
        preferences.learning_iterations += 1;
        preferences.last_updated = Timestamp::now();
        
        Ok(())
    }
    
    /// Bayesian inference learning
    async fn bayesian_inference_learning(
        &self,
        interaction: &InteractionData,
    ) -> PersonalizationResult<()> {
        // Simplified Bayesian update
        let mut preferences = self.preference_vector.write().await;
        
        let evidence = interaction.engagement_score;
        let prior_strength = 1.0;
        
        for category in &interaction.interaction_context.activity_context.content_focus {
            let prior = preferences.category_preferences.get(category).unwrap_or(&0.5);
            let iterations = preferences.learning_iterations as f64;
            
            // Bayesian update: posterior = (prior * prior_strength + evidence * 1) / (prior_strength + 1)
            let posterior = (prior * (prior_strength + iterations) + evidence) / (prior_strength + iterations + 1.0);
            preferences.category_preferences.insert(category.clone(), posterior);
            
            // Update confidence based on number of observations
            let confidence = (iterations + 1.0) / (iterations + 2.0);
            preferences.confidence_scores.insert(category.clone(), confidence);
        }
        
        preferences.learning_iterations += 1;
        preferences.last_updated = Timestamp::now();
        
        Ok(())
    }
    
    /// Neural collaborative filtering
    async fn neural_collaborative_filtering(
        &self,
        interaction: &InteractionData,
    ) -> PersonalizationResult<()> {
        // Simplified neural collaborative filtering using NymCompute if available
        if let Some(compute_client) = &self.compute_client {
            self.neural_collaborative_filtering_compute(interaction).await
        } else {
            // Fallback to simple matrix factorization
            self.matrix_factorization_learning(interaction).await
        }
    }
    
    /// Neural collaborative filtering via NymCompute
    async fn neural_collaborative_filtering_compute(
        &self,
        interaction: &InteractionData,
    ) -> PersonalizationResult<()> {
        let compute_client = self.compute_client.as_ref().unwrap();
        
        // Prepare training data
        let training_data = serde_json::json!({
            "user_id": self.user_id.to_hex(),
            "interaction": interaction,
            "current_preferences": *self.preference_vector.read().await
        });
        
        let job_spec = ComputeJobSpec {
            job_type: "neural_collaborative_filtering".to_string(),
            runtime: "wasm".to_string(),
            code_hash: nym_crypto::Hash256::from_bytes(&[0u8; 32]), // Placeholder
            input_data: serde_json::to_vec(&training_data)
                .map_err(|e| PersonalizationError::SerializationError(e.to_string()))?,
            max_execution_time: std::time::Duration::from_secs(300),
            resource_requirements: Default::default(),
            privacy_level: PrivacyLevel::ZeroKnowledge,
        };
        
        // Submit job
        let _result = compute_client.submit_job(job_spec).await
            .map_err(|e| PersonalizationError::ComputeError(format!("NCF training failed: {:?}", e)))?;
        
        // In a real implementation, we would parse the result and update preferences
        // For now, fall back to simple learning
        self.weighted_average_learning(interaction).await
    }
    
    /// Matrix factorization learning (simplified NCF fallback)
    async fn matrix_factorization_learning(
        &self,
        interaction: &InteractionData,
    ) -> PersonalizationResult<()> {
        // Simplified matrix factorization approach
        let mut preferences = self.preference_vector.write().await;
        
        let latent_factors = 10;
        let learning_rate = self.config.learning_rate;
        
        // Create user and item latent vectors (simplified)
        let user_vector = Array1::from_elem(latent_factors, 0.1);
        let item_vector = Array1::from_elem(latent_factors, 0.1);
        
        // Predicted rating
        let predicted = user_vector.dot(&item_vector);
        let actual = interaction.engagement_score;
        let error = actual - predicted;
        
        // Update preferences based on latent factors
        for category in &interaction.interaction_context.activity_context.content_focus {
            let current = preferences.category_preferences.get(category).unwrap_or(&0.0);
            let gradient = error * learning_rate;
            let updated = current + gradient;
            preferences.category_preferences.insert(category.clone(), updated.max(0.0).min(1.0));
        }
        
        preferences.learning_iterations += 1;
        preferences.last_updated = Timestamp::now();
        
        Ok(())
    }
    
    /// Federated learning
    async fn federated_learning(
        &self,
        interaction: &InteractionData,
    ) -> PersonalizationResult<()> {
        if let Some(federated_client) = &self.federated_client {
            // Add interaction to federated training data
            let interactions = vec![interaction.clone()];
            let private_preferences = self.convert_to_private_preferences().await;
            
            federated_client.add_training_data(interactions, &private_preferences).await
                .map_err(|e| PersonalizationError::FederatedLearningError(e.to_string()))?;
            
            info!("Added interaction to federated learning dataset");
        }
        
        // Also update local preferences
        self.weighted_average_learning(interaction).await
    }
    
    /// Calculate temporal preference score
    async fn calculate_temporal_preference(
        &self,
        preferences: &PreferenceVector,
        context: &PersonalizationContext,
    ) -> f64 {
        let time_key = format!(
            "{}_{}",
            context.temporal_context.time_of_day,
            context.temporal_context.day_of_week
        );
        
        preferences.temporal_preferences.get(&time_key).unwrap_or(&0.5).clone()
    }
    
    /// Calculate social preference score
    async fn calculate_social_preference(
        &self,
        preferences: &PreferenceVector,
        context: &PersonalizationContext,
    ) -> f64 {
        let social_activity = context.social_context.social_activity_level;
        let community_involvement = context.social_context.community_involvement;
        
        // Combine social factors
        (social_activity + community_involvement) / 2.0
    }
    
    /// Convert preference vector to private preferences format
    async fn convert_to_private_preferences(&self) -> PrivatePreferences {
        let preferences = self.preference_vector.read().await;
        
        PrivatePreferences {
            content_categories: preferences.category_preferences.clone(),
            creator_preferences: preferences.creator_preferences.clone(),
            topic_interests: HashMap::new(), // Would be populated from category preferences
            temporal_preferences: preferences.temporal_preferences.clone(),
            interaction_preferences: preferences.interaction_preferences.clone(),
            privacy_preferences: crate::privacy_personalization::PrivacyPreferences {
                allow_cross_user_learning: true,
                max_sharing_scope: crate::privacy_personalization::DataSharingScope::Global,
                anonymity_vs_personalization: 0.5,
                enable_temporal_obfuscation: true,
                update_frequency: crate::privacy_personalization::UpdateFrequency::Daily,
            },
        }
    }
    
    /// Generate anonymous user ID
    async fn generate_user_id() -> Hash256 {
        let mut rng = thread_rng();
        let mut bytes = [0u8; 32];
        rng.fill(&mut bytes);
        Hash256::from_bytes(&bytes)
    }
    
    /// Get learning analytics
    pub async fn get_learning_analytics(&self) -> LearningAnalytics {
        let analytics = self.learning_analytics.read().await;
        LearningAnalytics {
            total_interactions_processed: analytics.total_interactions_processed,
            preference_updates: analytics.preference_updates,
            learning_accuracy: analytics.learning_accuracy,
            privacy_budget_used: analytics.privacy_budget_used,
            federated_rounds_participated: analytics.federated_rounds_participated,
            average_confidence: analytics.average_confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy_personalization::{InteractionType, TemporalContext, ActivityContext, SocialContext, PlatformContext};
    
    #[tokio::test]
    async fn test_preference_learning() {
        let config = PreferenceLearningConfig::default();
        let learner = PreferenceLearner::new(config, LearningStrategy::WeightedAverage)
            .await.unwrap();
        
        let interaction = InteractionData {
            interaction_type: InteractionType::Like,
            content_id: axon_core::types::ContentHash::from_bytes(&[1; 32]),
            engagement_score: 0.8,
            dwell_time: 120,
            interaction_context: PersonalizationContext {
                temporal_context: TemporalContext {
                    time_of_day: "morning".to_string(),
                    day_of_week: "monday".to_string(),
                    season: "spring".to_string(),
                    timezone_offset: 0,
                },
                activity_context: ActivityContext {
                    recent_interactions: vec!["view".to_string()],
                    current_session_duration: 300,
                    interaction_velocity: 0.5,
                    content_focus: vec!["technology".to_string()],
                },
                social_context: SocialContext {
                    social_activity_level: 0.7,
                    recent_social_interactions: 5,
                    social_network_size: 100,
                    community_involvement: 0.6,
                },
                platform_context: PlatformContext {
                    device_type: "mobile".to_string(),
                    screen_size: "small".to_string(),
                    network_speed: "fast".to_string(),
                    available_modalities: vec!["text".to_string()],
                },
            },
            timestamp: Timestamp::now(),
        };
        
        let result = learner.learn_from_interaction(interaction).await;
        assert!(result.is_ok());
        
        let preferences = learner.get_preference_vector().await;
        assert!(preferences.category_preferences.contains_key("technology"));
        assert!(preferences.learning_iterations > 0);
    }
    
    #[tokio::test]
    async fn test_preference_prediction() {
        let config = PreferenceLearningConfig::default();
        let learner = PreferenceLearner::new(config, LearningStrategy::WeightedAverage)
            .await.unwrap();
        
        // Train with some interactions first
        for i in 0..10 {
            let interaction = InteractionData {
                interaction_type: InteractionType::Like,
                content_id: axon_core::types::ContentHash::from_bytes(&[i; 32]),
                engagement_score: 0.8,
                dwell_time: 120,
                interaction_context: PersonalizationContext {
                    temporal_context: TemporalContext {
                        time_of_day: "morning".to_string(),
                        day_of_week: "monday".to_string(),
                        season: "spring".to_string(),
                        timezone_offset: 0,
                    },
                    activity_context: ActivityContext {
                        recent_interactions: vec!["view".to_string()],
                        current_session_duration: 300,
                        interaction_velocity: 0.5,
                        content_focus: vec!["technology".to_string()],
                    },
                    social_context: SocialContext {
                        social_activity_level: 0.7,
                        recent_social_interactions: 5,
                        social_network_size: 100,
                        community_involvement: 0.6,
                    },
                    platform_context: PlatformContext {
                        device_type: "mobile".to_string(),
                        screen_size: "small".to_string(),
                        network_speed: "fast".to_string(),
                        available_modalities: vec!["text".to_string()],
                    },
                },
                timestamp: Timestamp::now(),
            };
            
            learner.learn_from_interaction(interaction).await.unwrap();
        }
        
        // Test prediction
        let content_features = [("category".to_string(), "technology".to_string().parse::<f64>().unwrap_or(1.0))]
            .iter().cloned().collect();
        
        let context = PersonalizationContext {
            temporal_context: TemporalContext {
                time_of_day: "morning".to_string(),
                day_of_week: "monday".to_string(),
                season: "spring".to_string(),
                timezone_offset: 0,
            },
            activity_context: ActivityContext {
                recent_interactions: vec!["view".to_string()],
                current_session_duration: 300,
                interaction_velocity: 0.5,
                content_focus: vec!["technology".to_string()],
            },
            social_context: SocialContext {
                social_activity_level: 0.7,
                recent_social_interactions: 5,
                social_network_size: 100,
                community_involvement: 0.6,
            },
            platform_context: PlatformContext {
                device_type: "mobile".to_string(),
                screen_size: "small".to_string(),
                network_speed: "fast".to_string(),
                available_modalities: vec!["text".to_string()],
            },
        };
        
        let prediction = learner.predict_preference(&content_features, &context).await.unwrap();
        
        assert!(prediction.preference_score >= 0.0 && prediction.preference_score <= 1.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(!prediction.factors.is_empty());
    }
}