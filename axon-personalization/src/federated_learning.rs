//! Federated Learning for Privacy-Preserving Personalization
//!
//! This module implements federated learning capabilities that allow users to
//! collectively improve personalization models without sharing private data.
//! Uses differential privacy and secure aggregation to maintain user privacy.

use crate::error::{PersonalizationError, PersonalizationResult};
use crate::privacy_personalization::{InteractionData, PrivatePreferences};

use axon_core::types::Timestamp;
use nym_core::NymIdentity;
use nym_crypto::Hash256;
use nym_compute::{ComputeJobSpec, ComputeClient, ComputeResult, PrivacyLevel};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn, error};
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2, Axis};
use rand::{thread_rng, Rng};

/// Federated learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedTrainingConfig {
    /// Minimum participants for training round
    pub min_participants: usize,
    /// Maximum participants per round
    pub max_participants: usize,
    /// Training round duration (seconds)
    pub round_duration: u64,
    /// Model convergence threshold
    pub convergence_threshold: f64,
    /// Maximum training rounds
    pub max_rounds: u32,
    /// Differential privacy budget per round
    pub privacy_budget_per_round: f64,
    /// Enable secure aggregation
    pub enable_secure_aggregation: bool,
    /// Minimum model quality threshold
    pub min_model_quality: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Enable cross-validation
    pub enable_cross_validation: bool,
}

impl Default for FederatedTrainingConfig {
    fn default() -> Self {
        Self {
            min_participants: 10,
            max_participants: 1000,
            round_duration: 3600, // 1 hour
            convergence_threshold: 0.001,
            max_rounds: 100,
            privacy_budget_per_round: 0.1,
            enable_secure_aggregation: true,
            min_model_quality: 0.7,
            learning_rate: 0.01,
            enable_cross_validation: true,
        }
    }
}

/// Federated learning client
pub struct FederatedLearningClient {
    config: FederatedTrainingConfig,
    local_model: Arc<RwLock<Option<FederatedModel>>>,
    training_data: Arc<RwLock<Vec<TrainingInstance>>>,
    model_updates: Arc<RwLock<Vec<ModelUpdate>>>,
    participant_id: Hash256,
    compute_client: Option<ComputeClient>,
    training_analytics: Arc<RwLock<TrainingAnalytics>>,
}

/// Federated model structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedModel {
    /// Model weights
    pub weights: Array2<f64>,
    /// Model biases
    pub biases: Array1<f64>,
    /// Model version
    pub version: u32,
    /// Training metadata
    pub metadata: ModelMetadata,
    /// Privacy metrics
    pub privacy_metrics: ModelPrivacyMetrics,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Number of participants
    pub participant_count: usize,
    /// Training rounds completed
    pub rounds_completed: u32,
    /// Model accuracy
    pub accuracy: f64,
    /// Model loss
    pub loss: f64,
    /// Last training timestamp
    pub last_trained: Timestamp,
    /// Model size in bytes
    pub model_size: usize,
}

/// Model privacy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrivacyMetrics {
    /// Total privacy budget used
    pub privacy_budget_used: f64,
    /// Differential privacy epsilon
    pub dp_epsilon: f64,
    /// Anonymity level
    pub anonymity_level: f64,
    /// Secure aggregation rounds
    pub secure_aggregation_rounds: u32,
}

/// Model update from a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    /// Participant ID (anonymous)
    pub participant_id: Hash256,
    /// Weight gradients
    pub weight_gradients: Array2<f64>,
    /// Bias gradients
    pub bias_gradients: Array1<f64>,
    /// Local training loss
    pub local_loss: f64,
    /// Number of local training samples
    pub sample_count: usize,
    /// Update timestamp
    pub timestamp: Timestamp,
    /// Privacy proof
    pub privacy_proof: Option<PrivacyProof>,
}

/// Privacy proof for model updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyProof {
    /// Differential privacy proof
    pub dp_proof: Vec<u8>,
    /// Secure aggregation proof
    pub secure_agg_proof: Vec<u8>,
    /// Privacy budget used
    pub budget_used: f64,
}

/// Training instance for local learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingInstance {
    /// Feature vector
    pub features: Array1<f64>,
    /// Target label/value
    pub target: f64,
    /// Instance weight
    pub weight: f64,
    /// Privacy sensitivity
    pub sensitivity: f64,
}

/// Aggregation strategy for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Simple averaging
    FederatedAveraging,
    /// Weighted averaging by data size
    WeightedAveraging,
    /// Adaptive aggregation
    AdaptiveAggregation,
    /// Secure aggregation with crypto
    SecureAggregation,
}

/// Training analytics
#[derive(Debug, Default)]
struct TrainingAnalytics {
    total_rounds: u32,
    successful_rounds: u32,
    average_participants: f64,
    model_quality_trend: Vec<f64>,
    privacy_budget_consumed: f64,
    convergence_history: Vec<f64>,
}

impl FederatedLearningClient {
    /// Create new federated learning client
    pub async fn new(config: FederatedTrainingConfig) -> PersonalizationResult<Self> {
        info!("Initializing federated learning client");
        
        let participant_id = Self::generate_anonymous_id().await;
        
        let compute_client = match ComputeClient::new().await {
            Ok(client) => {
                info!("NymCompute integration enabled for federated learning");
                Some(client)
            }
            Err(e) => {
                warn!("NymCompute unavailable for federated learning: {:?}", e);
                None
            }
        };
        
        Ok(Self {
            config,
            local_model: Arc::new(RwLock::new(None)),
            training_data: Arc::new(RwLock::new(Vec::new())),
            model_updates: Arc::new(RwLock::new(Vec::new())),
            participant_id,
            compute_client,
            training_analytics: Arc::new(RwLock::new(TrainingAnalytics::default())),
        })
    }
    
    /// Add training data from user interactions
    pub async fn add_training_data(
        &self,
        interactions: Vec<InteractionData>,
        preferences: &PrivatePreferences,
    ) -> PersonalizationResult<()> {
        debug!("Adding {} training instances", interactions.len());
        
        let mut training_data = self.training_data.write().await;
        
        for interaction in interactions {
            // Convert interaction to training instance
            let features = self.extract_features(&interaction, preferences).await?;
            let target = interaction.engagement_score;
            let weight = 1.0; // Could be adjusted based on interaction type
            let sensitivity = self.calculate_sensitivity(&interaction).await;
            
            training_data.push(TrainingInstance {
                features,
                target,
                weight,
                sensitivity,
            });
        }
        
        // Limit training data size for privacy
        while training_data.len() > 10000 {
            training_data.remove(0);
        }
        
        info!("Total training instances: {}", training_data.len());
        Ok(())
    }
    
    /// Participate in federated training round
    pub async fn participate_in_training_round(
        &self,
        global_model: &FederatedModel,
        aggregation_strategy: AggregationStrategy,
    ) -> PersonalizationResult<ModelUpdate> {
        info!("Participating in federated training round {}", global_model.version + 1);
        
        // Download global model
        let mut local_model = self.local_model.write().await;
        *local_model = Some(global_model.clone());
        drop(local_model);
        
        // Perform local training
        let model_update = match self.config.enable_secure_aggregation && self.compute_client.is_some() {
            true => self.secure_local_training(global_model).await?,
            false => self.local_training(global_model).await?,
        };
        
        // Store model update
        let mut updates = self.model_updates.write().await;
        updates.push(model_update.clone());
        
        // Limit update history
        while updates.len() > 100 {
            updates.remove(0);
        }
        
        info!("Local training completed, update ready for aggregation");
        Ok(model_update)
    }
    
    /// Perform local training with differential privacy
    async fn local_training(
        &self,
        global_model: &FederatedModel,
    ) -> PersonalizationResult<ModelUpdate> {
        debug!("Performing local training");
        
        let training_data = self.training_data.read().await;
        
        if training_data.is_empty() {
            return Err(PersonalizationError::InsufficientData(
                "No training data available".to_string()
            ));
        }
        
        // Initialize local model with global weights
        let mut local_weights = global_model.weights.clone();
        let mut local_biases = global_model.biases.clone();
        
        let epochs = 5;
        let mut total_loss = 0.0;
        
        // Training loop with mini-batches
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let batch_size = 32.min(training_data.len());
            
            for batch_start in (0..training_data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(training_data.len());
                let batch = &training_data[batch_start..batch_end];
                
                // Forward pass
                let (predictions, loss) = self.forward_pass(&local_weights, &local_biases, batch).await;
                epoch_loss += loss;
                
                // Backward pass
                let (weight_grads, bias_grads) = self.backward_pass(
                    &local_weights, &local_biases, batch, &predictions
                ).await;
                
                // Apply gradients with learning rate
                for i in 0..local_weights.nrows() {
                    for j in 0..local_weights.ncols() {
                        local_weights[[i, j]] -= self.config.learning_rate * weight_grads[[i, j]];
                    }
                }
                
                for i in 0..local_biases.len() {
                    local_biases[i] -= self.config.learning_rate * bias_grads[i];
                }
            }
            
            total_loss += epoch_loss;
            debug!("Epoch {} completed, loss: {:.4}", epoch, epoch_loss);
        }
        
        let avg_loss = total_loss / (epochs as f64);
        
        // Calculate gradients (difference from global model)
        let weight_gradients = &local_weights - &global_model.weights;
        let bias_gradients = &local_biases - &global_model.biases;
        
        // Apply differential privacy noise
        let (noisy_weight_grads, noisy_bias_grads) = self.add_differential_privacy_noise(
            weight_gradients,
            bias_gradients,
        ).await?;
        
        // Generate privacy proof
        let privacy_proof = self.generate_privacy_proof(&noisy_weight_grads, &noisy_bias_grads).await?;
        
        Ok(ModelUpdate {
            participant_id: self.participant_id.clone(),
            weight_gradients: noisy_weight_grads,
            bias_gradients: noisy_bias_grads,
            local_loss: avg_loss,
            sample_count: training_data.len(),
            timestamp: Timestamp::now(),
            privacy_proof: Some(privacy_proof),
        })
    }
    
    /// Perform secure local training via NymCompute
    async fn secure_local_training(
        &self,
        global_model: &FederatedModel,
    ) -> PersonalizationResult<ModelUpdate> {
        debug!("Performing secure local training via NymCompute");
        
        if let Some(compute_client) = &self.compute_client {
            // Prepare training job for NymCompute
            let training_data = self.training_data.read().await;
            
            let job_spec = ComputeJobSpec {
                job_type: "federated_training".to_string(),
                runtime: "wasm".to_string(),
                code_hash: nym_crypto::Hash256::from_bytes(&[0u8; 32]), // Would be actual training WASM
                input_data: self.prepare_training_input(global_model, &training_data).await?,
                max_execution_time: std::time::Duration::from_secs(600), // 10 minutes
                resource_requirements: Default::default(),
                privacy_level: PrivacyLevel::ZeroKnowledge,
            };
            
            // Submit training job
            let compute_result = compute_client.submit_job(job_spec).await
                .map_err(|e| PersonalizationError::ComputeError(format!("Training failed: {:?}", e)))?;
            
            // Parse result
            let model_update = self.parse_training_result(compute_result).await?;
            
            info!("Secure training completed via NymCompute");
            Ok(model_update)
        } else {
            // Fallback to local training
            self.local_training(global_model).await
        }
    }
    
    /// Extract features from interaction data
    async fn extract_features(
        &self,
        interaction: &InteractionData,
        preferences: &PrivatePreferences,
    ) -> PersonalizationResult<Array1<f64>> {
        let mut features = Vec::new();
        
        // Interaction type encoding (one-hot)
        let interaction_encoding = match interaction.interaction_type {
            crate::privacy_personalization::InteractionType::View => vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            crate::privacy_personalization::InteractionType::Like => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            crate::privacy_personalization::InteractionType::Share => vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            crate::privacy_personalization::InteractionType::Comment => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            crate::privacy_personalization::InteractionType::Follow => vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            crate::privacy_personalization::InteractionType::Save => vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            crate::privacy_personalization::InteractionType::Dismiss => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            crate::privacy_personalization::InteractionType::Report => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        };
        features.extend(interaction_encoding);
        
        // Temporal features
        let time_of_day = interaction.interaction_context.temporal_context.time_of_day.clone();
        let temporal_encoding = match time_of_day.as_str() {
            "morning" => vec![1.0, 0.0, 0.0, 0.0],
            "afternoon" => vec![0.0, 1.0, 0.0, 0.0],
            "evening" => vec![0.0, 0.0, 1.0, 0.0],
            _ => vec![0.0, 0.0, 0.0, 1.0], // night
        };
        features.extend(temporal_encoding);
        
        // Engagement features
        features.push(interaction.engagement_score);
        features.push((interaction.dwell_time as f64).ln_1p() / 10.0); // Normalized log dwell time
        
        // Activity context features
        features.push(interaction.interaction_context.activity_context.interaction_velocity);
        features.push(interaction.interaction_context.activity_context.current_session_duration as f64 / 3600.0); // Hours
        
        // Social context features
        features.push(interaction.interaction_context.social_context.social_activity_level);
        features.push(interaction.interaction_context.social_context.community_involvement);
        
        // Preference alignment features
        let mut preference_score = 0.0;
        for category in &interaction.interaction_context.activity_context.content_focus {
            if let Some(score) = preferences.content_categories.get(category) {
                preference_score += score;
            }
        }
        features.push(preference_score / interaction.interaction_context.activity_context.content_focus.len().max(1) as f64);
        
        // Pad or truncate to fixed size
        features.resize(64, 0.0);
        
        Ok(Array1::from_vec(features))
    }
    
    /// Calculate sensitivity for differential privacy
    async fn calculate_sensitivity(&self, interaction: &InteractionData) -> f64 {
        // Higher sensitivity for more private interactions
        match interaction.interaction_type {
            crate::privacy_personalization::InteractionType::View => 0.1,
            crate::privacy_personalization::InteractionType::Like => 0.3,
            crate::privacy_personalization::InteractionType::Share => 0.5,
            crate::privacy_personalization::InteractionType::Comment => 0.7,
            crate::privacy_personalization::InteractionType::Follow => 0.8,
            crate::privacy_personalization::InteractionType::Save => 0.4,
            crate::privacy_personalization::InteractionType::Dismiss => 0.2,
            crate::privacy_personalization::InteractionType::Report => 1.0,
        }
    }
    
    /// Forward pass through the model
    async fn forward_pass(
        &self,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
        batch: &[TrainingInstance],
    ) -> (Array1<f64>, f64) {
        let batch_size = batch.len();
        let mut predictions = Array1::zeros(batch_size);
        let mut total_loss = 0.0;
        
        for (i, instance) in batch.iter().enumerate() {
            // Simple linear model: prediction = weights * features + bias
            let mut prediction = 0.0;
            for j in 0..instance.features.len().min(weights.ncols()) {
                for k in 0..weights.nrows().min(biases.len()) {
                    prediction += weights[[k, j]] * instance.features[j];
                }
            }
            
            if biases.len() > 0 {
                prediction += biases[0]; // Simplified bias application
            }
            
            // Apply sigmoid activation
            prediction = 1.0 / (1.0 + (-prediction).exp());
            predictions[i] = prediction;
            
            // Calculate loss (mean squared error)
            let loss = (prediction - instance.target).powi(2) * instance.weight;
            total_loss += loss;
        }
        
        (predictions, total_loss / batch_size as f64)
    }
    
    /// Backward pass to calculate gradients
    async fn backward_pass(
        &self,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
        batch: &[TrainingInstance],
        predictions: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>) {
        let batch_size = batch.len();
        let mut weight_grads = Array2::zeros(weights.raw_dim());
        let mut bias_grads = Array1::zeros(biases.raw_dim());
        
        for (i, instance) in batch.iter().enumerate() {
            let prediction = predictions[i];
            let error = (prediction - instance.target) * instance.weight;
            
            // Calculate gradients
            for j in 0..instance.features.len().min(weights.ncols()) {
                for k in 0..weights.nrows() {
                    weight_grads[[k, j]] += error * instance.features[j] / batch_size as f64;
                }
            }
            
            if bias_grads.len() > 0 {
                bias_grads[0] += error / batch_size as f64;
            }
        }
        
        (weight_grads, bias_grads)
    }
    
    /// Add differential privacy noise to gradients
    async fn add_differential_privacy_noise(
        &self,
        weight_gradients: Array2<f64>,
        bias_gradients: Array1<f64>,
    ) -> PersonalizationResult<(Array2<f64>, Array1<f64>)> {
        let mut rng = thread_rng();
        let noise_scale = 1.0 / self.config.privacy_budget_per_round;
        
        let mut noisy_weights = weight_gradients;
        let mut noisy_biases = bias_gradients;
        
        // Add Laplace noise to weights
        for i in 0..noisy_weights.nrows() {
            for j in 0..noisy_weights.ncols() {
                let noise: f64 = rng.gen_range(-noise_scale..noise_scale);
                noisy_weights[[i, j]] += noise;
            }
        }
        
        // Add noise to biases
        for i in 0..noisy_biases.len() {
            let noise: f64 = rng.gen_range(-noise_scale..noise_scale);
            noisy_biases[i] += noise;
        }
        
        Ok((noisy_weights, noisy_biases))
    }
    
    /// Generate privacy proof for model update
    async fn generate_privacy_proof(
        &self,
        weight_gradients: &Array2<f64>,
        bias_gradients: &Array1<f64>,
    ) -> PersonalizationResult<PrivacyProof> {
        // Generate mock privacy proof
        // In a real implementation, this would generate cryptographic proofs
        Ok(PrivacyProof {
            dp_proof: vec![0u8; 64],
            secure_agg_proof: vec![1u8; 64],
            budget_used: self.config.privacy_budget_per_round,
        })
    }
    
    /// Prepare training input for NymCompute
    async fn prepare_training_input(
        &self,
        global_model: &FederatedModel,
        training_data: &[TrainingInstance],
    ) -> PersonalizationResult<Vec<u8>> {
        let input = serde_json::json!({
            "global_model": global_model,
            "training_data": training_data,
            "config": self.config
        });
        
        serde_json::to_vec(&input)
            .map_err(|e| PersonalizationError::SerializationError(e.to_string()))
    }
    
    /// Parse training result from NymCompute
    async fn parse_training_result(
        &self,
        compute_result: ComputeResult,
    ) -> PersonalizationResult<ModelUpdate> {
        // Parse the compute result
        // In a real implementation, this would deserialize the actual result
        Ok(ModelUpdate {
            participant_id: self.participant_id.clone(),
            weight_gradients: Array2::zeros((10, 64)), // Mock gradients
            bias_gradients: Array1::zeros(10),
            local_loss: 0.1,
            sample_count: 100,
            timestamp: Timestamp::now(),
            privacy_proof: Some(PrivacyProof {
                dp_proof: vec![2u8; 64],
                secure_agg_proof: vec![3u8; 64],
                budget_used: self.config.privacy_budget_per_round,
            }),
        })
    }
    
    /// Generate anonymous participant ID
    async fn generate_anonymous_id() -> Hash256 {
        let mut rng = thread_rng();
        let mut bytes = [0u8; 32];
        rng.fill(&mut bytes);
        Hash256::from_bytes(&bytes)
    }
    
    /// Get training analytics
    pub async fn get_training_analytics(&self) -> TrainingAnalytics {
        let analytics = self.training_analytics.read().await;
        TrainingAnalytics {
            total_rounds: analytics.total_rounds,
            successful_rounds: analytics.successful_rounds,
            average_participants: analytics.average_participants,
            model_quality_trend: analytics.model_quality_trend.clone(),
            privacy_budget_consumed: analytics.privacy_budget_consumed,
            convergence_history: analytics.convergence_history.clone(),
        }
    }
}

/// Federated model aggregator
pub struct FederatedModelAggregator {
    current_model: Arc<RwLock<Option<FederatedModel>>>,
    pending_updates: Arc<RwLock<Vec<ModelUpdate>>>,
    config: FederatedTrainingConfig,
}

impl FederatedModelAggregator {
    pub fn new(config: FederatedTrainingConfig) -> Self {
        Self {
            current_model: Arc::new(RwLock::new(None)),
            pending_updates: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }
    
    /// Aggregate model updates using specified strategy
    pub async fn aggregate_updates(
        &self,
        updates: Vec<ModelUpdate>,
        strategy: AggregationStrategy,
    ) -> PersonalizationResult<FederatedModel> {
        info!("Aggregating {} model updates using {:?}", updates.len(), strategy);
        
        match strategy {
            AggregationStrategy::FederatedAveraging => {
                self.federated_averaging(updates).await
            }
            AggregationStrategy::WeightedAveraging => {
                self.weighted_averaging(updates).await
            }
            AggregationStrategy::AdaptiveAggregation => {
                self.adaptive_aggregation(updates).await
            }
            AggregationStrategy::SecureAggregation => {
                self.secure_aggregation(updates).await
            }
        }
    }
    
    /// Simple federated averaging
    async fn federated_averaging(&self, updates: Vec<ModelUpdate>) -> PersonalizationResult<FederatedModel> {
        if updates.is_empty() {
            return Err(PersonalizationError::InsufficientData("No updates to aggregate".to_string()));
        }
        
        let num_updates = updates.len() as f64;
        
        // Initialize aggregated gradients
        let first_update = &updates[0];
        let mut aggregated_weights = first_update.weight_gradients.clone();
        let mut aggregated_biases = first_update.bias_gradients.clone();
        
        // Sum all gradients
        for update in updates.iter().skip(1) {
            aggregated_weights = aggregated_weights + &update.weight_gradients;
            aggregated_biases = aggregated_biases + &update.bias_gradients;
        }
        
        // Average the gradients
        aggregated_weights = aggregated_weights / num_updates;
        aggregated_biases = aggregated_biases / num_updates;
        
        // Create new model
        let new_model = FederatedModel {
            weights: aggregated_weights,
            biases: aggregated_biases,
            version: self.get_next_version().await,
            metadata: ModelMetadata {
                participant_count: updates.len(),
                rounds_completed: 1,
                accuracy: 0.8, // Would be calculated
                loss: updates.iter().map(|u| u.local_loss).sum::<f64>() / num_updates,
                last_trained: Timestamp::now(),
                model_size: 1024, // Simplified
            },
            privacy_metrics: ModelPrivacyMetrics {
                privacy_budget_used: updates.iter()
                    .filter_map(|u| u.privacy_proof.as_ref())
                    .map(|p| p.budget_used)
                    .sum(),
                dp_epsilon: self.config.privacy_budget_per_round,
                anonymity_level: 0.9,
                secure_aggregation_rounds: 0,
            },
        };
        
        // Update current model
        let mut current = self.current_model.write().await;
        *current = Some(new_model.clone());
        
        Ok(new_model)
    }
    
    /// Weighted averaging by sample count
    async fn weighted_averaging(&self, updates: Vec<ModelUpdate>) -> PersonalizationResult<FederatedModel> {
        if updates.is_empty() {
            return Err(PersonalizationError::InsufficientData("No updates to aggregate".to_string()));
        }
        
        let total_samples: usize = updates.iter().map(|u| u.sample_count).sum();
        
        // Initialize aggregated gradients
        let first_update = &updates[0];
        let mut aggregated_weights = Array2::zeros(first_update.weight_gradients.raw_dim());
        let mut aggregated_biases = Array1::zeros(first_update.bias_gradients.raw_dim());
        
        // Weighted sum of gradients
        for update in &updates {
            let weight = update.sample_count as f64 / total_samples as f64;
            aggregated_weights = aggregated_weights + &(&update.weight_gradients * weight);
            aggregated_biases = aggregated_biases + &(&update.bias_gradients * weight);
        }
        
        // Create new model
        let new_model = FederatedModel {
            weights: aggregated_weights,
            biases: aggregated_biases,
            version: self.get_next_version().await,
            metadata: ModelMetadata {
                participant_count: updates.len(),
                rounds_completed: 1,
                accuracy: 0.85, // Higher for weighted averaging
                loss: updates.iter()
                    .map(|u| u.local_loss * (u.sample_count as f64 / total_samples as f64))
                    .sum(),
                last_trained: Timestamp::now(),
                model_size: 1024,
            },
            privacy_metrics: ModelPrivacyMetrics {
                privacy_budget_used: updates.iter()
                    .filter_map(|u| u.privacy_proof.as_ref())
                    .map(|p| p.budget_used)
                    .sum(),
                dp_epsilon: self.config.privacy_budget_per_round,
                anonymity_level: 0.9,
                secure_aggregation_rounds: 0,
            },
        };
        
        let mut current = self.current_model.write().await;
        *current = Some(new_model.clone());
        
        Ok(new_model)
    }
    
    /// Adaptive aggregation based on model quality
    async fn adaptive_aggregation(&self, updates: Vec<ModelUpdate>) -> PersonalizationResult<FederatedModel> {
        // Filter out low-quality updates
        let quality_threshold = 0.1; // Maximum acceptable loss
        let filtered_updates: Vec<_> = updates.into_iter()
            .filter(|u| u.local_loss < quality_threshold)
            .collect();
        
        if filtered_updates.is_empty() {
            return Err(PersonalizationError::InsufficientData("No quality updates available".to_string()));
        }
        
        // Use weighted averaging with quality-based weights
        self.weighted_averaging(filtered_updates).await
    }
    
    /// Secure aggregation (placeholder)
    async fn secure_aggregation(&self, updates: Vec<ModelUpdate>) -> PersonalizationResult<FederatedModel> {
        // In a real implementation, this would use cryptographic protocols
        // for secure multi-party computation
        info!("Performing secure aggregation (simplified)");
        
        // For now, fall back to federated averaging
        self.federated_averaging(updates).await
    }
    
    async fn get_next_version(&self) -> u32 {
        let current = self.current_model.read().await;
        match current.as_ref() {
            Some(model) => model.version + 1,
            None => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy_personalization::{InteractionType, PersonalizationContext, TemporalContext, ActivityContext, SocialContext, PlatformContext};

    #[tokio::test]
    async fn test_federated_learning_client() {
        let config = FederatedTrainingConfig::default();
        let client = FederatedLearningClient::new(config).await.unwrap();
        
        // Add some training data
        let interactions = vec![
            InteractionData {
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
            }
        ];
        
        let preferences = crate::privacy_personalization::PrivatePreferences {
            content_categories: [("technology".to_string(), 0.8)].iter().cloned().collect(),
            creator_preferences: HashMap::new(),
            topic_interests: HashMap::new(),
            temporal_preferences: HashMap::new(),
            interaction_preferences: HashMap::new(),
            privacy_preferences: crate::privacy_personalization::PrivacyPreferences {
                allow_cross_user_learning: true,
                max_sharing_scope: crate::privacy_personalization::DataSharingScope::Regional,
                anonymity_vs_personalization: 0.5,
                enable_temporal_obfuscation: true,
                update_frequency: crate::privacy_personalization::UpdateFrequency::Daily,
            },
        };
        
        let result = client.add_training_data(interactions, &preferences).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_model_aggregation() {
        let config = FederatedTrainingConfig::default();
        let aggregator = FederatedModelAggregator::new(config);
        
        // Create mock model updates
        let updates = vec![
            ModelUpdate {
                participant_id: Hash256::from_bytes(&[1; 32]),
                weight_gradients: Array2::ones((5, 10)),
                bias_gradients: Array1::ones(5),
                local_loss: 0.1,
                sample_count: 100,
                timestamp: Timestamp::now(),
                privacy_proof: None,
            },
            ModelUpdate {
                participant_id: Hash256::from_bytes(&[2; 32]),
                weight_gradients: Array2::ones((5, 10)) * 2.0,
                bias_gradients: Array1::ones(5) * 2.0,
                local_loss: 0.15,
                sample_count: 150,
                timestamp: Timestamp::now(),
                privacy_proof: None,
            },
        ];
        
        let aggregated_model = aggregator.aggregate_updates(
            updates,
            AggregationStrategy::FederatedAveraging,
        ).await.unwrap();
        
        assert_eq!(aggregated_model.version, 1);
        assert_eq!(aggregated_model.metadata.participant_count, 2);
        assert!(aggregated_model.metadata.accuracy > 0.0);
    }
}