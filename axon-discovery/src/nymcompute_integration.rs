use crate::{
    error::{DiscoveryError, Result},
    types::*,
};
use nym_compute::{
    types::{ComputeJob, JobResult, ExecutionEnvironment},
    client::NymComputeClient,
};
use serde_json::Value;
use std::collections::HashMap;
use tokio::time::{timeout, Duration};
use tracing::{info, debug, warn, error};
use chrono::Utc;
use uuid::Uuid;

pub struct NymComputeDiscovery {
    compute_client: NymComputeClient,
    config: DiscoveryConfig,
}

impl NymComputeDiscovery {
    pub async fn new(config: DiscoveryConfig) -> Result<Self> {
        info!("Initializing NymCompute integration for discovery engine");

        let compute_client = NymComputeClient::new()
            .await
            .map_err(|e| DiscoveryError::NymComputeError(format!("Failed to initialize client: {}", e)))?;

        Ok(Self {
            compute_client,
            config,
        })
    }

    pub async fn discover_anonymously(&self, request: &DiscoveryRequest) -> Result<Vec<DiscoveryResult>> {
        info!("Processing anonymous discovery using NymCompute distributed processing");

        let job_spec = self.create_discovery_job_spec(request).await?;
        
        let job_result = self.submit_and_wait_for_job(job_spec).await?;
        
        self.parse_discovery_results(job_result).await
    }

    pub async fn enhance_discovery_results(
        &self,
        base_results: &[DiscoveryResult],
        request: &DiscoveryRequest,
    ) -> Result<Vec<DiscoveryResult>> {
        info!("Enhancing discovery results using distributed processing");

        let enhancement_job = self.create_enhancement_job_spec(base_results, request).await?;
        
        let job_result = self.submit_and_wait_for_job(enhancement_job).await?;
        
        self.parse_enhanced_results(job_result).await
    }

    pub async fn generate_computation_proof(
        &self,
        request: &DiscoveryRequest,
        results: &[DiscoveryResult],
    ) -> Result<PrivacyProof> {
        info!("Generating zero-knowledge proof for computation correctness");

        let proof_job = self.create_proof_generation_job(request, results).await?;
        
        let proof_result = self.submit_and_wait_for_job(proof_job).await?;
        
        self.parse_computation_proof(proof_result).await
    }

    pub async fn sync_interest_updates(&self, anonymous_id: &str) -> Result<()> {
        info!("Syncing interest updates across distributed network");

        let sync_job = self.create_interest_sync_job(anonymous_id).await?;
        
        let _result = self.submit_and_wait_for_job(sync_job).await?;
        
        Ok(())
    }

    async fn create_discovery_job_spec(&self, request: &DiscoveryRequest) -> Result<ComputeJob> {
        let job_data = serde_json::json!({
            "type": "anonymous_discovery",
            "request_id": request.request_id,
            "interests": request.interests,
            "content_types": request.content_types,
            "max_results": request.max_results,
            "privacy_level": request.privacy_level,
            "anonymity_threshold": self.config.anonymity_threshold
        });

        let job = ComputeJob {
            job_id: Uuid::new_v4().to_string(),
            job_type: "privacy_preserving_discovery".to_string(),
            execution_environment: ExecutionEnvironment::WASM,
            code: self.get_discovery_wasm_code().await?,
            input_data: serde_json::to_vec(&job_data)
                .map_err(|e| DiscoveryError::SerializationError(e))?,
            max_compute_time_ms: 30000,
            memory_limit_mb: 256,
            privacy_requirements: vec![
                "zero_knowledge_proof".to_string(),
                "differential_privacy".to_string(),
                "k_anonymity".to_string(),
            ],
            payment_amount: 10,
            created_at: Utc::now(),
        };

        Ok(job)
    }

    async fn create_enhancement_job_spec(
        &self,
        base_results: &[DiscoveryResult],
        request: &DiscoveryRequest,
    ) -> Result<ComputeJob> {
        let job_data = serde_json::json!({
            "type": "result_enhancement",
            "base_results": base_results,
            "request": request,
            "enhancement_type": "semantic_similarity"
        });

        let job = ComputeJob {
            job_id: Uuid::new_v4().to_string(),
            job_type: "discovery_enhancement".to_string(),
            execution_environment: ExecutionEnvironment::Native,
            code: self.get_enhancement_code().await?,
            input_data: serde_json::to_vec(&job_data)
                .map_err(|e| DiscoveryError::SerializationError(e))?,
            max_compute_time_ms: 60000,
            memory_limit_mb: 512,
            privacy_requirements: vec![
                "anonymous_computation".to_string(),
                "result_encryption".to_string(),
            ],
            payment_amount: 25,
            created_at: Utc::now(),
        };

        Ok(job)
    }

    async fn create_proof_generation_job(
        &self,
        request: &DiscoveryRequest,
        results: &[DiscoveryResult],
    ) -> Result<ComputeJob> {
        let proof_data = serde_json::json!({
            "type": "computation_proof",
            "request_hash": self.hash_request(request),
            "results_hash": self.hash_results(results),
            "privacy_guarantees": ["differential_privacy", "k_anonymity", "zero_knowledge"]
        });

        let job = ComputeJob {
            job_id: Uuid::new_v4().to_string(),
            job_type: "zk_proof_generation".to_string(),
            execution_environment: ExecutionEnvironment::TEE,
            code: self.get_proof_generation_code().await?,
            input_data: serde_json::to_vec(&proof_data)
                .map_err(|e| DiscoveryError::SerializationError(e))?,
            max_compute_time_ms: 45000,
            memory_limit_mb: 1024,
            privacy_requirements: vec![
                "trusted_execution".to_string(),
                "zero_knowledge_proof".to_string(),
                "verifiable_computation".to_string(),
            ],
            payment_amount: 50,
            created_at: Utc::now(),
        };

        Ok(job)
    }

    async fn create_interest_sync_job(&self, anonymous_id: &str) -> Result<ComputeJob> {
        let sync_data = serde_json::json!({
            "type": "interest_sync",
            "anonymous_id_hash": self.hash_anonymous_id(anonymous_id),
            "sync_timestamp": Utc::now(),
            "privacy_level": "anonymous"
        });

        let job = ComputeJob {
            job_id: Uuid::new_v4().to_string(),
            job_type: "interest_synchronization".to_string(),
            execution_environment: ExecutionEnvironment::WASM,
            code: self.get_sync_code().await?,
            input_data: serde_json::to_vec(&sync_data)
                .map_err(|e| DiscoveryError::SerializationError(e))?,
            max_compute_time_ms: 15000,
            memory_limit_mb: 128,
            privacy_requirements: vec![
                "anonymous_sync".to_string(),
                "encrypted_state".to_string(),
            ],
            payment_amount: 5,
            created_at: Utc::now(),
        };

        Ok(job)
    }

    async fn submit_and_wait_for_job(&self, job: ComputeJob) -> Result<JobResult> {
        info!("Submitting compute job: {} of type: {}", job.job_id, job.job_type);

        let job_id = self.compute_client
            .submit_job(job)
            .await
            .map_err(|e| DiscoveryError::NymComputeError(format!("Job submission failed: {}", e)))?;

        let timeout_duration = Duration::from_millis(60000);
        
        let result = timeout(timeout_duration, async {
            loop {
                match self.compute_client.get_job_result(&job_id).await {
                    Ok(Some(result)) => return Ok(result),
                    Ok(None) => {
                        tokio::time::sleep(Duration::from_millis(1000)).await;
                        continue;
                    },
                    Err(e) => return Err(DiscoveryError::NymComputeError(
                        format!("Failed to get job result: {}", e)
                    )),
                }
            }
        }).await;

        match result {
            Ok(job_result) => job_result,
            Err(_) => Err(DiscoveryError::NymComputeError(
                "Job execution timeout".to_string()
            )),
        }
    }

    async fn parse_discovery_results(&self, job_result: JobResult) -> Result<Vec<DiscoveryResult>> {
        if !job_result.success {
            return Err(DiscoveryError::NymComputeError(
                format!("Job failed: {}", job_result.error_message.unwrap_or_default())
            ));
        }

        let output_data: Value = serde_json::from_slice(&job_result.output_data)
            .map_err(|e| DiscoveryError::SerializationError(e))?;

        let results: Vec<DiscoveryResult> = serde_json::from_value(output_data["results"].clone())
            .map_err(|e| DiscoveryError::SerializationError(e))?;

        Ok(results)
    }

    async fn parse_enhanced_results(&self, job_result: JobResult) -> Result<Vec<DiscoveryResult>> {
        self.parse_discovery_results(job_result).await
    }

    async fn parse_computation_proof(&self, job_result: JobResult) -> Result<PrivacyProof> {
        if !job_result.success {
            return Err(DiscoveryError::NymComputeError(
                format!("Proof generation failed: {}", job_result.error_message.unwrap_or_default())
            ));
        }

        let output_data: Value = serde_json::from_slice(&job_result.output_data)
            .map_err(|e| DiscoveryError::SerializationError(e))?;

        let proof = PrivacyProof {
            proof_type: "zero_knowledge_computation".to_string(),
            proof_data: hex::decode(output_data["proof_data"].as_str().unwrap_or_default())
                .map_err(|e| DiscoveryError::CryptoError(format!("Invalid proof data: {}", e)))?,
            verification_key: hex::decode(output_data["verification_key"].as_str().unwrap_or_default())
                .map_err(|e| DiscoveryError::CryptoError(format!("Invalid verification key: {}", e)))?,
        };

        Ok(proof)
    }

    async fn get_discovery_wasm_code(&self) -> Result<Vec<u8>> {
        Ok(include_bytes!("../wasm/discovery_engine.wasm").to_vec())
    }

    async fn get_enhancement_code(&self) -> Result<Vec<u8>> {
        Ok(b"enhancement_algorithm_native_code".to_vec())
    }

    async fn get_proof_generation_code(&self) -> Result<Vec<u8>> {
        Ok(b"zk_proof_generation_tee_code".to_vec())
    }

    async fn get_sync_code(&self) -> Result<Vec<u8>> {
        Ok(include_bytes!("../wasm/interest_sync.wasm").to_vec())
    }

    fn hash_request(&self, request: &DiscoveryRequest) -> String {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        let request_bytes = serde_json::to_vec(request).unwrap_or_default();
        hasher.update(&request_bytes);
        hex::encode(hasher.finalize())
    }

    fn hash_results(&self, results: &[DiscoveryResult]) -> String {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        let results_bytes = serde_json::to_vec(results).unwrap_or_default();
        hasher.update(&results_bytes);
        hex::encode(hasher.finalize())
    }

    fn hash_anonymous_id(&self, anonymous_id: &str) -> String {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(anonymous_id.as_bytes());
        hasher.update(b"discovery_sync_salt");
        hex::encode(hasher.finalize())
    }
}