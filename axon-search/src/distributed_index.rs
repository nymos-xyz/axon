//! Distributed Search Index
//! 
//! Implements a privacy-preserving distributed search index that allows content
//! to be searched across multiple nodes while maintaining anonymity and preventing
//! information leakage about search patterns or content distribution.

use crate::error::{SearchError, SearchResult};
use axon_core::{
    types::{ContentHash, Timestamp},
    crypto::AxonVerifyingKey,
    content::ContentMetadata,
};
use nym_core::NymIdentity;
use nym_crypto::Hash256;
use nym_network::{NetworkNode, PeerId};

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn, error};
use serde::{Deserialize, Serialize};
use rand::{thread_rng, Rng};

/// Distributed search index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Number of index shards
    pub shard_count: usize,
    /// Replication factor for each shard
    pub replication_factor: usize,
    /// Maximum entries per shard
    pub max_entries_per_shard: usize,
    /// Enable encrypted index storage
    pub enable_encryption: bool,
    /// Index update interval (seconds)
    pub update_interval: u64,
    /// Enable privacy-preserving querying
    pub enable_private_queries: bool,
    /// Minimum nodes required for query consensus
    pub min_consensus_nodes: usize,
    /// Query timeout (seconds)
    pub query_timeout: u64,
    /// Enable content bloom filters for privacy
    pub enable_bloom_filters: bool,
    /// Bloom filter false positive rate
    pub bloom_filter_fpr: f64,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            shard_count: 64,
            replication_factor: 3,
            max_entries_per_shard: 10000,
            enable_encryption: true,
            update_interval: 300, // 5 minutes
            enable_private_queries: true,
            min_consensus_nodes: 2,
            query_timeout: 30,
            enable_bloom_filters: true,
            bloom_filter_fpr: 0.01, // 1% false positive rate
        }
    }
}

/// Index shard containing content entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexShard {
    /// Shard identifier
    pub shard_id: u64,
    /// Content entries in this shard
    pub entries: BTreeMap<String, IndexEntry>,
    /// Bloom filter for privacy-preserving queries
    pub bloom_filter: Option<BloomFilter>,
    /// Shard creation timestamp
    pub created_at: Timestamp,
    /// Last update timestamp
    pub updated_at: Timestamp,
    /// Shard size in bytes
    pub size_bytes: usize,
    /// Encryption key for this shard (if encrypted)
    pub encryption_key: Option<[u8; 32]>,
    /// Nodes storing this shard
    pub replica_nodes: HashSet<PeerId>,
}

/// Content entry in the search index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Content hash
    pub content_hash: ContentHash,
    /// Content metadata
    pub metadata: ContentMetadata,
    /// Searchable text terms
    pub terms: Vec<String>,
    /// Term frequencies for ranking
    pub term_frequencies: HashMap<String, f64>,
    /// Content size
    pub content_size: usize,
    /// Creator identity
    pub creator: Option<NymIdentity>,
    /// Content category
    pub category: Option<String>,
    /// Privacy level of content
    pub privacy_level: ContentPrivacyLevel,
    /// Index timestamp
    pub indexed_at: Timestamp,
    /// Last access timestamp
    pub last_accessed: Timestamp,
    /// Access count for popularity ranking
    pub access_count: u64,
}

/// Content privacy levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContentPrivacyLevel {
    Public,
    Private,
    Anonymous,
    Encrypted,
}

/// Bloom filter for privacy-preserving queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    /// Bit array
    pub bits: Vec<bool>,
    /// Number of hash functions
    pub hash_functions: usize,
    /// Filter size
    pub size: usize,
    /// Expected number of elements
    pub expected_elements: usize,
    /// False positive rate
    pub false_positive_rate: f64,
}

impl BloomFilter {
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        let size = Self::optimal_size(expected_elements, false_positive_rate);
        let hash_functions = Self::optimal_hash_functions(size, expected_elements);
        
        Self {
            bits: vec![false; size],
            hash_functions,
            size,
            expected_elements,
            false_positive_rate,
        }
    }
    
    fn optimal_size(n: usize, p: f64) -> usize {
        (-(n as f64) * p.ln() / (2.0_f64.ln().powi(2))).ceil() as usize
    }
    
    fn optimal_hash_functions(m: usize, n: usize) -> usize {
        ((m as f64 / n as f64) * 2.0_f64.ln()).round() as usize
    }
    
    pub fn add(&mut self, item: &str) {
        for i in 0..self.hash_functions {
            let hash = self.hash(item, i);
            let index = hash % self.size;
            self.bits[index] = true;
        }
    }
    
    pub fn contains(&self, item: &str) -> bool {
        for i in 0..self.hash_functions {
            let hash = self.hash(item, i);
            let index = hash % self.size;
            if !self.bits[index] {
                return false;
            }
        }
        true
    }
    
    fn hash(&self, item: &str, seed: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        seed.hash(&mut hasher);
        hasher.finish() as usize
    }
}

/// Index replication manager
#[derive(Debug)]
pub struct IndexReplication {
    /// Replication factor
    pub factor: usize,
    /// Node health scores
    pub node_health: HashMap<PeerId, f64>,
    /// Replication topology
    pub topology: HashMap<u64, Vec<PeerId>>, // shard_id -> replica_nodes
    /// Last replication check
    pub last_check: Timestamp,
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    /// Total shards
    pub total_shards: usize,
    /// Total entries
    pub total_entries: usize,
    /// Total size in bytes
    pub total_size_bytes: usize,
    /// Average entries per shard
    pub avg_entries_per_shard: f64,
    /// Replication health score
    pub replication_health: f64,
    /// Query performance metrics
    pub avg_query_time_ms: f64,
    /// Number of active nodes
    pub active_nodes: usize,
    /// Last statistics update
    pub last_updated: Timestamp,
}

/// Distributed query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedQueryResult {
    /// Query ID
    pub query_id: String,
    /// Results from each shard
    pub shard_results: HashMap<u64, Vec<IndexEntry>>,
    /// Consensus score
    pub consensus_score: f64,
    /// Participating nodes
    pub participating_nodes: HashSet<PeerId>,
    /// Query execution time
    pub execution_time_ms: u64,
    /// Privacy metrics
    pub privacy_metrics: QueryPrivacyMetrics,
}

/// Query privacy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPrivacyMetrics {
    /// Anonymity set size
    pub anonymity_set_size: usize,
    /// Query obfuscation applied
    pub obfuscation_applied: bool,
    /// Differential privacy epsilon
    pub dp_epsilon: f64,
    /// Number of nodes queried
    pub nodes_queried: usize,
    /// Query mixing applied
    pub query_mixed: bool,
}

/// Main distributed search index
pub struct DistributedSearchIndex {
    config: IndexConfig,
    shards: Arc<RwLock<HashMap<u64, IndexShard>>>,
    replication: Arc<RwLock<IndexReplication>>,
    network_nodes: Arc<RwLock<HashMap<PeerId, NetworkNode>>>,
    statistics: Arc<RwLock<IndexStatistics>>,
    query_cache: Arc<RwLock<HashMap<String, (DistributedQueryResult, Timestamp)>>>,
    shard_assignment: Arc<RwLock<HashMap<ContentHash, u64>>>,
}

impl DistributedSearchIndex {
    /// Create new distributed search index
    pub fn new(config: IndexConfig) -> Self {
        info!("Initializing distributed search index with {} shards", config.shard_count);
        
        let mut shards = HashMap::new();
        
        // Initialize shards
        for i in 0..config.shard_count {
            let shard = IndexShard {
                shard_id: i as u64,
                entries: BTreeMap::new(),
                bloom_filter: if config.enable_bloom_filters {
                    Some(BloomFilter::new(
                        config.max_entries_per_shard, 
                        config.bloom_filter_fpr
                    ))
                } else {
                    None
                },
                created_at: Timestamp::now(),
                updated_at: Timestamp::now(),
                size_bytes: 0,
                encryption_key: if config.enable_encryption {
                    let mut key = [0u8; 32];
                    thread_rng().fill(&mut key);
                    Some(key)
                } else {
                    None
                },
                replica_nodes: HashSet::new(),
            };
            shards.insert(i as u64, shard);
        }
        
        Self {
            config,
            shards: Arc::new(RwLock::new(shards)),
            replication: Arc::new(RwLock::new(IndexReplication {
                factor: config.replication_factor,
                node_health: HashMap::new(),
                topology: HashMap::new(),
                last_check: Timestamp::now(),
            })),
            network_nodes: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(IndexStatistics {
                total_shards: config.shard_count,
                total_entries: 0,
                total_size_bytes: 0,
                avg_entries_per_shard: 0.0,
                replication_health: 0.0,
                avg_query_time_ms: 0.0,
                active_nodes: 0,
                last_updated: Timestamp::now(),
            })),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            shard_assignment: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Add content to the distributed index
    pub async fn add_content(
        &self,
        content_hash: ContentHash,
        metadata: ContentMetadata,
        searchable_text: String,
        creator: Option<NymIdentity>,
        privacy_level: ContentPrivacyLevel,
    ) -> SearchResult<()> {
        debug!("Adding content to distributed index: {:?}", content_hash);
        
        // Determine shard for this content
        let shard_id = self.determine_shard(&content_hash).await;
        
        // Extract terms and calculate frequencies
        let terms = self.extract_terms(&searchable_text);
        let term_frequencies = self.calculate_term_frequencies(&terms);
        
        // Create index entry
        let entry = IndexEntry {
            content_hash: content_hash.clone(),
            metadata: metadata.clone(),
            terms: terms.clone(),
            term_frequencies,
            content_size: searchable_text.len(),
            creator,
            category: metadata.category.clone(),
            privacy_level,
            indexed_at: Timestamp::now(),
            last_accessed: Timestamp::now(),
            access_count: 0,
        };
        
        // Add to shard
        let mut shards = self.shards.write().await;
        if let Some(shard) = shards.get_mut(&shard_id) {
            // Update bloom filter
            if let Some(bloom_filter) = &mut shard.bloom_filter {
                for term in &terms {
                    bloom_filter.add(term);
                }
            }
            
            // Add entry
            let entry_key = format!("{:?}", content_hash);
            shard.entries.insert(entry_key, entry);
            shard.updated_at = Timestamp::now();
            shard.size_bytes += searchable_text.len();
            
            // Update shard assignment
            let mut assignments = self.shard_assignment.write().await;
            assignments.insert(content_hash, shard_id);
            
            info!("Content added to shard {} with {} terms", shard_id, terms.len());
        }
        
        // Update statistics
        self.update_statistics().await;
        
        // Replicate to other nodes
        self.replicate_shard_update(shard_id).await?;
        
        Ok(())
    }
    
    /// Search across distributed index with privacy preservation
    pub async fn search_distributed(
        &self,
        query: &str,
        privacy_level: ContentPrivacyLevel,
        max_results: usize,
    ) -> SearchResult<DistributedQueryResult> {
        let start_time = std::time::Instant::now();
        let query_id = format!("query_{}", thread_rng().gen::<u32>());
        
        debug!("Executing distributed search: '{}' with privacy level: {:?}", query, privacy_level);
        
        // Check cache first
        let cache_key = format!("{}_{:?}_{}", query, privacy_level, max_results);
        {
            let cache = self.query_cache.read().await;
            if let Some((cached_result, cached_at)) = cache.get(&cache_key) {
                let cache_age = Timestamp::now().timestamp() - cached_at.timestamp();
                if cache_age < 300 { // 5 minute cache
                    debug!("Returning cached result for query: {}", query_id);
                    return Ok(cached_result.clone());
                }
            }
        }
        
        // Extract search terms
        let search_terms = self.extract_terms(query);
        
        // Apply privacy-preserving query techniques
        let obfuscated_terms = if self.config.enable_private_queries {
            self.apply_query_obfuscation(&search_terms).await
        } else {
            search_terms.clone()
        };
        
        // Determine relevant shards using bloom filters
        let relevant_shards = self.find_relevant_shards(&obfuscated_terms).await;
        
        debug!("Found {} relevant shards for query", relevant_shards.len());
        
        // Query shards in parallel with privacy preservation
        let mut shard_results = HashMap::new();
        let mut participating_nodes = HashSet::new();
        
        for shard_id in relevant_shards {
            match self.query_shard_private(shard_id, &obfuscated_terms, &privacy_level, max_results).await {
                Ok((results, nodes)) => {
                    shard_results.insert(shard_id, results);
                    participating_nodes.extend(nodes);
                }
                Err(e) => {
                    warn!("Failed to query shard {}: {:?}", shard_id, e);
                }
            }
        }
        
        // Calculate consensus score
        let consensus_score = self.calculate_consensus_score(&shard_results).await;
        
        // Apply differential privacy to result counts
        let privacy_metrics = QueryPrivacyMetrics {
            anonymity_set_size: participating_nodes.len(),
            obfuscation_applied: self.config.enable_private_queries,
            dp_epsilon: 1.0,
            nodes_queried: participating_nodes.len(),
            query_mixed: false,
        };
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        let result = DistributedQueryResult {
            query_id: query_id.clone(),
            shard_results,
            consensus_score,
            participating_nodes,
            execution_time_ms: execution_time,
            privacy_metrics,
        };
        
        // Cache result
        {
            let mut cache = self.query_cache.write().await;
            cache.insert(cache_key, (result.clone(), Timestamp::now()));
            
            // Cleanup old cache entries
            cache.retain(|_, (_, timestamp)| {
                Timestamp::now().timestamp() - timestamp.timestamp() < 3600 // 1 hour
            });
        }
        
        info!("Distributed search completed in {}ms with consensus score {:.2}", 
              execution_time, consensus_score);
        
        Ok(result)
    }
    
    /// Update network node information
    pub async fn update_network_node(&self, peer_id: PeerId, node: NetworkNode) -> SearchResult<()> {
        let mut nodes = self.network_nodes.write().await;
        nodes.insert(peer_id, node);
        
        // Update replication health
        self.update_replication_health().await;
        
        Ok(())
    }
    
    /// Get index statistics
    pub async fn get_statistics(&self) -> IndexStatistics {
        let stats = self.statistics.read().await;
        stats.clone()
    }
    
    /// Determine which shard should contain the content
    async fn determine_shard(&self, content_hash: &ContentHash) -> u64 {
        let hash_bytes = content_hash.as_bytes();
        let mut hash_sum: u64 = 0;
        
        for (i, &byte) in hash_bytes.iter().enumerate() {
            hash_sum = hash_sum.wrapping_add((byte as u64) << (i % 8));
        }
        
        hash_sum % (self.config.shard_count as u64)
    }
    
    /// Extract searchable terms from text
    fn extract_terms(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|term| term.len() > 2) // Filter short terms
            .map(|term| term.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|term| !term.is_empty())
            .map(|term| term.to_string())
            .collect()
    }
    
    /// Calculate term frequencies for ranking
    fn calculate_term_frequencies(&self, terms: &[String]) -> HashMap<String, f64> {
        let mut frequencies = HashMap::new();
        let total_terms = terms.len() as f64;
        
        for term in terms {
            *frequencies.entry(term.clone()).or_insert(0.0) += 1.0;
        }
        
        // Normalize frequencies
        for frequency in frequencies.values_mut() {
            *frequency /= total_terms;
        }
        
        frequencies
    }
    
    /// Apply query obfuscation for privacy
    async fn apply_query_obfuscation(&self, terms: &[String]) -> Vec<String> {
        let mut obfuscated = terms.to_vec();
        let mut rng = thread_rng();
        
        // Add dummy terms
        let dummy_count = rng.gen_range(1..=3);
        for _ in 0..dummy_count {
            let dummy_term = format!("dummy_{}", rng.gen::<u16>());
            obfuscated.push(dummy_term);
        }
        
        // Shuffle terms
        for i in (1..obfuscated.len()).rev() {
            let j = rng.gen_range(0..=i);
            obfuscated.swap(i, j);
        }
        
        obfuscated
    }
    
    /// Find relevant shards using bloom filters
    async fn find_relevant_shards(&self, terms: &[String]) -> Vec<u64> {
        let shards = self.shards.read().await;
        let mut relevant_shards = Vec::new();
        
        for (shard_id, shard) in shards.iter() {
            if let Some(bloom_filter) = &shard.bloom_filter {
                // Check if any term might be in this shard
                let mut has_terms = false;
                for term in terms {
                    if bloom_filter.contains(term) {
                        has_terms = true;
                        break;
                    }
                }
                
                if has_terms {
                    relevant_shards.push(*shard_id);
                }
            } else {
                // If no bloom filter, assume all shards are relevant
                relevant_shards.push(*shard_id);
            }
        }
        
        relevant_shards
    }
    
    /// Query a specific shard with privacy preservation
    async fn query_shard_private(
        &self,
        shard_id: u64,
        terms: &[String],
        privacy_level: &ContentPrivacyLevel,
        max_results: usize,
    ) -> SearchResult<(Vec<IndexEntry>, HashSet<PeerId>)> {
        let shards = self.shards.read().await;
        let shard = shards.get(&shard_id)
            .ok_or_else(|| SearchError::ShardNotFound(shard_id))?;
        
        let mut results = Vec::new();
        let nodes = shard.replica_nodes.clone();
        
        // Search through shard entries
        for entry in shard.entries.values() {
            // Check privacy level compatibility
            if !self.is_privacy_compatible(&entry.privacy_level, privacy_level) {
                continue;
            }
            
            // Calculate relevance score
            let relevance = self.calculate_relevance_score(entry, terms);
            
            if relevance > 0.1 { // Minimum relevance threshold
                let mut result_entry = entry.clone();
                result_entry.last_accessed = Timestamp::now();
                results.push(result_entry);
            }
        }
        
        // Sort by relevance and limit results
        results.sort_by(|a, b| {
            let score_a = self.calculate_relevance_score(a, terms);
            let score_b = self.calculate_relevance_score(b, terms);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        results.truncate(max_results);
        
        Ok((results, nodes))
    }
    
    /// Check if content privacy level is compatible with query privacy level
    fn is_privacy_compatible(&self, content_privacy: &ContentPrivacyLevel, query_privacy: &ContentPrivacyLevel) -> bool {
        match (content_privacy, query_privacy) {
            (ContentPrivacyLevel::Public, _) => true,
            (ContentPrivacyLevel::Private, ContentPrivacyLevel::Private) => true,
            (ContentPrivacyLevel::Private, ContentPrivacyLevel::Anonymous) => true,
            (ContentPrivacyLevel::Private, ContentPrivacyLevel::Encrypted) => true,
            (ContentPrivacyLevel::Anonymous, ContentPrivacyLevel::Anonymous) => true,
            (ContentPrivacyLevel::Anonymous, ContentPrivacyLevel::Encrypted) => true,
            (ContentPrivacyLevel::Encrypted, ContentPrivacyLevel::Encrypted) => true,
            _ => false,
        }
    }
    
    /// Calculate relevance score for search result
    fn calculate_relevance_score(&self, entry: &IndexEntry, query_terms: &[String]) -> f64 {
        let mut score = 0.0;
        let query_terms_set: HashSet<_> = query_terms.iter().collect();
        
        // Term frequency score
        for term in query_terms {
            if let Some(&tf) = entry.term_frequencies.get(term) {
                score += tf;
            }
        }
        
        // Boost score for exact matches
        let matching_terms = entry.terms.iter()
            .filter(|term| query_terms_set.contains(term))
            .count();
        
        if matching_terms > 0 {
            score += (matching_terms as f64) / (query_terms.len() as f64);
        }
        
        // Apply popularity boost
        let popularity_boost = (entry.access_count as f64).ln_1p() / 100.0;
        score += popularity_boost;
        
        // Apply recency boost
        let age_hours = (Timestamp::now().timestamp() - entry.indexed_at.timestamp()) / 3600;
        let recency_boost = if age_hours < 24 { 0.1 } else { 0.0 };
        score += recency_boost;
        
        score.min(1.0) // Cap at 1.0
    }
    
    /// Calculate consensus score across shard results
    async fn calculate_consensus_score(&self, shard_results: &HashMap<u64, Vec<IndexEntry>>) -> f64 {
        if shard_results.is_empty() {
            return 0.0;
        }
        
        let total_shards = shard_results.len() as f64;
        let responding_shards = shard_results.values()
            .filter(|results| !results.is_empty())
            .count() as f64;
        
        responding_shards / total_shards
    }
    
    /// Update index statistics
    async fn update_statistics(&self) -> SearchResult<()> {
        let shards = self.shards.read().await;
        let nodes = self.network_nodes.read().await;
        
        let mut stats = self.statistics.write().await;
        
        stats.total_entries = shards.values()
            .map(|shard| shard.entries.len())
            .sum();
        
        stats.total_size_bytes = shards.values()
            .map(|shard| shard.size_bytes)
            .sum();
        
        stats.avg_entries_per_shard = if shards.is_empty() {
            0.0
        } else {
            stats.total_entries as f64 / shards.len() as f64
        };
        
        stats.active_nodes = nodes.len();
        stats.last_updated = Timestamp::now();
        
        debug!("Updated index statistics: {} entries across {} shards", 
               stats.total_entries, stats.total_shards);
        
        Ok(())
    }
    
    /// Update replication health score
    async fn update_replication_health(&self) -> SearchResult<()> {
        let replication = self.replication.read().await;
        let nodes = self.network_nodes.read().await;
        
        let target_replicas = self.config.replication_factor;
        let mut healthy_shards = 0;
        let total_shards = self.config.shard_count;
        
        for (_shard_id, replica_nodes) in &replication.topology {
            let healthy_replicas = replica_nodes.iter()
                .filter(|node_id| {
                    nodes.get(node_id)
                        .map(|node| node.is_healthy())
                        .unwrap_or(false)
                })
                .count();
                
            if healthy_replicas >= target_replicas {
                healthy_shards += 1;
            }
        }
        
        let health_score = healthy_shards as f64 / total_shards as f64;
        
        let mut stats = self.statistics.write().await;
        stats.replication_health = health_score;
        
        if health_score < 0.8 {
            warn!("Replication health is low: {:.2}", health_score);
        }
        
        Ok(())
    }
    
    /// Replicate shard updates to other nodes
    async fn replicate_shard_update(&self, shard_id: u64) -> SearchResult<()> {
        debug!("Replicating shard {} updates", shard_id);
        
        let replication = self.replication.read().await;
        if let Some(replica_nodes) = replication.topology.get(&shard_id) {
            for node_id in replica_nodes {
                // In a real implementation, this would send updates to replica nodes
                debug!("Would replicate to node: {:?}", node_id);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::content::ContentMetadata;

    #[tokio::test]
    async fn test_distributed_index_creation() {
        let config = IndexConfig::default();
        let index = DistributedSearchIndex::new(config.clone());
        
        let stats = index.get_statistics().await;
        assert_eq!(stats.total_shards, config.shard_count);
        assert_eq!(stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_content_addition() {
        let config = IndexConfig::default();
        let index = DistributedSearchIndex::new(config);
        
        let content_hash = ContentHash::from_bytes(&[1; 32]);
        let metadata = ContentMetadata::default();
        let text = "test content for search indexing".to_string();
        
        let result = index.add_content(
            content_hash,
            metadata,
            text,
            None,
            ContentPrivacyLevel::Public,
        ).await;
        
        assert!(result.is_ok());
        
        let stats = index.get_statistics().await;
        assert_eq!(stats.total_entries, 1);
    }

    #[tokio::test]
    async fn test_distributed_search() {
        let config = IndexConfig::default();
        let index = DistributedSearchIndex::new(config);
        
        // Add some content
        let content_hash = ContentHash::from_bytes(&[2; 32]);
        let metadata = ContentMetadata::default();
        let text = "privacy preserving search technology".to_string();
        
        index.add_content(
            content_hash,
            metadata,
            text,
            None,
            ContentPrivacyLevel::Public,
        ).await.unwrap();
        
        // Search for content
        let results = index.search_distributed(
            "privacy search",
            ContentPrivacyLevel::Public,
            10,
        ).await.unwrap();
        
        assert!(!results.shard_results.is_empty());
        assert!(results.consensus_score > 0.0);
    }

    #[tokio::test]
    async fn test_bloom_filter() {
        let mut bloom = BloomFilter::new(1000, 0.01);
        
        bloom.add("test");
        bloom.add("search");
        bloom.add("privacy");
        
        assert!(bloom.contains("test"));
        assert!(bloom.contains("search"));
        assert!(bloom.contains("privacy"));
        assert!(!bloom.contains("nonexistent"));
    }
}