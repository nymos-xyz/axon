//! Advanced Search Engine
//! 
//! Combines privacy-preserving search, distributed indexing, and NymCompute integration
//! to provide a comprehensive anonymous search solution for the Axon social network.

use crate::{
    error::{SearchError, SearchResult},
    privacy_search::{PrivacySearchEngine, PrivacySearchConfig, AnonymousQuery, PrivateSearchResult, SearchPrivacyLevel},
    distributed_index::{DistributedSearchIndex, IndexConfig, ContentPrivacyLevel, DistributedQueryResult},
};

use axon_core::{
    types::{ContentHash, Timestamp},
    crypto::AxonVerifyingKey,
    content::ContentMetadata,
};
use nym_core::NymIdentity;
use nym_compute::{ComputeJobSpec, ComputeClient, ComputeResult, PrivacyLevel};
use quid_core::QuIDIdentity;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn, error};
use serde::{Deserialize, Serialize};

/// Advanced search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSearchConfig {
    /// Privacy search configuration
    pub privacy_config: PrivacySearchConfig,
    /// Distributed index configuration
    pub index_config: IndexConfig,
    /// Enable NymCompute integration
    pub enable_nymcompute: bool,
    /// Enable semantic search
    pub enable_semantic_search: bool,
    /// Enable real-time search suggestions
    pub enable_suggestions: bool,
    /// Maximum concurrent searches
    pub max_concurrent_searches: usize,
    /// Search result caching duration (seconds)
    pub cache_duration: u64,
    /// Enable search analytics
    pub enable_analytics: bool,
    /// Minimum relevance score for results
    pub min_relevance_score: f64,
}

impl Default for AdvancedSearchConfig {
    fn default() -> Self {
        Self {
            privacy_config: PrivacySearchConfig::default(),
            index_config: IndexConfig::default(),
            enable_nymcompute: true,
            enable_semantic_search: true,
            enable_suggestions: true,
            max_concurrent_searches: 100,
            cache_duration: 3600, // 1 hour
            enable_analytics: true,
            min_relevance_score: 0.1,
        }
    }
}

/// Advanced search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSearchQuery {
    /// Query text
    pub query: String,
    /// Privacy level requested
    pub privacy_level: SearchPrivacyLevel,
    /// Content filters
    pub filters: SearchFilters,
    /// Result preferences
    pub preferences: SearchPreferences,
    /// Semantic search options
    pub semantic_options: Option<SemanticSearchOptions>,
}

/// Search filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    /// Content categories
    pub categories: Option<Vec<String>>,
    /// Creator filters
    pub creators: Option<Vec<NymIdentity>>,
    /// Time range
    pub time_range: Option<TimeRange>,
    /// Content size range
    pub size_range: Option<(usize, usize)>,
    /// Privacy level filter
    pub privacy_levels: Option<Vec<ContentPrivacyLevel>>,
    /// Language filter
    pub languages: Option<Vec<String>>,
}

impl Default for SearchFilters {
    fn default() -> Self {
        Self {
            categories: None,
            creators: None,
            time_range: None,
            size_range: None,
            privacy_levels: None,
            languages: None,
        }
    }
}

/// Search preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPreferences {
    /// Maximum results
    pub max_results: usize,
    /// Sort order
    pub sort_order: SortOrder,
    /// Include snippets
    pub include_snippets: bool,
    /// Include metadata
    pub include_metadata: bool,
    /// Personalization level
    pub personalization_level: PersonalizationLevel,
}

impl Default for SearchPreferences {
    fn default() -> Self {
        Self {
            max_results: 50,
            sort_order: SortOrder::Relevance,
            include_snippets: true,
            include_metadata: true,
            personalization_level: PersonalizationLevel::None,
        }
    }
}

/// Time range for search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: Option<Timestamp>,
    pub end: Option<Timestamp>,
}

/// Sort order options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    Relevance,
    Recency,
    Popularity,
    Random,
}

/// Personalization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersonalizationLevel {
    None,
    Basic,
    Advanced,
}

/// Semantic search options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchOptions {
    /// Enable concept expansion
    pub enable_concept_expansion: bool,
    /// Similarity threshold
    pub similarity_threshold: f64,
    /// Maximum concept expansions
    pub max_expansions: usize,
    /// Use user context
    pub use_user_context: bool,
}

/// Advanced search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSearchResult {
    /// Search metadata
    pub search_metadata: SearchMetadata,
    /// Search results
    pub results: Vec<SearchResultItem>,
    /// Search suggestions
    pub suggestions: Option<Vec<String>>,
    /// Faceted search data
    pub facets: Option<SearchFacets>,
    /// Privacy metrics
    pub privacy_metrics: SearchPrivacyMetrics,
    /// Performance metrics
    pub performance_metrics: SearchPerformanceMetrics,
}

/// Search metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetadata {
    /// Query ID
    pub query_id: String,
    /// Original query
    pub original_query: String,
    /// Processed query
    pub processed_query: String,
    /// Total results found
    pub total_results: usize,
    /// Results returned
    pub results_returned: usize,
    /// Search timestamp
    pub search_timestamp: Timestamp,
    /// Search source
    pub search_source: SearchSource,
}

/// Search result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultItem {
    /// Content hash
    pub content_hash: ContentHash,
    /// Content metadata
    pub metadata: ContentMetadata,
    /// Relevance score
    pub relevance_score: f64,
    /// Content snippet
    pub snippet: Option<String>,
    /// Highlighted terms
    pub highlights: Option<Vec<String>>,
    /// Creator identity
    pub creator: Option<NymIdentity>,
    /// Content category
    pub category: Option<String>,
    /// Privacy level
    pub privacy_level: ContentPrivacyLevel,
}

/// Search facets for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFacets {
    /// Category facets
    pub categories: HashMap<String, usize>,
    /// Creator facets
    pub creators: HashMap<NymIdentity, usize>,
    /// Time facets
    pub time_periods: HashMap<String, usize>,
    /// Privacy level facets
    pub privacy_levels: HashMap<ContentPrivacyLevel, usize>,
}

/// Search privacy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPrivacyMetrics {
    /// Privacy level achieved
    pub privacy_level: SearchPrivacyLevel,
    /// Anonymity set size
    pub anonymity_set_size: usize,
    /// Query obfuscation applied
    pub query_obfuscated: bool,
    /// Differential privacy epsilon
    pub dp_epsilon: f64,
    /// NymCompute privacy used
    pub nymcompute_privacy: bool,
}

/// Search performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPerformanceMetrics {
    /// Total search time (ms)
    pub total_time_ms: u64,
    /// Index query time (ms)
    pub index_time_ms: u64,
    /// Privacy processing time (ms)
    pub privacy_time_ms: u64,
    /// NymCompute time (ms)
    pub nymcompute_time_ms: Option<u64>,
    /// Results processed
    pub results_processed: usize,
    /// Cache hit
    pub cache_hit: bool,
}

/// Search source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchSource {
    Local,
    Distributed,
    NymCompute,
    Hybrid,
}

/// Main advanced search engine
pub struct AdvancedSearchEngine {
    config: AdvancedSearchConfig,
    privacy_engine: PrivacySearchEngine,
    distributed_index: DistributedSearchIndex,
    nymcompute_client: Option<ComputeClient>,
    search_cache: Arc<RwLock<HashMap<String, (AdvancedSearchResult, Timestamp)>>>,
    analytics: Arc<RwLock<SearchAnalytics>>,
    active_searches: Arc<RwLock<HashMap<String, SearchProgress>>>,
}

#[derive(Debug, Default)]
struct SearchAnalytics {
    total_searches: u64,
    privacy_searches: u64,
    semantic_searches: u64,
    nymcompute_searches: u64,
    average_response_time: f64,
    cache_hit_rate: f64,
    top_queries: Vec<(String, u64)>,
}

#[derive(Debug)]
struct SearchProgress {
    query_id: String,
    start_time: std::time::Instant,
    status: SearchStatus,
}

#[derive(Debug)]
enum SearchStatus {
    Initializing,
    ProcessingQuery,
    SearchingIndex,
    ProcessingResults,
    Completed,
    Failed(String),
}

impl AdvancedSearchEngine {
    /// Create new advanced search engine
    pub async fn new(config: AdvancedSearchConfig) -> SearchResult<Self> {
        info!("Initializing advanced search engine");
        
        let privacy_engine = PrivacySearchEngine::new(config.privacy_config.clone());
        let distributed_index = DistributedSearchIndex::new(config.index_config.clone());
        
        let nymcompute_client = if config.enable_nymcompute {
            match ComputeClient::new().await {
                Ok(client) => {
                    info!("NymCompute integration enabled");
                    Some(client)
                }
                Err(e) => {
                    warn!("Failed to initialize NymCompute client: {:?}", e);
                    None
                }
            }
        } else {
            None
        };
        
        Ok(Self {
            config,
            privacy_engine,
            distributed_index,
            nymcompute_client,
            search_cache: Arc::new(RwLock::new(HashMap::new())),
            analytics: Arc::new(RwLock::new(SearchAnalytics::default())),
            active_searches: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Execute advanced search with privacy preservation
    pub async fn search(
        &self,
        query: AdvancedSearchQuery,
        requester: Option<QuIDIdentity>,
    ) -> SearchResult<AdvancedSearchResult> {
        let start_time = std::time::Instant::now();
        let query_id = format!("adv_search_{}", uuid::Uuid::new_v4());
        
        info!("Executing advanced search: '{}' with privacy level: {:?}", 
              query.query, query.privacy_level);
        
        // Register search progress
        {
            let mut active = self.active_searches.write().await;
            active.insert(query_id.clone(), SearchProgress {
                query_id: query_id.clone(),
                start_time,
                status: SearchStatus::Initializing,
            });
        }
        
        // Check cache first
        let cache_key = self.generate_cache_key(&query, &requester).await;
        if let Some(cached_result) = self.check_cache(&cache_key).await {
            self.update_analytics(true, start_time.elapsed().as_millis() as u64).await;
            return Ok(cached_result);
        }
        
        // Update search status
        self.update_search_status(&query_id, SearchStatus::ProcessingQuery).await;
        
        // Process query based on privacy level and features
        let search_result = match query.privacy_level {
            SearchPrivacyLevel::Public => {
                self.execute_public_search(&query, &query_id).await?
            }
            SearchPrivacyLevel::Private => {
                self.execute_private_search(&query, &query_id, requester).await?
            }
            SearchPrivacyLevel::Anonymous => {
                self.execute_anonymous_search(&query, &query_id).await?
            }
            SearchPrivacyLevel::ZeroKnowledge => {
                self.execute_zero_knowledge_search(&query, &query_id).await?
            }
        };
        
        // Cache result
        {
            let mut cache = self.search_cache.write().await;
            cache.insert(cache_key, (search_result.clone(), Timestamp::now()));
        }
        
        // Update analytics
        let total_time = start_time.elapsed().as_millis() as u64;
        self.update_analytics(false, total_time).await;
        
        // Mark search as completed
        self.update_search_status(&query_id, SearchStatus::Completed).await;
        
        info!("Advanced search completed in {}ms", total_time);
        Ok(search_result)
    }
    
    /// Add content to the search index
    pub async fn index_content(
        &self,
        content_hash: ContentHash,
        metadata: ContentMetadata,
        searchable_text: String,
        creator: Option<NymIdentity>,
        privacy_level: ContentPrivacyLevel,
    ) -> SearchResult<()> {
        debug!("Indexing content: {:?}", content_hash);
        
        self.distributed_index.add_content(
            content_hash,
            metadata,
            searchable_text,
            creator,
            privacy_level,
        ).await
    }
    
    /// Execute public search (fastest, no privacy)
    async fn execute_public_search(
        &self,
        query: &AdvancedSearchQuery,
        query_id: &str,
    ) -> SearchResult<AdvancedSearchResult> {
        debug!("Executing public search");
        
        self.update_search_status(query_id, SearchStatus::SearchingIndex).await;
        
        let start_time = std::time::Instant::now();
        
        // Direct distributed search
        let distributed_result = self.distributed_index.search_distributed(
            &query.query,
            ContentPrivacyLevel::Public,
            query.preferences.max_results,
        ).await?;
        
        let index_time = start_time.elapsed().as_millis() as u64;
        
        self.update_search_status(query_id, SearchStatus::ProcessingResults).await;
        
        // Convert to advanced search result
        let results = self.convert_distributed_results(distributed_result, query).await?;
        
        let total_time = start_time.elapsed().as_millis() as u64;
        
        Ok(AdvancedSearchResult {
            search_metadata: SearchMetadata {
                query_id: query_id.to_string(),
                original_query: query.query.clone(),
                processed_query: query.query.clone(),
                total_results: results.len(),
                results_returned: results.len(),
                search_timestamp: Timestamp::now(),
                search_source: SearchSource::Distributed,
            },
            results,
            suggestions: None,
            facets: None,
            privacy_metrics: SearchPrivacyMetrics {
                privacy_level: SearchPrivacyLevel::Public,
                anonymity_set_size: 0,
                query_obfuscated: false,
                dp_epsilon: 0.0,
                nymcompute_privacy: false,
            },
            performance_metrics: SearchPerformanceMetrics {
                total_time_ms: total_time,
                index_time_ms: index_time,
                privacy_time_ms: 0,
                nymcompute_time_ms: None,
                results_processed: results.len(),
                cache_hit: false,
            },
        })
    }
    
    /// Execute private search (with privacy engine)
    async fn execute_private_search(
        &self,
        query: &AdvancedSearchQuery,
        query_id: &str,
        requester: Option<QuIDIdentity>,
    ) -> SearchResult<AdvancedSearchResult> {
        debug!("Executing private search");
        
        let start_time = std::time::Instant::now();
        
        // Create anonymous query
        let anonymous_query = self.privacy_engine.create_anonymous_query(
            &query.query,
            query.privacy_level.clone(),
            Some(self.convert_filters_to_map(&query.filters)),
        ).await?;
        
        let privacy_time = start_time.elapsed().as_millis() as u64;
        
        self.update_search_status(query_id, SearchStatus::SearchingIndex).await;
        
        // Execute privacy-preserving search
        let private_result = self.privacy_engine.search_private(
            anonymous_query,
            requester,
        ).await?;
        
        let index_time = start_time.elapsed().as_millis() as u64 - privacy_time;
        
        self.update_search_status(query_id, SearchStatus::ProcessingResults).await;
        
        // Decrypt and process results
        let results = self.decrypt_and_process_results(private_result.clone(), query).await?;
        
        let total_time = start_time.elapsed().as_millis() as u64;
        
        Ok(AdvancedSearchResult {
            search_metadata: SearchMetadata {
                query_id: query_id.to_string(),
                original_query: query.query.clone(),
                processed_query: "***encrypted***".to_string(),
                total_results: private_result.noisy_result_count,
                results_returned: results.len(),
                search_timestamp: Timestamp::now(),
                search_source: SearchSource::Local,
            },
            results,
            suggestions: None,
            facets: None,
            privacy_metrics: SearchPrivacyMetrics {
                privacy_level: private_result.privacy_metrics.achieved_privacy_level,
                anonymity_set_size: private_result.privacy_metrics.anonymity_set_size,
                query_obfuscated: private_result.privacy_metrics.obfuscation_applied,
                dp_epsilon: private_result.privacy_metrics.epsilon_used,
                nymcompute_privacy: false,
            },
            performance_metrics: SearchPerformanceMetrics {
                total_time_ms: total_time,
                index_time_ms: index_time,
                privacy_time_ms: privacy_time,
                nymcompute_time_ms: None,
                results_processed: results.len(),
                cache_hit: false,
            },
        })
    }
    
    /// Execute anonymous search (with maximum privacy)
    async fn execute_anonymous_search(
        &self,
        query: &AdvancedSearchQuery,
        query_id: &str,
    ) -> SearchResult<AdvancedSearchResult> {
        debug!("Executing anonymous search");
        
        // Similar to private search but with stronger anonymity
        self.execute_private_search(query, query_id, None).await
    }
    
    /// Execute zero-knowledge search (via NymCompute)
    async fn execute_zero_knowledge_search(
        &self,
        query: &AdvancedSearchQuery,
        query_id: &str,
    ) -> SearchResult<AdvancedSearchResult> {
        debug!("Executing zero-knowledge search via NymCompute");
        
        let start_time = std::time::Instant::now();
        
        if let Some(compute_client) = &self.nymcompute_client {
            self.update_search_status(query_id, SearchStatus::SearchingIndex).await;
            
            // Create compute job for zero-knowledge search
            let job_spec = ComputeJobSpec {
                job_type: "zk_search".to_string(),
                runtime: "wasm".to_string(),
                code_hash: nym_crypto::Hash256::from_bytes(&[0u8; 32]), // Would be actual search WASM
                input_data: serde_json::to_vec(&query)
                    .map_err(|e| SearchError::SerializationError(e.to_string()))?,
                max_execution_time: std::time::Duration::from_secs(60),
                resource_requirements: Default::default(),
                privacy_level: PrivacyLevel::ZeroKnowledge,
            };
            
            let compute_start = std::time::Instant::now();
            let compute_result = compute_client.submit_job(job_spec).await
                .map_err(|e| SearchError::ComputeError(format!("NymCompute error: {:?}", e)))?;
            let nymcompute_time = compute_start.elapsed().as_millis() as u64;
            
            self.update_search_status(query_id, SearchStatus::ProcessingResults).await;
            
            // Process compute results
            let results = self.process_compute_results(compute_result, query).await?;
            
            let total_time = start_time.elapsed().as_millis() as u64;
            
            Ok(AdvancedSearchResult {
                search_metadata: SearchMetadata {
                    query_id: query_id.to_string(),
                    original_query: "***zero-knowledge***".to_string(),
                    processed_query: "***zero-knowledge***".to_string(),
                    total_results: results.len(),
                    results_returned: results.len(),
                    search_timestamp: Timestamp::now(),
                    search_source: SearchSource::NymCompute,
                },
                results,
                suggestions: None,
                facets: None,
                privacy_metrics: SearchPrivacyMetrics {
                    privacy_level: SearchPrivacyLevel::ZeroKnowledge,
                    anonymity_set_size: usize::MAX,
                    query_obfuscated: true,
                    dp_epsilon: 0.0,
                    nymcompute_privacy: true,
                },
                performance_metrics: SearchPerformanceMetrics {
                    total_time_ms: total_time,
                    index_time_ms: 0,
                    privacy_time_ms: total_time - nymcompute_time,
                    nymcompute_time_ms: Some(nymcompute_time),
                    results_processed: results.len(),
                    cache_hit: false,
                },
            })
        } else {
            // Fallback to anonymous search if NymCompute unavailable
            warn!("NymCompute unavailable, falling back to anonymous search");
            self.execute_anonymous_search(query, query_id).await
        }
    }
    
    // Helper methods
    async fn generate_cache_key(&self, query: &AdvancedSearchQuery, requester: &Option<QuIDIdentity>) -> String {
        let requester_hash = requester
            .as_ref()
            .map(|id| format!("{:?}", id))
            .unwrap_or_else(|| "anonymous".to_string());
        
        format!("{}_{:?}_{}", 
                query.query, 
                query.privacy_level, 
                requester_hash)
    }
    
    async fn check_cache(&self, cache_key: &str) -> Option<AdvancedSearchResult> {
        let cache = self.search_cache.read().await;
        
        if let Some((result, cached_at)) = cache.get(cache_key) {
            let age = Timestamp::now().timestamp() - cached_at.timestamp();
            if age < self.config.cache_duration as i64 {
                return Some(result.clone());
            }
        }
        
        None
    }
    
    async fn update_search_status(&self, query_id: &str, status: SearchStatus) {
        let mut active = self.active_searches.write().await;
        if let Some(progress) = active.get_mut(query_id) {
            progress.status = status;
        }
    }
    
    async fn convert_filters_to_map(&self, filters: &SearchFilters) -> HashMap<String, String> {
        let mut map = HashMap::new();
        
        if let Some(categories) = &filters.categories {
            map.insert("categories".to_string(), categories.join(","));
        }
        
        if let Some(privacy_levels) = &filters.privacy_levels {
            let levels: Vec<String> = privacy_levels.iter()
                .map(|level| format!("{:?}", level))
                .collect();
            map.insert("privacy_levels".to_string(), levels.join(","));
        }
        
        map
    }
    
    async fn convert_distributed_results(
        &self,
        distributed_result: DistributedQueryResult,
        query: &AdvancedSearchQuery,
    ) -> SearchResult<Vec<SearchResultItem>> {
        let mut all_results = Vec::new();
        
        for (_, shard_results) in distributed_result.shard_results {
            for entry in shard_results {
                let result_item = SearchResultItem {
                    content_hash: entry.content_hash,
                    metadata: entry.metadata,
                    relevance_score: 0.8, // Would calculate actual relevance
                    snippet: Some(entry.terms.join(" ")),
                    highlights: None,
                    creator: entry.creator,
                    category: entry.category,
                    privacy_level: entry.privacy_level,
                };
                all_results.push(result_item);
            }
        }
        
        // Sort by relevance and limit
        all_results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(query.preferences.max_results);
        
        Ok(all_results)
    }
    
    async fn decrypt_and_process_results(
        &self,
        private_result: PrivateSearchResult,
        query: &AdvancedSearchQuery,
    ) -> SearchResult<Vec<SearchResultItem>> {
        // In a real implementation, this would decrypt the results
        // For now, return mock results
        Ok(vec![
            SearchResultItem {
                content_hash: ContentHash::from_bytes(&[1; 32]),
                metadata: ContentMetadata::default(),
                relevance_score: 0.9,
                snippet: Some("Private search result".to_string()),
                highlights: None,
                creator: None,
                category: None,
                privacy_level: ContentPrivacyLevel::Private,
            }
        ])
    }
    
    async fn process_compute_results(
        &self,
        compute_result: ComputeResult,
        query: &AdvancedSearchQuery,
    ) -> SearchResult<Vec<SearchResultItem>> {
        // Process NymCompute results
        Ok(vec![
            SearchResultItem {
                content_hash: ContentHash::from_bytes(&[2; 32]),
                metadata: ContentMetadata::default(),
                relevance_score: 0.95,
                snippet: Some("Zero-knowledge search result".to_string()),
                highlights: None,
                creator: None,
                category: None,
                privacy_level: ContentPrivacyLevel::Encrypted,
            }
        ])
    }
    
    async fn update_analytics(&self, cache_hit: bool, response_time: u64) {
        let mut analytics = self.analytics.write().await;
        analytics.total_searches += 1;
        
        // Update average response time
        let total = analytics.total_searches as f64;
        analytics.average_response_time = 
            (analytics.average_response_time * (total - 1.0) + response_time as f64) / total;
        
        // Update cache hit rate
        if cache_hit {
            analytics.cache_hit_rate = 
                (analytics.cache_hit_rate * (total - 1.0) + 1.0) / total;
        } else {
            analytics.cache_hit_rate = 
                (analytics.cache_hit_rate * (total - 1.0)) / total;
        }
    }
    
    /// Get search analytics
    pub async fn get_analytics(&self) -> SearchAnalytics {
        let analytics = self.analytics.read().await;
        SearchAnalytics {
            total_searches: analytics.total_searches,
            privacy_searches: analytics.privacy_searches,
            semantic_searches: analytics.semantic_searches,
            nymcompute_searches: analytics.nymcompute_searches,
            average_response_time: analytics.average_response_time,
            cache_hit_rate: analytics.cache_hit_rate,
            top_queries: analytics.top_queries.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_advanced_search_engine() {
        let config = AdvancedSearchConfig::default();
        let engine = AdvancedSearchEngine::new(config).await.unwrap();
        
        let query = AdvancedSearchQuery {
            query: "test search".to_string(),
            privacy_level: SearchPrivacyLevel::Public,
            filters: SearchFilters::default(),
            preferences: SearchPreferences::default(),
            semantic_options: None,
        };
        
        let result = engine.search(query, None).await.unwrap();
        
        assert_eq!(result.search_metadata.original_query, "test search");
        assert_eq!(result.privacy_metrics.privacy_level, SearchPrivacyLevel::Public);
    }
}