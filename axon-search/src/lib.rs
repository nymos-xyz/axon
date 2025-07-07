//! # Axon Advanced Search
//! 
//! Privacy-preserving advanced search engine for the Axon social network.
//! Provides sophisticated search capabilities while maintaining user anonymity
//! and content privacy through zero-knowledge proofs and distributed indexing.

pub mod error;
pub mod privacy_search;
pub mod distributed_index;
pub mod advanced_search_engine;
pub mod query_processing;
pub mod result_ranking;
pub mod search_analytics;
pub mod vector_search;

pub use error::{SearchError, SearchResult};
pub use privacy_search::{
    PrivacySearchEngine, PrivacySearchConfig, AnonymousQuery, 
    PrivateSearchResult, SearchPrivacyLevel
};
pub use distributed_index::{
    DistributedSearchIndex, IndexConfig, IndexShard, 
    IndexReplication, IndexStatistics, ContentPrivacyLevel
};
pub use advanced_search_engine::{
    AdvancedSearchEngine, AdvancedSearchConfig, AdvancedSearchQuery,
    AdvancedSearchResult, SearchFilters, SearchPreferences
};
pub use query_processing::{
    QueryProcessor, QueryAnalysis, QueryRewrite, 
    QueryOptimization, ProcessedQuery
};
pub use result_ranking::{
    ResultRanker, RankingAlgorithm, RankingFeatures, 
    RankedResult, RelevanceScore
};
pub use search_analytics::{
    SearchAnalytics, SearchMetrics, QueryAnalytics, 
    PerformanceMetrics, PrivacyMetrics
};
pub use vector_search::{
    VectorSearchEngine, SemanticSearchConfig, ContentEmbedding,
    SimilarityResult, EmbeddingModel
};

/// Search protocol version
pub const SEARCH_PROTOCOL_VERSION: u32 = 1;

/// Maximum query length for security
pub const MAX_QUERY_LENGTH: usize = 500;

/// Maximum results per search
pub const MAX_SEARCH_RESULTS: usize = 1000;

/// Default search timeout (seconds)
pub const DEFAULT_SEARCH_TIMEOUT: u64 = 30;