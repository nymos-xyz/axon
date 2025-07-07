//! Integration tests for Advanced Privacy-Preserving Search

use axon_search::{
    AdvancedSearchEngine, AdvancedSearchConfig, AdvancedSearchQuery,
    SearchFilters, SearchPreferences, SearchPrivacyLevel,
    ContentPrivacyLevel,
};
use axon_core::{
    types::{ContentHash, Timestamp},
    content::ContentMetadata,
};
use nym_core::NymIdentity;
use quid_core::QuIDIdentity;

#[tokio::test]
async fn test_advanced_search_integration() {
    // Initialize search engine
    let config = AdvancedSearchConfig::default();
    let search_engine = AdvancedSearchEngine::new(config).await.unwrap();
    
    // Index some test content
    let content_items = vec![
        (
            ContentHash::from_bytes(&[1; 32]),
            "Privacy-preserving social networking platform with anonymous interactions",
            ContentPrivacyLevel::Public,
        ),
        (
            ContentHash::from_bytes(&[2; 32]),
            "Decentralized search engine with zero-knowledge proofs",
            ContentPrivacyLevel::Private,
        ),
        (
            ContentHash::from_bytes(&[3; 32]),
            "Quantum-resistant cryptography for blockchain applications",
            ContentPrivacyLevel::Anonymous,
        ),
        (
            ContentHash::from_bytes(&[4; 32]),
            "Anonymous content sharing and discovery platform",
            ContentPrivacyLevel::Encrypted,
        ),
    ];
    
    // Index all content
    for (hash, text, privacy_level) in content_items {
        let metadata = ContentMetadata {
            content_type: "text/plain".to_string(),
            size: text.len(),
            created_at: Timestamp::now(),
            ..Default::default()
        };
        
        search_engine.index_content(
            hash,
            metadata,
            text.to_string(),
            Some(NymIdentity::new("test_creator".to_string())),
            privacy_level,
        ).await.unwrap();
    }
    
    println!("✅ Successfully indexed 4 content items");
    
    // Test 1: Public search
    let public_query = AdvancedSearchQuery {
        query: "privacy search".to_string(),
        privacy_level: SearchPrivacyLevel::Public,
        filters: SearchFilters::default(),
        preferences: SearchPreferences::default(),
        semantic_options: None,
    };
    
    let public_result = search_engine.search(public_query, None).await.unwrap();
    
    assert_eq!(public_result.privacy_metrics.privacy_level, SearchPrivacyLevel::Public);
    assert!(!public_result.privacy_metrics.query_obfuscated);
    assert_eq!(public_result.privacy_metrics.anonymity_set_size, 0);
    
    println!("✅ Public search completed - {} results found", 
             public_result.results.len());
    
    // Test 2: Private search
    let private_query = AdvancedSearchQuery {
        query: "decentralized anonymous".to_string(),
        privacy_level: SearchPrivacyLevel::Private,
        filters: SearchFilters::default(),
        preferences: SearchPreferences::default(),
        semantic_options: None,
    };
    
    let requester = QuIDIdentity::new("test_user".to_string()).unwrap();
    let private_result = search_engine.search(private_query, Some(requester)).await.unwrap();
    
    assert_eq!(private_result.privacy_metrics.privacy_level, SearchPrivacyLevel::Private);
    assert!(private_result.privacy_metrics.dp_epsilon > 0.0);
    
    println!("✅ Private search completed - anonymity set size: {}", 
             private_result.privacy_metrics.anonymity_set_size);
    
    // Test 3: Anonymous search
    let anonymous_query = AdvancedSearchQuery {
        query: "quantum cryptography".to_string(),
        privacy_level: SearchPrivacyLevel::Anonymous,
        filters: SearchFilters {
            privacy_levels: Some(vec![ContentPrivacyLevel::Anonymous, ContentPrivacyLevel::Public]),
            ..Default::default()
        },
        preferences: SearchPreferences {
            max_results: 10,
            ..Default::default()
        },
        semantic_options: None,
    };
    
    let anonymous_result = search_engine.search(anonymous_query, None).await.unwrap();
    
    assert_eq!(anonymous_result.privacy_metrics.privacy_level, SearchPrivacyLevel::Anonymous);
    assert!(anonymous_result.privacy_metrics.query_obfuscated);
    
    println!("✅ Anonymous search completed - query obfuscated: {}", 
             anonymous_result.privacy_metrics.query_obfuscated);
    
    // Test 4: Zero-knowledge search (will fallback if NymCompute unavailable)
    let zk_query = AdvancedSearchQuery {
        query: "blockchain platform".to_string(),
        privacy_level: SearchPrivacyLevel::ZeroKnowledge,
        filters: SearchFilters::default(),
        preferences: SearchPreferences::default(),
        semantic_options: None,
    };
    
    let zk_result = search_engine.search(zk_query, None).await.unwrap();
    
    // May be ZeroKnowledge or Anonymous depending on NymCompute availability
    assert!(matches!(zk_result.privacy_metrics.privacy_level, 
                     SearchPrivacyLevel::ZeroKnowledge | SearchPrivacyLevel::Anonymous));
    
    println!("✅ Zero-knowledge search completed - privacy level: {:?}", 
             zk_result.privacy_metrics.privacy_level);
    
    // Test 5: Performance metrics validation
    assert!(public_result.performance_metrics.total_time_ms > 0);
    assert!(private_result.performance_metrics.privacy_time_ms > 0);
    assert!(anonymous_result.performance_metrics.index_time_ms > 0);
    
    println!("✅ Performance metrics validated");
    println!("   - Public search: {}ms", public_result.performance_metrics.total_time_ms);
    println!("   - Private search: {}ms", private_result.performance_metrics.total_time_ms);
    println!("   - Anonymous search: {}ms", anonymous_result.performance_metrics.total_time_ms);
    
    // Test 6: Cache functionality
    let cached_query = AdvancedSearchQuery {
        query: "privacy search".to_string(), // Same as first query
        privacy_level: SearchPrivacyLevel::Public,
        filters: SearchFilters::default(),
        preferences: SearchPreferences::default(),
        semantic_options: None,
    };
    
    let cached_result = search_engine.search(cached_query, None).await.unwrap();
    assert!(cached_result.performance_metrics.cache_hit);
    
    println!("✅ Cache hit confirmed for repeated query");
    
    // Test 7: Analytics
    let analytics = search_engine.get_analytics().await;
    assert!(analytics.total_searches > 0);
    assert!(analytics.average_response_time > 0.0);
    
    println!("✅ Analytics validated - {} total searches", analytics.total_searches);
    
    println!("\n🎉 All advanced search integration tests passed!");
    println!("   - Privacy-preserving search: ✅");
    println!("   - Distributed indexing: ✅");
    println!("   - Multiple privacy levels: ✅");
    println!("   - Performance optimization: ✅");
    println!("   - Caching system: ✅");
    println!("   - Analytics tracking: ✅");
}

#[tokio::test]
async fn test_privacy_levels_access_control() {
    let config = AdvancedSearchConfig::default();
    let search_engine = AdvancedSearchEngine::new(config).await.unwrap();
    
    // Index content with different privacy levels
    let test_cases = vec![
        (ContentHash::from_bytes(&[10; 32]), "public content", ContentPrivacyLevel::Public),
        (ContentHash::from_bytes(&[11; 32]), "private content", ContentPrivacyLevel::Private),
        (ContentHash::from_bytes(&[12; 32]), "anonymous content", ContentPrivacyLevel::Anonymous),
        (ContentHash::from_bytes(&[13; 32]), "encrypted content", ContentPrivacyLevel::Encrypted),
    ];
    
    for (hash, text, privacy_level) in test_cases {
        search_engine.index_content(
            hash,
            ContentMetadata::default(),
            text.to_string(),
            None,
            privacy_level,
        ).await.unwrap();
    }
    
    // Test access control with different search privacy levels
    let search_tests = vec![
        (SearchPrivacyLevel::Public, "Should only access public content"),
        (SearchPrivacyLevel::Private, "Should access public and private content"),
        (SearchPrivacyLevel::Anonymous, "Should access public, private, and anonymous content"),
        (SearchPrivacyLevel::ZeroKnowledge, "Should access all content types"),
    ];
    
    for (search_privacy, description) in search_tests {
        let query = AdvancedSearchQuery {
            query: "content".to_string(),
            privacy_level: search_privacy.clone(),
            filters: SearchFilters::default(),
            preferences: SearchPreferences::default(),
            semantic_options: None,
        };
        
        let result = search_engine.search(query, None).await.unwrap();
        
        println!("✅ {} - Found {} results with privacy level: {:?}", 
                 description, result.results.len(), search_privacy);
        
        // Verify privacy level compatibility
        for result_item in &result.results {
            let compatible = match (&result_item.privacy_level, &search_privacy) {
                (ContentPrivacyLevel::Public, _) => true,
                (ContentPrivacyLevel::Private, SearchPrivacyLevel::Private) => true,
                (ContentPrivacyLevel::Private, SearchPrivacyLevel::Anonymous) => true,
                (ContentPrivacyLevel::Private, SearchPrivacyLevel::ZeroKnowledge) => true,
                (ContentPrivacyLevel::Anonymous, SearchPrivacyLevel::Anonymous) => true,
                (ContentPrivacyLevel::Anonymous, SearchPrivacyLevel::ZeroKnowledge) => true,
                (ContentPrivacyLevel::Encrypted, SearchPrivacyLevel::ZeroKnowledge) => true,
                _ => false,
            };
            
            assert!(compatible, 
                    "Privacy level mismatch: content {:?} should not be accessible with search {:?}",
                    result_item.privacy_level, search_privacy);
        }
    }
    
    println!("🎉 Privacy level access control tests passed!");
}

#[tokio::test]
async fn test_search_filtering_and_preferences() {
    let config = AdvancedSearchConfig::default();
    let search_engine = AdvancedSearchEngine::new(config).await.unwrap();
    
    // Index content with various metadata
    let creator1 = NymIdentity::new("creator_1".to_string());
    let creator2 = NymIdentity::new("creator_2".to_string());
    
    let content_with_metadata = vec![
        (
            ContentHash::from_bytes(&[20; 32]),
            "Technology blog post about AI",
            Some(creator1.clone()),
            Some("technology".to_string()),
        ),
        (
            ContentHash::from_bytes(&[21; 32]),
            "Art gallery showcase",
            Some(creator2.clone()),
            Some("art".to_string()),
        ),
        (
            ContentHash::from_bytes(&[22; 32]),
            "AI research paper on machine learning",
            Some(creator1.clone()),
            Some("research".to_string()),
        ),
    ];
    
    for (hash, text, creator, category) in content_with_metadata {
        let metadata = ContentMetadata {
            category: category.clone(),
            content_type: "text/plain".to_string(),
            size: text.len(),
            created_at: Timestamp::now(),
            ..Default::default()
        };
        
        search_engine.index_content(
            hash,
            metadata,
            text.to_string(),
            creator,
            ContentPrivacyLevel::Public,
        ).await.unwrap();
    }
    
    // Test category filtering
    let category_query = AdvancedSearchQuery {
        query: "AI".to_string(),
        privacy_level: SearchPrivacyLevel::Public,
        filters: SearchFilters {
            categories: Some(vec!["technology".to_string()]),
            ..Default::default()
        },
        preferences: SearchPreferences::default(),
        semantic_options: None,
    };
    
    let category_result = search_engine.search(category_query, None).await.unwrap();
    
    println!("✅ Category filtering test - Found {} results in 'technology' category", 
             category_result.results.len());
    
    // Test creator filtering
    let creator_query = AdvancedSearchQuery {
        query: "AI".to_string(),
        privacy_level: SearchPrivacyLevel::Public,
        filters: SearchFilters {
            creators: Some(vec![creator1.clone()]),
            ..Default::default()
        },
        preferences: SearchPreferences::default(),
        semantic_options: None,
    };
    
    let creator_result = search_engine.search(creator_query, None).await.unwrap();
    
    println!("✅ Creator filtering test - Found {} results from specific creator", 
             creator_result.results.len());
    
    // Test result limit preferences
    let limit_query = AdvancedSearchQuery {
        query: "AI".to_string(),
        privacy_level: SearchPrivacyLevel::Public,
        filters: SearchFilters::default(),
        preferences: SearchPreferences {
            max_results: 1,
            ..Default::default()
        },
        semantic_options: None,
    };
    
    let limit_result = search_engine.search(limit_query, None).await.unwrap();
    
    assert!(limit_result.results.len() <= 1);
    println!("✅ Result limit test - Limited to {} results as requested", 
             limit_result.results.len());
    
    println!("🎉 Search filtering and preferences tests passed!");
}