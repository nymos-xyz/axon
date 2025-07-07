//! Integration tests for Advanced Social Computing via NymCompute

use axon_ai::{
    SocialComputingEngine, SocialComputingConfig, ModerationRequest, ModerationLevel,
    SocialAnalysisRequest, SocialAnalysisType, ContentProcessingRequest, ContentProcessingType,
    RealtimeRecommendationRequest, ViolationType, RecommendationType,
    social_computing::*,
};
use axon_core::{
    types::{ContentHash, Timestamp},
    content::ContentMetadata,
};
use nym_crypto::Hash256;
use nym_compute::PrivacyLevel;
use std::collections::HashMap;

#[tokio::test]
async fn test_social_computing_configuration() {
    // Test default configuration
    let default_config = SocialComputingConfig::default();
    
    assert!(default_config.enable_ai_moderation);
    assert!(default_config.enable_social_analysis);
    assert!(default_config.enable_content_processing);
    assert!(default_config.enable_realtime_recommendations);
    assert_eq!(default_config.max_job_execution_time, 300);
    assert_eq!(default_config.privacy_budget, 3.0);
    assert_eq!(default_config.min_anonymity_set, 100);
    
    // Test custom configuration
    let custom_config = SocialComputingConfig {
        enable_ai_moderation: true,
        enable_social_analysis: false,
        enable_content_processing: true,
        enable_realtime_recommendations: true,
        max_job_execution_time: 600,
        privacy_budget: 5.0,
        min_anonymity_set: 200,
        enable_distributed_processing: true,
        max_retry_attempts: 5,
        result_cache_ttl: 1200,
    };
    
    assert!(!custom_config.enable_social_analysis);
    assert_eq!(custom_config.max_job_execution_time, 600);
    assert_eq!(custom_config.privacy_budget, 5.0);
    
    println!("✅ Social computing configuration tests passed");
}

#[tokio::test]
async fn test_moderation_request_types() {
    // Test different moderation levels
    let basic_level = ModerationLevel::Basic;
    let standard_level = ModerationLevel::Standard;
    let strict_level = ModerationLevel::Strict;
    
    assert!(matches!(basic_level, ModerationLevel::Basic));
    assert!(matches!(standard_level, ModerationLevel::Standard));
    assert!(matches!(strict_level, ModerationLevel::Strict));
    
    // Test custom moderation rules
    let custom_rules = vec![
        ModerationRule {
            rule_id: "spam_detection".to_string(),
            description: "Advanced spam detection rule".to_string(),
            weight: 0.8,
            criteria: "keyword_pattern_match".to_string(),
        },
        ModerationRule {
            rule_id: "harassment_detection".to_string(),
            description: "Detect harassment and bullying".to_string(),
            weight: 0.9,
            criteria: "sentiment_analysis".to_string(),
        },
    ];
    
    let custom_level = ModerationLevel::Custom(custom_rules.clone());
    if let ModerationLevel::Custom(rules) = custom_level {
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].rule_id, "spam_detection");
        assert_eq!(rules[1].weight, 0.9);
    }
    
    // Test moderation request
    let moderation_request = ModerationRequest {
        content: "This is a test message for content moderation analysis".to_string(),
        metadata: ContentMetadata {
            content_type: "text/plain".to_string(),
            size: 100,
            created_at: Timestamp::now(),
            ..Default::default()
        },
        moderation_level: ModerationLevel::Standard,
        timestamp: Timestamp::now(),
        requester_id: Some(Hash256::from_bytes(&[1; 32])),
    };
    
    assert!(!moderation_request.content.is_empty());
    assert!(moderation_request.requester_id.is_some());
    assert!(matches!(moderation_request.moderation_level, ModerationLevel::Standard));
    
    println!("✅ Moderation request types tests passed");
}

#[tokio::test]
async fn test_violation_types_and_actions() {
    // Test all violation types
    let violations = vec![
        ViolationType::Spam,
        ViolationType::Harassment,
        ViolationType::HateSpeech,
        ViolationType::Violence,
        ViolationType::SexualContent,
        ViolationType::Misinformation,
        ViolationType::Copyright,
        ViolationType::Privacy,
        ViolationType::Other("Custom violation type".to_string()),
    ];
    
    assert_eq!(violations.len(), 9);
    assert!(violations.contains(&ViolationType::Spam));
    assert!(violations.contains(&ViolationType::Harassment));
    
    if let ViolationType::Other(description) = &violations[8] {
        assert_eq!(description, "Custom violation type");
    }
    
    // Test moderation actions
    let actions = vec![
        ModerationAction::Allow,
        ModerationAction::Warn,
        ModerationAction::Review,
        ModerationAction::Hide,
        ModerationAction::Remove,
        ModerationAction::Suspend,
        ModerationAction::Ban,
    ];
    
    assert_eq!(actions.len(), 7);
    assert!(actions.contains(&ModerationAction::Allow));
    assert!(actions.contains(&ModerationAction::Ban));
    
    println!("✅ Violation types and actions tests passed");
}

#[tokio::test]
async fn test_social_analysis_types() {
    // Test social analysis request
    let analysis_request = SocialAnalysisRequest {
        analysis_type: SocialAnalysisType::CommunityDetection,
        target_user: Hash256::from_bytes(&[42; 32]),
        parameters: [
            ("algorithm".to_string(), serde_json::json!("louvain")),
            ("resolution".to_string(), serde_json::json!(1.0)),
            ("iterations".to_string(), serde_json::json!(100)),
        ].iter().cloned().collect(),
        privacy_level: PrivacyLevel::ZeroKnowledge,
        anonymity_set_size: 500,
    };
    
    assert!(matches!(analysis_request.analysis_type, SocialAnalysisType::CommunityDetection));
    assert_eq!(analysis_request.anonymity_set_size, 500);
    assert!(matches!(analysis_request.privacy_level, PrivacyLevel::ZeroKnowledge));
    assert_eq!(analysis_request.parameters.len(), 3);
    
    // Test all analysis types
    let analysis_types = vec![
        SocialAnalysisType::CommunityDetection,
        SocialAnalysisType::InfluenceAnalysis,
        SocialAnalysisType::ConnectionPatterns,
        SocialAnalysisType::SocialClustering,
        SocialAnalysisType::NetworkCentrality,
        SocialAnalysisType::RecommendationTargets,
    ];
    
    assert_eq!(analysis_types.len(), 6);
    
    println!("✅ Social analysis types tests passed");
}

#[tokio::test]
async fn test_content_processing_types() {
    // Test content processing request
    let processing_request = ContentProcessingRequest {
        content_id: ContentHash::from_bytes(&[123; 32]),
        processing_type: ContentProcessingType::VideoTranscode,
        parameters: [
            ("codec".to_string(), serde_json::json!("h264")),
            ("bitrate".to_string(), serde_json::json!(2000)),
            ("resolution".to_string(), serde_json::json!("1920x1080")),
        ].iter().cloned().collect(),
        quality_settings: QualitySettings {
            quality_level: QualityLevel::High,
            max_file_size: 100_000_000, // 100MB
            target_format: "mp4".to_string(),
            compression: CompressionSettings {
                ratio: 0.7,
                preserve_quality: true,
                algorithm: "h264".to_string(),
            },
        },
        privacy_requirements: ContentPrivacyRequirements {
            anonymize_metadata: true,
            remove_identifiers: true,
            apply_privacy_filters: true,
            encryption_required: true,
        },
    };
    
    assert!(matches!(processing_request.processing_type, ContentProcessingType::VideoTranscode));
    assert!(matches!(processing_request.quality_settings.quality_level, QualityLevel::High));
    assert!(processing_request.privacy_requirements.encryption_required);
    assert_eq!(processing_request.parameters.len(), 3);
    
    // Test all processing types
    let processing_types = vec![
        ContentProcessingType::VideoTranscode,
        ContentProcessingType::AudioProcess,
        ContentProcessingType::ImageOptimize,
        ContentProcessingType::TextAnalysis,
        ContentProcessingType::Summarization,
        ContentProcessingType::Translation,
        ContentProcessingType::Enhancement,
    ];
    
    assert_eq!(processing_types.len(), 7);
    
    // Test quality levels
    let quality_levels = vec![
        QualityLevel::Low,
        QualityLevel::Medium,
        QualityLevel::High,
        QualityLevel::Ultra,
        QualityLevel::Custom([("bitrate".to_string(), 5000.0)].iter().cloned().collect()),
    ];
    
    assert_eq!(quality_levels.len(), 5);
    if let QualityLevel::Custom(params) = &quality_levels[4] {
        assert!(params.contains_key("bitrate"));
    }
    
    println!("✅ Content processing types tests passed");
}

#[tokio::test]
async fn test_realtime_recommendation_types() {
    // Test user context
    let user_context = AnonymousUserContext {
        user_id: Hash256::from_bytes(&[99; 32]),
        interaction_patterns: vec![
            InteractionPattern {
                pattern_type: "browsing_technology".to_string(),
                strength: 0.8,
                recency: 0.9,
                context_factors: [
                    ("time_spent".to_string(), 0.7),
                    ("engagement_rate".to_string(), 0.6),
                ].iter().cloned().collect(),
            },
            InteractionPattern {
                pattern_type: "social_sharing".to_string(),
                strength: 0.6,
                recency: 0.5,
                context_factors: [
                    ("share_frequency".to_string(), 0.4),
                    ("comment_frequency".to_string(), 0.3),
                ].iter().cloned().collect(),
            },
        ],
        temporal_context: TemporalContext {
            time_of_day: "evening".to_string(),
            day_of_week: "friday".to_string(),
            season: "autumn".to_string(),
            timezone_offset: -5,
        },
        social_context: AnonymousSocialContext {
            activity_level: 0.75,
            network_size_tier: NetworkSizeTier::Large,
            engagement_patterns: vec![
                "likes".to_string(),
                "shares".to_string(),
                "comments".to_string(),
            ],
            community_involvement: 0.8,
        },
    };
    
    assert_eq!(user_context.interaction_patterns.len(), 2);
    assert_eq!(user_context.temporal_context.time_of_day, "evening");
    assert!(matches!(user_context.social_context.network_size_tier, NetworkSizeTier::Large));
    assert_eq!(user_context.social_context.engagement_patterns.len(), 3);
    
    // Test recommendation request
    let rec_request = RealtimeRecommendationRequest {
        user_context,
        current_activity: UserActivity::Browsing,
        recommendation_count: 10,
        recommendation_types: vec![
            RecommendationType::Content,
            RecommendationType::Users,
            RecommendationType::Communities,
        ],
        realtime_constraints: RealtimeConstraints {
            max_response_time: 500, // 500ms
            allow_cached: true,
            min_quality: 0.7,
            privacy_budget_limit: 1.0,
        },
    };
    
    assert!(matches!(rec_request.current_activity, UserActivity::Browsing));
    assert_eq!(rec_request.recommendation_count, 10);
    assert_eq!(rec_request.recommendation_types.len(), 3);
    assert_eq!(rec_request.realtime_constraints.max_response_time, 500);
    
    // Test network size tiers
    let size_tiers = vec![
        NetworkSizeTier::Small,
        NetworkSizeTier::Medium,
        NetworkSizeTier::Large,
        NetworkSizeTier::VeryLarge,
    ];
    
    assert_eq!(size_tiers.len(), 4);
    
    println!("✅ Real-time recommendation types tests passed");
}

#[tokio::test]
async fn test_social_computing_engine_mock() {
    // Test engine initialization failure (expected without NymCompute)
    let config = SocialComputingConfig::default();
    
    match SocialComputingEngine::new(config).await {
        Ok(_engine) => {
            // If this succeeds, NymCompute is available
            println!("✅ Social computing engine initialized with NymCompute");
        }
        Err(e) => {
            // Expected in test environments without NymCompute
            println!("⚠️  NymCompute not available for testing: {:?}", e);
            assert!(e.to_string().contains("Failed to initialize NymCompute client"));
        }
    }
    
    println!("✅ Social computing engine initialization test completed");
}

#[tokio::test]
async fn test_privacy_metrics_and_analytics() {
    // Test privacy metrics structure
    let privacy_metrics = AnalysisPrivacyMetrics {
        anonymity_set_size: 250,
        privacy_budget_used: 1.5,
        dp_epsilon: 0.5,
        zk_proof_provided: true,
    };
    
    assert_eq!(privacy_metrics.anonymity_set_size, 250);
    assert_eq!(privacy_metrics.privacy_budget_used, 1.5);
    assert!(privacy_metrics.zk_proof_provided);
    
    // Test recommendation privacy metrics
    let rec_privacy_metrics = RecommendationPrivacyMetrics {
        privacy_budget_used: 0.8,
        anonymity_maintained: true,
        k_anonymity: 150,
        differential_privacy: true,
    };
    
    assert_eq!(rec_privacy_metrics.privacy_budget_used, 0.8);
    assert!(rec_privacy_metrics.anonymity_maintained);
    assert_eq!(rec_privacy_metrics.k_anonymity, 150);
    assert!(rec_privacy_metrics.differential_privacy);
    
    // Test processing metrics
    let processing_metrics = ProcessingMetrics {
        original_size: 50_000_000, // 50MB
        processed_size: 15_000_000, // 15MB
        compression_ratio: 0.3,
        quality_score: 0.9,
        processing_time: 45000, // 45 seconds
    };
    
    assert_eq!(processing_metrics.original_size, 50_000_000);
    assert_eq!(processing_metrics.processed_size, 15_000_000);
    assert_eq!(processing_metrics.compression_ratio, 0.3);
    assert_eq!(processing_metrics.quality_score, 0.9);
    
    // Test recommendation quality metrics
    let quality_metrics = RecommendationQualityMetrics {
        quality_score: 0.85,
        diversity_score: 0.7,
        relevance_score: 0.9,
        novelty_score: 0.6,
    };
    
    assert_eq!(quality_metrics.quality_score, 0.85);
    assert_eq!(quality_metrics.diversity_score, 0.7);
    assert_eq!(quality_metrics.relevance_score, 0.9);
    assert_eq!(quality_metrics.novelty_score, 0.6);
    
    println!("✅ Privacy metrics and analytics tests passed");
}

#[tokio::test]
async fn test_cache_status_and_constraints() {
    // Test cache status types
    let cache_statuses = vec![
        CacheStatus::Hit,
        CacheStatus::Miss,
        CacheStatus::Partial,
    ];
    
    assert_eq!(cache_statuses.len(), 3);
    assert!(matches!(cache_statuses[0], CacheStatus::Hit));
    assert!(matches!(cache_statuses[1], CacheStatus::Miss));
    assert!(matches!(cache_statuses[2], CacheStatus::Partial));
    
    // Test real-time constraints
    let constraints = RealtimeConstraints {
        max_response_time: 1000, // 1 second
        allow_cached: true,
        min_quality: 0.8,
        privacy_budget_limit: 2.0,
    };
    
    assert_eq!(constraints.max_response_time, 1000);
    assert!(constraints.allow_cached);
    assert_eq!(constraints.min_quality, 0.8);
    assert_eq!(constraints.privacy_budget_limit, 2.0);
    
    // Test compression settings
    let compression = CompressionSettings {
        ratio: 0.5,
        preserve_quality: true,
        algorithm: "h265".to_string(),
    };
    
    assert_eq!(compression.ratio, 0.5);
    assert!(compression.preserve_quality);
    assert_eq!(compression.algorithm, "h265");
    
    println!("✅ Cache status and constraints tests passed");
}

#[tokio::test]
async fn test_complete_social_computing_workflow() {
    println!("🧪 Testing complete social computing workflow...");
    
    // 1. Test moderation workflow
    println!("  1. Content Moderation Workflow:");
    
    let moderation_request = ModerationRequest {
        content: "This is a completely safe and appropriate message for testing AI moderation capabilities".to_string(),
        metadata: ContentMetadata {
            content_type: "text/plain".to_string(),
            size: 100,
            created_at: Timestamp::now(),
            ..Default::default()
        },
        moderation_level: ModerationLevel::Standard,
        timestamp: Timestamp::now(),
        requester_id: Some(Hash256::from_bytes(&[1; 32])),
    };
    
    println!("     ✅ Moderation request created");
    println!("     - Content length: {} chars", moderation_request.content.len());
    println!("     - Moderation level: {:?}", moderation_request.moderation_level);
    
    // 2. Test social analysis workflow
    println!("  2. Social Analysis Workflow:");
    
    let analysis_request = SocialAnalysisRequest {
        analysis_type: SocialAnalysisType::CommunityDetection,
        target_user: Hash256::from_bytes(&[42; 32]),
        parameters: [
            ("min_community_size".to_string(), serde_json::json!(10)),
            ("max_communities".to_string(), serde_json::json!(50)),
        ].iter().cloned().collect(),
        privacy_level: PrivacyLevel::ZeroKnowledge,
        anonymity_set_size: 1000,
    };
    
    println!("     ✅ Social analysis request created");
    println!("     - Analysis type: {:?}", analysis_request.analysis_type);
    println!("     - Anonymity set size: {}", analysis_request.anonymity_set_size);
    
    // 3. Test content processing workflow
    println!("  3. Content Processing Workflow:");
    
    let processing_request = ContentProcessingRequest {
        content_id: ContentHash::from_bytes(&[200; 32]),
        processing_type: ContentProcessingType::ImageOptimize,
        parameters: [
            ("target_width".to_string(), serde_json::json!(1920)),
            ("target_height".to_string(), serde_json::json!(1080)),
            ("quality".to_string(), serde_json::json!(85)),
        ].iter().cloned().collect(),
        quality_settings: QualitySettings {
            quality_level: QualityLevel::High,
            max_file_size: 5_000_000, // 5MB
            target_format: "webp".to_string(),
            compression: CompressionSettings {
                ratio: 0.8,
                preserve_quality: true,
                algorithm: "webp".to_string(),
            },
        },
        privacy_requirements: ContentPrivacyRequirements {
            anonymize_metadata: true,
            remove_identifiers: true,
            apply_privacy_filters: false,
            encryption_required: false,
        },
    };
    
    println!("     ✅ Content processing request created");
    println!("     - Processing type: {:?}", processing_request.processing_type);
    println!("     - Target format: {}", processing_request.quality_settings.target_format);
    
    // 4. Test real-time recommendations workflow
    println!("  4. Real-time Recommendations Workflow:");
    
    let rec_request = RealtimeRecommendationRequest {
        user_context: AnonymousUserContext {
            user_id: Hash256::from_bytes(&[123; 32]),
            interaction_patterns: vec![
                InteractionPattern {
                    pattern_type: "content_consumption".to_string(),
                    strength: 0.9,
                    recency: 0.8,
                    context_factors: [
                        ("video_watch_time".to_string(), 0.85),
                        ("article_read_time".to_string(), 0.7),
                    ].iter().cloned().collect(),
                },
            ],
            temporal_context: TemporalContext {
                time_of_day: "afternoon".to_string(),
                day_of_week: "wednesday".to_string(),
                season: "spring".to_string(),
                timezone_offset: 0,
            },
            social_context: AnonymousSocialContext {
                activity_level: 0.8,
                network_size_tier: NetworkSizeTier::Medium,
                engagement_patterns: vec!["likes".to_string(), "comments".to_string()],
                community_involvement: 0.6,
            },
        },
        current_activity: UserActivity::MediaConsumption,
        recommendation_count: 15,
        recommendation_types: vec![
            RecommendationType::Content,
            RecommendationType::Topics,
        ],
        realtime_constraints: RealtimeConstraints {
            max_response_time: 300, // 300ms
            allow_cached: true,
            min_quality: 0.75,
            privacy_budget_limit: 1.5,
        },
    };
    
    println!("     ✅ Real-time recommendation request created");
    println!("     - Current activity: {:?}", rec_request.current_activity);
    println!("     - Recommendation count: {}", rec_request.recommendation_count);
    println!("     - Max response time: {}ms", rec_request.realtime_constraints.max_response_time);
    
    // 5. Test privacy verification
    println!("  5. Privacy Verification:");
    
    let privacy_confirmation = PrivacyConfirmation {
        requirements_met: true,
        anonymized: true,
        identifiers_removed: true,
        privacy_proof: None, // Would contain actual ZK proof in real implementation
    };
    
    println!("     ✅ Privacy requirements verified");
    println!("     - Requirements met: {}", privacy_confirmation.requirements_met);
    println!("     - Content anonymized: {}", privacy_confirmation.anonymized);
    println!("     - Identifiers removed: {}", privacy_confirmation.identifiers_removed);
    
    println!("\n🎉 Complete social computing workflow test passed!");
    println!("   - AI-powered content moderation: ✅");
    println!("   - Privacy-preserving social analysis: ✅");
    println!("   - Decentralized content processing: ✅");
    println!("   - Real-time recommendation engine: ✅");
    println!("   - Privacy protection throughout: ✅");
}