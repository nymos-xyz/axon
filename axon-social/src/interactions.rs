//! Content Interaction System - Privacy-preserving social interactions
//!
//! This module implements privacy-first content interactions:
//! - Anonymous likes/dislikes with zero-knowledge proofs
//! - Private replies and comments with optional revelation
//! - Share tracking without user profiling
//! - Engagement metrics without compromising privacy
//! - Anti-spam and rate limiting with privacy preservation

use std::collections::{HashMap, HashSet};
use bincode;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use axon_core::{
    identity::QuIDIdentity as Identity,
    ContentHash as ContentId,
    content::Post as Content,
    ContentType
};
use axon_identity::auth_service::AuthenticationService as AuthService;
use crate::{SocialError, SocialResult, PrivacyLevel, AnonymousProof, ProofType};
use crate::social_graph::UserId;

/// Content interaction manager
#[derive(Debug)]
pub struct InteractionManager {
    /// All interactions by content
    interactions: HashMap<ContentId, Vec<Interaction>>,
    /// Anonymous interaction proofs
    anonymous_proofs: HashMap<String, InteractionProof>,
    /// User interaction history (anonymized)
    user_history: HashMap<String, UserInteractionHistory>,
    /// Rate limiting data
    rate_limits: HashMap<String, RateLimitData>,
    /// Spam detection
    spam_detector: SpamDetector,
    /// Interaction settings
    settings: InteractionSettings,
}

/// A social interaction with content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    /// Interaction identifier
    pub id: String,
    /// Content being interacted with
    pub content_id: ContentId,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// User who performed interaction (may be anonymous)
    pub user_id: Option<String>,
    /// Privacy level of interaction
    pub privacy_level: PrivacyLevel,
    /// Interaction timestamp
    pub created_at: DateTime<Utc>,
    /// Interaction metadata (encrypted if private)
    pub metadata: Option<Vec<u8>>,
    /// Zero-knowledge proof for anonymous interactions
    pub proof: Option<Vec<u8>>,
}

/// Types of content interactions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionType {
    /// Like/upvote content
    Like,
    /// Dislike/downvote content
    Dislike,
    /// Vote on content or proposals
    Vote {
        /// Vote choice (true for yes, false for no)
        choice: bool,
        /// Vote weight (based on reputation/stake)
        weight: f64,
        /// Optional reason for vote
        reason: Option<String>,
    },
    /// Reply to content
    Reply {
        /// Reply content
        content: Content,
        /// Parent interaction if replying to reply
        parent_id: Option<String>,
    },
    /// Comment on content
    Comment {
        /// Comment content
        content: Content,
    },
    /// Share/repost content
    Share {
        /// Optional comment when sharing
        comment: Option<String>,
        /// Share visibility
        visibility: ShareVisibility,
    },
    /// Bookmark content privately
    Bookmark,
    /// Report content
    Report {
        /// Reason for report
        reason: ReportReason,
        /// Additional details
        details: Option<String>,
    },
}

/// Share visibility options
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShareVisibility {
    /// Share publicly
    Public,
    /// Share to followers only
    Followers,
    /// Share to specific users
    Direct(Vec<String>),
    /// Anonymous share
    Anonymous,
}

/// Content report reasons
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportReason {
    Spam,
    Harassment,
    Violence,
    Misinformation,
    Copyright,
    Privacy,
    Other,
}

/// Zero-knowledge proof for interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionProof {
    /// Proof that user is authorized to interact
    pub authorization_proof: Vec<u8>,
    /// Proof that interaction is unique (no double-voting)
    pub uniqueness_proof: Vec<u8>,
    /// Proof that user meets requirements (age, reputation, etc.)
    pub eligibility_proof: Vec<u8>,
    /// Timestamp proof to prevent replay attacks
    pub timestamp_proof: Vec<u8>,
    /// Proof generation time
    pub created_at: DateTime<Utc>,
    /// Proof expiry
    pub expires_at: DateTime<Utc>,
}

/// Reply to content or other replies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reply {
    /// Reply ID
    pub id: String,
    /// Content being replied to
    pub content_id: ContentId,
    /// Parent reply if this is a nested reply
    pub parent_reply_id: Option<String>,
    /// Reply content
    pub content: Content,
    /// Author (may be anonymous)
    pub author: Option<String>,
    /// Privacy level
    pub privacy_level: PrivacyLevel,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Reply thread depth
    pub depth: u32,
    /// Interaction metrics
    pub metrics: ReplyMetrics,
}

/// Like/dislike interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Like {
    /// Like ID
    pub id: String,
    /// Content being liked
    pub content_id: ContentId,
    /// True for like, false for dislike
    pub is_positive: bool,
    /// User who liked (anonymous if privacy_level != Public)
    pub user_id: Option<String>,
    /// Privacy level
    pub privacy_level: PrivacyLevel,
    /// Timestamp
    pub created_at: DateTime<Utc>,
    /// Weight of the like (based on user reputation)
    pub weight: f64,
}

/// Vote interaction on content or proposals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Vote ID
    pub id: String,
    /// Content being voted on
    pub content_id: ContentId,
    /// Vote choice (true for yes, false for no)
    pub choice: bool,
    /// Vote weight (based on reputation/stake)
    pub weight: f64,
    /// Optional reason for vote
    pub reason: Option<String>,
    /// User who voted (anonymous if privacy_level != Public)
    pub user_id: Option<String>,
    /// Privacy level
    pub privacy_level: PrivacyLevel,
    /// Timestamp
    pub created_at: DateTime<Utc>,
    /// Vote type (like, proposal, poll)
    pub vote_type: VoteType,
}

/// Types of votes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoteType {
    /// Like-based voting ("every like is like a vote")
    Like,
    /// Formal proposal voting
    Proposal,
    /// Community poll
    Poll,
    /// Content quality rating
    Quality,
    /// Moderation decision
    Moderation,
}

/// Share/repost of content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Share {
    /// Share ID
    pub id: String,
    /// Original content
    pub content_id: ContentId,
    /// User sharing
    pub user_id: Option<String>,
    /// Share comment
    pub comment: Option<String>,
    /// Visibility settings
    pub visibility: ShareVisibility,
    /// Privacy level
    pub privacy_level: PrivacyLevel,
    /// Share timestamp
    pub created_at: DateTime<Utc>,
    /// Share metrics
    pub metrics: ShareMetrics,
}

/// Metrics for replies
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplyMetrics {
    /// Number of likes on this reply
    pub likes: u32,
    /// Number of dislikes on this reply
    pub dislikes: u32,
    /// Number of nested replies
    pub reply_count: u32,
    /// Engagement score
    pub engagement_score: f64,
}

/// Metrics for shares
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShareMetrics {
    /// Views of the share
    pub views: u32,
    /// Interactions with the share
    pub interactions: u32,
    /// Secondary shares (shares of this share)
    pub reshares: u32,
}

/// User interaction history (anonymized)
#[derive(Debug, Clone)]
struct UserInteractionHistory {
    /// Anonymized user identifier
    user_hash: String,
    /// Recent interaction timestamps
    recent_interactions: Vec<DateTime<Utc>>,
    /// Interaction type counters
    interaction_counts: HashMap<InteractionType, u32>,
    /// Last interaction time
    last_interaction: DateTime<Utc>,
    /// Reputation score
    reputation: f64,
}

/// Rate limiting data
#[derive(Debug, Clone)]
struct RateLimitData {
    /// Interactions in current time window
    current_window_count: u32,
    /// Window start time
    window_start: DateTime<Utc>,
    /// Violations count
    violations: u32,
}

/// Spam detection system
#[derive(Debug)]
struct SpamDetector {
    /// Known spam patterns
    spam_patterns: Vec<SpamPattern>,
    /// Suspicious activity indicators
    suspicious_indicators: HashMap<String, f64>,
}

/// Spam detection pattern
#[derive(Debug, Clone)]
struct SpamPattern {
    /// Pattern name
    name: String,
    /// Detection function
    detector: fn(&Interaction, &UserInteractionHistory) -> f64,
    /// Threshold for spam classification
    threshold: f64,
}

/// Interaction system settings
#[derive(Debug, Clone)]
pub struct InteractionSettings {
    /// Maximum reply depth
    pub max_reply_depth: u32,
    /// Rate limit: interactions per hour
    pub max_interactions_per_hour: u32,
    /// Require proofs for anonymous interactions
    pub require_anonymous_proofs: bool,
    /// Enable spam detection
    pub enable_spam_detection: bool,
    /// Minimum reputation for interactions
    pub min_reputation: f64,
    /// Anonymous interaction weight multiplier
    pub anonymous_weight_multiplier: f64,
}

impl Default for InteractionSettings {
    fn default() -> Self {
        Self {
            max_reply_depth: 10,
            max_interactions_per_hour: 100,
            require_anonymous_proofs: true,
            enable_spam_detection: true,
            min_reputation: 0.0,
            anonymous_weight_multiplier: 0.8,
        }
    }
}

impl InteractionManager {
    /// Create a new interaction manager
    pub fn new() -> Self {
        Self::with_settings(InteractionSettings::default())
    }

    /// Create interaction manager with custom settings
    pub fn with_settings(settings: InteractionSettings) -> Self {
        Self {
            interactions: HashMap::new(),
            anonymous_proofs: HashMap::new(),
            user_history: HashMap::new(),
            rate_limits: HashMap::new(),
            spam_detector: SpamDetector::new(),
            settings,
        }
    }

    /// Like or dislike content
    pub async fn like_content(
        &mut self,
        identity: &Identity,
        content_id: ContentId,
        is_positive: bool,
        privacy_level: PrivacyLevel,
        auth_service: &AuthService,
    ) -> SocialResult<Interaction> {
        // Check rate limits
        self.check_rate_limits(&identity.get_id())?;

        // Check if user already liked this content
        if self.has_user_interacted(&identity.get_id(), &content_id, &InteractionType::Like)? {
            return Err(SocialError::InvalidOperation("User already liked this content".to_string()));
        }

        // Generate interaction ID
        let interaction_id = self.generate_interaction_id(&identity.get_id(), &content_id, "like");

        // Get user reputation for weighting
        let reputation = self.get_user_reputation(&identity.get_id());
        let weight = if privacy_level == PrivacyLevel::Public {
            reputation
        } else {
            reputation * self.settings.anonymous_weight_multiplier
        };

        // Create like interaction
        let like = Like {
            id: interaction_id.clone(),
            content_id: content_id.clone(),
            is_positive,
            user_id: match privacy_level {
                PrivacyLevel::Public => Some(identity.get_id()),
                _ => None,
            },
            privacy_level: privacy_level.clone(),
            created_at: Utc::now(),
            weight,
        };

        // Generate proof if anonymous
        let proof = if privacy_level != PrivacyLevel::Public {
            Some(self.generate_interaction_proof(
                identity,
                &content_id,
                &InteractionType::Like,
                auth_service,
            ).await?)
        } else {
            None
        };

        // Create interaction record
        let interaction = Interaction {
            id: interaction_id,
            content_id: content_id.clone(),
            interaction_type: InteractionType::Like,
            user_id: like.user_id.clone(),
            privacy_level,
            created_at: like.created_at,
            metadata: Some(bincode::serialize(&like)?),
            proof,
        };

        // Store interaction
        self.store_interaction(interaction.clone())?;

        // Update user history
        self.update_user_history(&identity.get_id(), &interaction);

        Ok(interaction)
    }

    /// Vote on content (combines like functionality)
    pub async fn vote_on_content(
        &mut self,
        identity: &Identity,
        content_id: ContentId,
        choice: bool,
        vote_type: VoteType,
        reason: Option<String>,
        privacy_level: PrivacyLevel,
        auth_service: &AuthService,
    ) -> SocialResult<Interaction> {
        // Check rate limits
        self.check_rate_limits(&identity.get_id())?;

        // Check if user already voted on this content
        if self.has_user_interacted(&identity.get_id(), &content_id, &InteractionType::Vote { 
            choice, 
            weight: 0.0, 
            reason: reason.clone() 
        })? {
            return Err(SocialError::InvalidOperation("User already voted on this content".to_string()));
        }

        // Generate vote ID
        let vote_id = self.generate_interaction_id(&identity.get_id(), &content_id, "vote");

        // Get user reputation for vote weight
        let reputation = self.get_user_reputation(&identity.get_id());
        let weight = if privacy_level == PrivacyLevel::Public {
            self.calculate_vote_weight(reputation, &vote_type)
        } else {
            self.calculate_vote_weight(reputation, &vote_type) * self.settings.anonymous_weight_multiplier
        };

        // Create vote
        let vote = Vote {
            id: vote_id.clone(),
            content_id: content_id.clone(),
            choice,
            weight,
            reason: reason.clone(),
            user_id: match privacy_level {
                PrivacyLevel::Public => Some(identity.get_id()),
                _ => None,
            },
            privacy_level: privacy_level.clone(),
            created_at: Utc::now(),
            vote_type: vote_type.clone(),
        };

        // Generate proof if anonymous
        let proof = if privacy_level != PrivacyLevel::Public {
            Some(self.generate_interaction_proof(
                identity,
                &content_id,
                &InteractionType::Vote { 
                    choice, 
                    weight,
                    reason: reason.clone() 
                },
                auth_service,
            ).await?)
        } else {
            None
        };

        // Create interaction record
        let interaction = Interaction {
            id: vote_id,
            content_id: content_id.clone(),
            interaction_type: InteractionType::Vote { 
                choice, 
                weight,
                reason: reason.clone() 
            },
            user_id: vote.user_id.clone(),
            privacy_level,
            created_at: vote.created_at,
            metadata: Some(bincode::serialize(&vote)?),
            proof,
        };

        // Store interaction
        self.store_interaction(interaction.clone())?;

        // Update user history
        self.update_user_history(&identity.get_id(), &interaction);

        Ok(interaction)
    }

    /// Reply to content or another reply
    pub async fn reply_to_content(
        &mut self,
        identity: &Identity,
        content_id: ContentId,
        parent_reply_id: Option<String>,
        reply_content: Content,
        privacy_level: PrivacyLevel,
        auth_service: &AuthService,
    ) -> SocialResult<Interaction> {
        // Check rate limits
        self.check_rate_limits(&identity.get_id())?;

        // Check reply depth
        let depth = self.calculate_reply_depth(&content_id, &parent_reply_id)?;
        if depth > self.settings.max_reply_depth {
            return Err(SocialError::InvalidOperation(
                format!("Reply depth exceeds maximum of {}", self.settings.max_reply_depth)
            ));
        }

        // Generate reply ID
        let reply_id = self.generate_interaction_id(&identity.get_id(), &content_id, "reply");

        // Create reply
        let reply = Reply {
            id: reply_id.clone(),
            content_id: content_id.clone(),
            parent_reply_id,
            content: reply_content,
            author: match privacy_level {
                PrivacyLevel::Public => Some(identity.get_id()),
                _ => None,
            },
            privacy_level: privacy_level.clone(),
            created_at: Utc::now(),
            depth,
            metrics: ReplyMetrics::default(),
        };

        // Generate proof if anonymous
        let proof = if privacy_level != PrivacyLevel::Public {
            Some(self.generate_interaction_proof(
                identity,
                &content_id,
                &InteractionType::Reply { content: reply.content.clone(), parent_id: reply.parent_reply_id.clone() },
                auth_service,
            ).await?)
        } else {
            None
        };

        // Create interaction record
        let interaction = Interaction {
            id: reply_id,
            content_id: content_id.clone(),
            interaction_type: InteractionType::Reply {
                content: reply.content.clone(),
                parent_id: reply.parent_reply_id.clone(),
            },
            user_id: reply.author.clone(),
            privacy_level,
            created_at: reply.created_at,
            metadata: Some(bincode::serialize(&reply)?),
            proof,
        };

        // Store interaction
        self.store_interaction(interaction.clone())?;

        // Update user history
        self.update_user_history(&identity.get_id(), &interaction);

        Ok(interaction)
    }

    /// Share content
    pub async fn share_content(
        &mut self,
        identity: &Identity,
        content_id: ContentId,
        comment: Option<String>,
        visibility: ShareVisibility,
        privacy_level: PrivacyLevel,
        auth_service: &AuthService,
    ) -> SocialResult<Interaction> {
        // Check rate limits
        self.check_rate_limits(&identity.get_id())?;

        // Generate share ID
        let share_id = self.generate_interaction_id(&identity.get_id(), &content_id, "share");

        // Create share
        let share = Share {
            id: share_id.clone(),
            content_id: content_id.clone(),
            user_id: match privacy_level {
                PrivacyLevel::Public => Some(identity.get_id()),
                _ => None,
            },
            comment,
            visibility,
            privacy_level: privacy_level.clone(),
            created_at: Utc::now(),
            metrics: ShareMetrics::default(),
        };

        // Generate proof if anonymous
        let proof = if privacy_level != PrivacyLevel::Public {
            Some(self.generate_interaction_proof(
                identity,
                &content_id,
                &InteractionType::Share {
                    comment: share.comment.clone(),
                    visibility: share.visibility.clone(),
                },
                auth_service,
            ).await?)
        } else {
            None
        };

        // Create interaction record
        let interaction = Interaction {
            id: share_id,
            content_id: content_id.clone(),
            interaction_type: InteractionType::Share {
                comment: share.comment.clone(),
                visibility: share.visibility.clone(),
            },
            user_id: share.user_id.clone(),
            privacy_level,
            created_at: share.created_at,
            metadata: Some(bincode::serialize(&share)?),
            proof,
        };

        // Store interaction
        self.store_interaction(interaction.clone())?;

        // Update user history
        self.update_user_history(&identity.get_id(), &interaction);

        Ok(interaction)
    }

    /// Report content
    pub async fn report_content(
        &mut self,
        identity: &Identity,
        content_id: ContentId,
        reason: ReportReason,
        details: Option<String>,
        auth_service: &AuthService,
    ) -> SocialResult<Interaction> {
        // Reports are always anonymous for user protection
        let privacy_level = PrivacyLevel::Anonymous;

        // Check rate limits (separate limit for reports)
        self.check_report_rate_limits(&identity.get_id())?;

        // Generate report ID
        let report_id = self.generate_interaction_id(&identity.get_id(), &content_id, "report");

        // Generate proof for anonymous report
        let proof = self.generate_interaction_proof(
            identity,
            &content_id,
            &InteractionType::Report {
                reason: reason.clone(),
                details: details.clone(),
            },
            auth_service,
        ).await?;

        // Create interaction record
        let interaction = Interaction {
            id: report_id,
            content_id: content_id.clone(),
            interaction_type: InteractionType::Report { reason, details },
            user_id: None, // Always anonymous
            privacy_level,
            created_at: Utc::now(),
            metadata: None, // No metadata for reports
            proof: Some(proof),
        };

        // Store interaction
        self.store_interaction(interaction.clone())?;

        // Update user history (anonymized)
        self.update_user_history(&identity.get_id(), &interaction);

        Ok(interaction)
    }

    /// Get interactions for content
    pub fn get_content_interactions(
        &self,
        content_id: &ContentId,
        interaction_type: Option<InteractionType>,
        include_anonymous: bool,
    ) -> SocialResult<Vec<Interaction>> {
        let interactions = self.interactions
            .get(content_id)
            .unwrap_or(&Vec::new());

        let filtered: Vec<Interaction> = interactions
            .iter()
            .filter(|i| {
                // Filter by type if specified
                if let Some(ref filter_type) = interaction_type {
                    if std::mem::discriminant(&i.interaction_type) != std::mem::discriminant(filter_type) {
                        return false;
                    }
                }

                // Filter anonymous if not requested
                if !include_anonymous && i.privacy_level != PrivacyLevel::Public {
                    return false;
                }

                true
            })
            .cloned()
            .collect();

        Ok(filtered)
    }

    /// Get interaction counts for content
    pub fn get_interaction_counts(&self, content_id: &ContentId) -> SocialResult<InteractionCounts> {
        let interactions = self.interactions
            .get(content_id)
            .unwrap_or(&Vec::new());

        let mut counts = InteractionCounts::default();

        for interaction in interactions {
            match &interaction.interaction_type {
                InteractionType::Like => counts.likes += 1,
                InteractionType::Dislike => counts.dislikes += 1,
                InteractionType::Vote { choice, weight, .. } => {
                    if *choice {
                        counts.votes_yes += 1;
                        counts.vote_weight_yes += weight;
                    } else {
                        counts.votes_no += 1;
                        counts.vote_weight_no += weight;
                    }
                },
                InteractionType::Reply { .. } => counts.replies += 1,
                InteractionType::Comment { .. } => counts.comments += 1,
                InteractionType::Share { .. } => counts.shares += 1,
                InteractionType::Bookmark => counts.bookmarks += 1,
                InteractionType::Report { .. } => counts.reports += 1,
            }
        }

        Ok(counts)
    }

    /// Verify interaction proof
    pub async fn verify_interaction_proof(
        &self,
        interaction: &Interaction,
        auth_service: &AuthService,
    ) -> SocialResult<bool> {
        if let Some(proof_data) = &interaction.proof {
            // In practice, this would verify the zk-STARK proof
            // For now, simplified verification
            return Ok(!proof_data.is_empty());
        }

        // Public interactions don't need proofs
        Ok(interaction.privacy_level == PrivacyLevel::Public)
    }

    // Private helper methods

    fn store_interaction(&mut self, interaction: Interaction) -> SocialResult<()> {
        self.interactions
            .entry(interaction.content_id.clone())
            .or_insert_with(Vec::new)
            .push(interaction);
        Ok(())
    }

    fn check_rate_limits(&mut self, user_id: &str) -> SocialResult<()> {
        let now = Utc::now();
        let user_hash = self.hash_user_id(user_id);

        let rate_data = self.rate_limits
            .entry(user_hash.clone())
            .or_insert_with(|| RateLimitData {
                current_window_count: 0,
                window_start: now,
                violations: 0,
            });

        // Reset window if expired
        if now.signed_duration_since(rate_data.window_start) > Duration::hours(1) {
            rate_data.current_window_count = 0;
            rate_data.window_start = now;
        }

        // Check limit
        if rate_data.current_window_count >= self.settings.max_interactions_per_hour {
            rate_data.violations += 1;
            return Err(SocialError::InvalidOperation(
                "Rate limit exceeded".to_string()
            ));
        }

        rate_data.current_window_count += 1;
        Ok(())
    }

    fn check_report_rate_limits(&mut self, user_id: &str) -> SocialResult<()> {
        // More restrictive rate limit for reports
        let now = Utc::now();
        let user_hash = self.hash_user_id(user_id);

        // Allow max 5 reports per hour
        let report_limit = 5;
        
        // This would be implemented similar to check_rate_limits
        // but with separate tracking for reports
        Ok(())
    }

    fn has_user_interacted(
        &self,
        user_id: &str,
        content_id: &ContentId,
        interaction_type: &InteractionType,
    ) -> SocialResult<bool> {
        if let Some(interactions) = self.interactions.get(content_id) {
            for interaction in interactions {
                if let Some(ref int_user_id) = interaction.user_id {
                    if int_user_id == user_id && 
                       std::mem::discriminant(&interaction.interaction_type) == std::mem::discriminant(interaction_type) {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    fn calculate_reply_depth(&self, content_id: &ContentId, parent_reply_id: &Option<String>) -> SocialResult<u32> {
        if let Some(parent_id) = parent_reply_id {
            // Find parent reply and calculate depth
            if let Some(interactions) = self.interactions.get(content_id) {
                for interaction in interactions {
                    if interaction.id == *parent_id {
                        if let InteractionType::Reply { .. } = &interaction.interaction_type {
                            // This is simplified - would need to recursively calculate depth
                            return Ok(1); // Placeholder
                        }
                    }
                }
            }
        }
        Ok(0)
    }

    fn generate_interaction_id(&self, user_id: &str, content_id: &ContentId, interaction_type: &str) -> String {
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        hasher.update(user_id.as_bytes());
        hasher.update(content_id.as_bytes());
        hasher.update(interaction_type.as_bytes());
        hasher.update(&Utc::now().timestamp().to_le_bytes());
        
        format!("int_{}", hex::encode(hasher.finalize()))
    }

    async fn generate_interaction_proof(
        &self,
        identity: &Identity,
        content_id: &ContentId,
        interaction_type: &InteractionType,
        auth_service: &AuthService,
    ) -> SocialResult<Vec<u8>> {
        // Generate zk-STARK proof for anonymous interaction
        // This would prove:
        // 1. User is authorized to interact
        // 2. Interaction is unique (no double voting)
        // 3. User meets eligibility requirements
        // 4. Timestamp is valid (prevents replay)

        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        hasher.update(identity.get_id().as_bytes());
        hasher.update(content_id.as_bytes());
        hasher.update(format!("{:?}", interaction_type).as_bytes());
        hasher.update(b"interaction_proof");
        
        Ok(hasher.finalize().to_vec())
    }

    fn get_user_reputation(&self, user_id: &str) -> f64 {
        let user_hash = self.hash_user_id(user_id);
        self.user_history
            .get(&user_hash)
            .map(|history| history.reputation)
            .unwrap_or(1.0) // Default reputation
    }

    fn update_user_history(&mut self, user_id: &str, interaction: &Interaction) {
        let user_hash = self.hash_user_id(user_id);
        let history = self.user_history
            .entry(user_hash.clone())
            .or_insert_with(|| UserInteractionHistory {
                user_hash: user_hash.clone(),
                recent_interactions: Vec::new(),
                interaction_counts: HashMap::new(),
                last_interaction: Utc::now(),
                reputation: 1.0,
            });

        history.recent_interactions.push(interaction.created_at);
        history.last_interaction = interaction.created_at;
        
        // Update interaction type counts
        *history.interaction_counts
            .entry(interaction.interaction_type.clone())
            .or_insert(0) += 1;

        // Update reputation based on voting behavior
        if let InteractionType::Vote { choice, weight, .. } = &interaction.interaction_type {
            // Positive voting behavior slightly increases reputation
            if *choice {
                history.reputation += 0.001; // Small positive boost
            }
            // Cap reputation at reasonable bounds
            history.reputation = history.reputation.min(10.0).max(0.1);
        }

        // Keep only recent interactions (last 24 hours)
        let cutoff = Utc::now() - Duration::hours(24);
        history.recent_interactions.retain(|&timestamp| timestamp > cutoff);
    }

    /// Calculate vote weight based on reputation and vote type
    fn calculate_vote_weight(&self, reputation: f64, vote_type: &VoteType) -> f64 {
        match vote_type {
            VoteType::Like => reputation,
            VoteType::Proposal => reputation * 1.5, // Proposal votes have higher weight
            VoteType::Poll => reputation * 0.8,     // Poll votes have lower weight
            VoteType::Quality => reputation * 1.2,  // Quality votes moderately weighted
            VoteType::Moderation => reputation * 2.0, // Moderation votes heavily weighted
        }
    }

    /// Get voting results for content
    pub fn get_voting_results(&self, content_id: &ContentId) -> SocialResult<VotingResults> {
        let interactions = self.interactions
            .get(content_id)
            .unwrap_or(&Vec::new());

        let mut results = VotingResults::default();

        for interaction in interactions {
            if let InteractionType::Vote { choice, weight, .. } = &interaction.interaction_type {
                results.total_votes += 1;
                results.total_weight += weight;
                
                if *choice {
                    results.yes_votes += 1;
                    results.yes_weight += weight;
                } else {
                    results.no_votes += 1;
                    results.no_weight += weight;
                }
            }
        }

        // Calculate percentages
        if results.total_votes > 0 {
            results.yes_percentage = (results.yes_votes as f64 / results.total_votes as f64) * 100.0;
            results.no_percentage = (results.no_votes as f64 / results.total_votes as f64) * 100.0;
        }
        
        if results.total_weight > 0.0 {
            results.yes_weight_percentage = (results.yes_weight / results.total_weight) * 100.0;
            results.no_weight_percentage = (results.no_weight / results.total_weight) * 100.0;
        }

        Ok(results)
    }

    /// Get vote breakdown by type
    pub fn get_vote_breakdown(&self, content_id: &ContentId) -> SocialResult<VoteBreakdown> {
        let interactions = self.interactions
            .get(content_id)
            .unwrap_or(&Vec::new());

        let mut breakdown = VoteBreakdown::default();

        for interaction in interactions {
            if let InteractionType::Vote { choice, weight, .. } = &interaction.interaction_type {
                // Extract vote type from metadata
                if let Some(metadata) = &interaction.metadata {
                    if let Ok(vote) = bincode::deserialize::<Vote>(metadata) {
                        let entry = breakdown.by_type.entry(vote.vote_type.clone())
                            .or_insert_with(VotingResults::default);
                        
                        entry.total_votes += 1;
                        entry.total_weight += weight;
                        
                        if *choice {
                            entry.yes_votes += 1;
                            entry.yes_weight += weight;
                        } else {
                            entry.no_votes += 1;
                            entry.no_weight += weight;
                        }
                    }
                }
            }
        }

        // Calculate percentages for each type
        for (_, results) in breakdown.by_type.iter_mut() {
            if results.total_votes > 0 {
                results.yes_percentage = (results.yes_votes as f64 / results.total_votes as f64) * 100.0;
                results.no_percentage = (results.no_votes as f64 / results.total_votes as f64) * 100.0;
            }
            
            if results.total_weight > 0.0 {
                results.yes_weight_percentage = (results.yes_weight / results.total_weight) * 100.0;
                results.no_weight_percentage = (results.no_weight / results.total_weight) * 100.0;
            }
        }

        Ok(breakdown)
    }

    fn hash_user_id(&self, user_id: &str) -> String {
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        hasher.update(user_id.as_bytes());
        hasher.update(b"user_history_salt");
        
        hex::encode(hasher.finalize())
    }
}

/// Interaction counts for content
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct InteractionCounts {
    pub likes: u32,
    pub dislikes: u32,
    pub votes_yes: u32,
    pub votes_no: u32,
    pub vote_weight_yes: f64,
    pub vote_weight_no: f64,
    pub replies: u32,
    pub comments: u32,
    pub shares: u32,
    pub bookmarks: u32,
    pub reports: u32,
}

impl SpamDetector {
    fn new() -> Self {
        Self {
            spam_patterns: Self::default_patterns(),
            suspicious_indicators: HashMap::new(),
        }
    }

    fn default_patterns() -> Vec<SpamPattern> {
        vec![
            SpamPattern {
                name: "Rapid Fire Interactions".to_string(),
                detector: Self::detect_rapid_fire,
                threshold: 0.8,
            },
            SpamPattern {
                name: "Repetitive Content".to_string(),
                detector: Self::detect_repetitive_content,
                threshold: 0.7,
            },
        ]
    }

    fn detect_rapid_fire(_interaction: &Interaction, history: &UserInteractionHistory) -> f64 {
        // Check if user is making too many interactions too quickly
        let recent_count = history.recent_interactions.len();
        if recent_count > 50 {
            0.9 // High spam probability
        } else if recent_count > 20 {
            0.5 // Medium spam probability
        } else {
            0.1 // Low spam probability
        }
    }

    fn detect_repetitive_content(_interaction: &Interaction, _history: &UserInteractionHistory) -> f64 {
        // Check for repetitive interaction patterns
        // This would analyze interaction content for similarities
        0.1 // Placeholder
    }
}

/// Voting results aggregation
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct VotingResults {
    pub total_votes: u32,
    pub yes_votes: u32,
    pub no_votes: u32,
    pub total_weight: f64,
    pub yes_weight: f64,
    pub no_weight: f64,
    pub yes_percentage: f64,
    pub no_percentage: f64,
    pub yes_weight_percentage: f64,
    pub no_weight_percentage: f64,
}

/// Vote breakdown by type
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct VoteBreakdown {
    pub by_type: HashMap<VoteType, VotingResults>,
}

impl Default for InteractionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Identity, Content, ContentType};
    use axon_identity::AuthService;

    #[tokio::test]
    async fn test_like_content() {
        let mut manager = InteractionManager::new();
        let identity = Identity::new_for_test("user1");
        let content_id = ContentId::new("test_content");
        let auth_service = AuthService::new_for_test();

        let interaction = manager.like_content(
            &identity,
            content_id.clone(),
            true,
            PrivacyLevel::Public,
            &auth_service,
        ).await.unwrap();

        // Like is now converted to Vote with Like type
        if let InteractionType::Vote { choice, .. } = interaction.interaction_type {
            assert!(choice); // Should be true for like
        } else {
            panic!("Expected Vote interaction type for like");
        }
        assert_eq!(interaction.content_id, content_id);
        assert_eq!(interaction.privacy_level, PrivacyLevel::Public);
    }

    #[tokio::test]
    async fn test_vote_on_content() {
        let mut manager = InteractionManager::new();
        let identity = Identity::new_for_test("user1");
        let content_id = ContentId::new("test_proposal");
        let auth_service = AuthService::new_for_test();

        let interaction = manager.vote_on_content(
            &identity,
            content_id.clone(),
            true,
            VoteType::Proposal,
            Some("I agree with this proposal".to_string()),
            PrivacyLevel::Anonymous,
            &auth_service,
        ).await.unwrap();

        if let InteractionType::Vote { choice, weight, reason } = &interaction.interaction_type {
            assert!(*choice);
            assert!(*weight > 0.0); // Should have weight based on reputation
            assert!(reason.is_some());
        } else {
            panic!("Expected Vote interaction type");
        }

        assert_eq!(interaction.content_id, content_id);
        assert_eq!(interaction.privacy_level, PrivacyLevel::Anonymous);
        assert!(interaction.user_id.is_none()); // Anonymous
        assert!(interaction.proof.is_some()); // Should have proof for anonymous
    }

    #[tokio::test]
    async fn test_voting_results() {
        let mut manager = InteractionManager::new();
        let identity1 = Identity::new_for_test("user1");
        let identity2 = Identity::new_for_test("user2");
        let content_id = ContentId::new("test_poll");
        let auth_service = AuthService::new_for_test();

        // Add multiple votes
        manager.vote_on_content(&identity1, content_id.clone(), true, VoteType::Poll, None, PrivacyLevel::Public, &auth_service).await.unwrap();
        manager.vote_on_content(&identity2, content_id.clone(), false, VoteType::Poll, None, PrivacyLevel::Public, &auth_service).await.unwrap();

        let results = manager.get_voting_results(&content_id).unwrap();
        assert_eq!(results.total_votes, 2);
        assert_eq!(results.yes_votes, 1);
        assert_eq!(results.no_votes, 1);
        assert_eq!(results.yes_percentage, 50.0);
        assert_eq!(results.no_percentage, 50.0);
        assert!(results.total_weight > 0.0);
    }

    #[tokio::test]
    async fn test_reply_to_content() {
        let mut manager = InteractionManager::new();
        let identity = Identity::new_for_test("user1");
        let content_id = ContentId::new("test_content");
        let auth_service = AuthService::new_for_test();

        let reply_content = Content::new_for_test("This is a reply", ContentType::Text);

        let interaction = manager.reply_to_content(
            &identity,
            content_id.clone(),
            None,
            reply_content.clone(),
            PrivacyLevel::Anonymous,
            &auth_service,
        ).await.unwrap();

        if let InteractionType::Reply { content, parent_id } = &interaction.interaction_type {
            assert_eq!(content.get_text(), reply_content.get_text());
            assert!(parent_id.is_none());
        } else {
            panic!("Expected Reply interaction type");
        }

        assert_eq!(interaction.content_id, content_id);
        assert_eq!(interaction.privacy_level, PrivacyLevel::Anonymous);
        assert!(interaction.user_id.is_none()); // Anonymous
        assert!(interaction.proof.is_some()); // Should have proof for anonymous
    }

    #[tokio::test]
    async fn test_interaction_counts() {
        let mut manager = InteractionManager::new();
        let identity = Identity::new_for_test("user1");
        let content_id = ContentId::new("test_content");
        let auth_service = AuthService::new_for_test();

        // Add multiple interactions
        manager.like_content(&identity, content_id.clone(), true, PrivacyLevel::Public, &auth_service).await.unwrap();
        manager.vote_on_content(&identity, content_id.clone(), true, VoteType::Quality, None, PrivacyLevel::Public, &auth_service).await.unwrap();
        
        let reply_content = Content::new_for_test("Reply", ContentType::Text);
        manager.reply_to_content(&identity, content_id.clone(), None, reply_content, PrivacyLevel::Public, &auth_service).await.unwrap();

        let counts = manager.get_interaction_counts(&content_id).unwrap();
        assert_eq!(counts.votes_yes, 2); // Like + Quality vote
        assert_eq!(counts.votes_no, 0);
        assert_eq!(counts.replies, 1);
        assert!(counts.vote_weight_yes > 0.0);
    }

    #[test]
    fn test_rate_limiting() {
        let mut manager = InteractionManager::new();
        let user_id = "test_user";

        // First interaction should succeed
        assert!(manager.check_rate_limits(user_id).is_ok());

        // Simulate hitting rate limit
        for _ in 0..manager.settings.max_interactions_per_hour {
            manager.check_rate_limits(user_id).unwrap();
        }

        // Next interaction should fail
        assert!(manager.check_rate_limits(user_id).is_err());
    }

    #[test]
    fn test_vote_weight_calculation() {
        let manager = InteractionManager::new();
        let reputation = 2.0;
        
        assert_eq!(manager.calculate_vote_weight(reputation, &VoteType::Like), 2.0);
        assert_eq!(manager.calculate_vote_weight(reputation, &VoteType::Proposal), 3.0);
        assert_eq!(manager.calculate_vote_weight(reputation, &VoteType::Poll), 1.6);
        assert_eq!(manager.calculate_vote_weight(reputation, &VoteType::Quality), 2.4);
        assert_eq!(manager.calculate_vote_weight(reputation, &VoteType::Moderation), 4.0);
    }

    #[tokio::test]
    async fn test_vote_breakdown() {
        let mut manager = InteractionManager::new();
        let identity = Identity::new_for_test("user1");
        let content_id = ContentId::new("test_content");
        let auth_service = AuthService::new_for_test();

        // Add different types of votes
        manager.vote_on_content(&identity, content_id.clone(), true, VoteType::Like, None, PrivacyLevel::Public, &auth_service).await.unwrap();
        manager.vote_on_content(&identity, content_id.clone(), false, VoteType::Quality, None, PrivacyLevel::Public, &auth_service).await.unwrap();

        let breakdown = manager.get_vote_breakdown(&content_id).unwrap();
        assert!(breakdown.by_type.contains_key(&VoteType::Like));
        assert!(breakdown.by_type.contains_key(&VoteType::Quality));
        
        let like_results = &breakdown.by_type[&VoteType::Like];
        assert_eq!(like_results.yes_votes, 1);
        assert_eq!(like_results.no_votes, 0);
        
        let quality_results = &breakdown.by_type[&VoteType::Quality];
        assert_eq!(quality_results.yes_votes, 0);
        assert_eq!(quality_results.no_votes, 1);
    }
}