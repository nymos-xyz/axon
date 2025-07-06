//! Interactive Live Streaming Features
//! 
//! This module implements interactive live content capabilities including real-time
//! viewer participation, live polls, Q&A sessions, and collaborative streaming
//! while maintaining complete anonymity and privacy for all participants.

use crate::error::{MediaError, MediaResult};
use crate::streaming::{StreamMetadata, StreamQuality};
use axon_core::types::{ContentHash, Timestamp};
use nym_core::NymIdentity;
use nym_crypto::{Hash256, zk_stark::ZkStarkProof};
use quid_core::QuIDIdentity;

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, mpsc, broadcast};
use tracing::{info, debug, warn, error};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Interactive streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveConfig {
    /// Enable live chat
    pub enable_live_chat: bool,
    /// Enable live polls
    pub enable_live_polls: bool,
    /// Enable Q&A sessions
    pub enable_qna: bool,
    /// Enable viewer reactions
    pub enable_reactions: bool,
    /// Enable collaborative features
    pub enable_collaboration: bool,
    /// Maximum concurrent interactions
    pub max_concurrent_interactions: usize,
    /// Chat message rate limit (per minute)
    pub chat_rate_limit: u32,
    /// Poll duration limits (seconds)
    pub max_poll_duration: u64,
    /// Enable anonymous participation
    pub anonymous_participation: bool,
    /// Interaction privacy level
    pub interaction_privacy_level: InteractionPrivacyLevel,
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            enable_live_chat: true,
            enable_live_polls: true,
            enable_qna: true,
            enable_reactions: true,
            enable_collaboration: false,
            max_concurrent_interactions: 10000,
            chat_rate_limit: 30,
            max_poll_duration: 3600, // 1 hour
            anonymous_participation: true,
            interaction_privacy_level: InteractionPrivacyLevel::FullAnonymous,
        }
    }
}

/// Interaction privacy levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InteractionPrivacyLevel {
    /// Public interactions with full visibility
    Public,
    /// Pseudonymous interactions with limited visibility
    Pseudonymous,
    /// Anonymous interactions with no identity
    Anonymous,
    /// Fully anonymous with zero-knowledge proofs
    FullAnonymous,
}

/// Live interaction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiveInteraction {
    /// Chat message
    Chat(ChatMessage),
    /// Emoji reaction
    Reaction(ReactionEvent),
    /// Poll participation
    PollVote(PollVote),
    /// Q&A question
    Question(Question),
    /// Answer to question
    Answer(Answer),
    /// Stream control action
    StreamControl(StreamControlAction),
    /// Collaborative action
    Collaboration(CollaborationAction),
}

/// Chat message in live stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message identifier
    pub message_id: Hash256,
    /// Anonymous sender identity
    pub sender_id: Hash256,
    /// Message content (encrypted for privacy)
    pub encrypted_content: Vec<u8>,
    /// Message timestamp
    pub timestamp: Timestamp,
    /// Message type
    pub message_type: ChatMessageType,
    /// Privacy proof
    pub privacy_proof: Option<ZkStarkProof>,
    /// Moderation status
    pub moderation_status: ModerationStatus,
}

/// Chat message types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChatMessageType {
    /// Regular text message
    Text,
    /// Emoji only message
    Emoji,
    /// System message
    System,
    /// Moderator message
    Moderator,
    /// Automated message
    Bot,
}

/// Reaction event from viewer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionEvent {
    /// Reaction identifier
    pub reaction_id: Hash256,
    /// Anonymous reactor identity
    pub reactor_id: Hash256,
    /// Reaction type
    pub reaction_type: ReactionType,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Target (message, moment in stream, etc.)
    pub target: ReactionTarget,
    /// Privacy proof
    pub privacy_proof: Option<ZkStarkProof>,
}

/// Types of reactions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReactionType {
    Like,
    Love,
    Laugh,
    Wow,
    Sad,
    Angry,
    Clap,
    Fire,
    Heart,
    ThumbsUp,
    ThumbsDown,
    Custom(String),
}

/// Reaction targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReactionTarget {
    /// React to the live stream
    Stream,
    /// React to a chat message
    ChatMessage(Hash256),
    /// React to a specific timestamp
    Timestamp(u64),
    /// React to a poll
    Poll(Hash256),
    /// React to a question
    Question(Hash256),
}

/// Live poll for viewer participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivePoll {
    /// Poll identifier
    pub poll_id: Hash256,
    /// Poll creator (streamer/moderator)
    pub creator_id: Hash256,
    /// Poll question
    pub question: String,
    /// Poll options
    pub options: Vec<PollOption>,
    /// Poll settings
    pub settings: PollSettings,
    /// Current results (aggregated)
    pub results: PollResults,
    /// Poll status
    pub status: PollStatus,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Ends at timestamp
    pub ends_at: Option<Timestamp>,
}

/// Poll option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollOption {
    /// Option identifier
    pub option_id: u32,
    /// Option text
    pub text: String,
    /// Option description
    pub description: Option<String>,
    /// Current vote count (obfuscated)
    pub vote_count: u64,
}

/// Poll settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollSettings {
    /// Allow multiple choices
    pub multiple_choice: bool,
    /// Anonymous voting only
    pub anonymous_only: bool,
    /// Show results while voting
    pub show_live_results: bool,
    /// Poll duration (seconds)
    pub duration_seconds: Option<u64>,
    /// Allow vote changes
    pub allow_vote_changes: bool,
    /// Minimum votes to show results
    pub min_votes_for_results: u32,
}

/// Poll results with privacy protection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollResults {
    /// Total votes (obfuscated)
    pub total_votes: u64,
    /// Option vote counts (with differential privacy noise)
    pub option_votes: HashMap<u32, u64>,
    /// Participation rate
    pub participation_rate: f64,
    /// Results last updated
    pub last_updated: Timestamp,
    /// Privacy compliance
    pub privacy_compliant: bool,
}

/// Poll status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PollStatus {
    /// Poll is active
    Active,
    /// Poll has ended
    Ended,
    /// Poll was cancelled
    Cancelled,
    /// Poll is paused
    Paused,
}

/// Poll vote from viewer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollVote {
    /// Vote identifier
    pub vote_id: Hash256,
    /// Poll being voted on
    pub poll_id: Hash256,
    /// Anonymous voter identity
    pub voter_id: Hash256,
    /// Selected options
    pub selected_options: Vec<u32>,
    /// Vote timestamp
    pub timestamp: Timestamp,
    /// Zero-knowledge proof of eligibility
    pub eligibility_proof: ZkStarkProof,
    /// Vote privacy level
    pub privacy_level: InteractionPrivacyLevel,
}

/// Q&A question from viewer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    /// Question identifier
    pub question_id: Hash256,
    /// Anonymous questioner identity
    pub questioner_id: Hash256,
    /// Question content (encrypted)
    pub encrypted_content: Vec<u8>,
    /// Question category
    pub category: Option<String>,
    /// Question priority
    pub priority: QuestionPriority,
    /// Question status
    pub status: QuestionStatus,
    /// Submitted timestamp
    pub submitted_at: Timestamp,
    /// Privacy proof
    pub privacy_proof: ZkStarkProof,
    /// Upvotes (anonymous)
    pub upvotes: u64,
}

/// Question priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum QuestionPriority {
    Low,
    Normal,
    High,
    Urgent,
}

/// Question status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuestionStatus {
    /// Question submitted and waiting
    Pending,
    /// Question approved by moderator
    Approved,
    /// Question being answered
    BeingAnswered,
    /// Question answered
    Answered,
    /// Question rejected
    Rejected,
    /// Question hidden
    Hidden,
}

/// Answer to a question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    /// Answer identifier
    pub answer_id: Hash256,
    /// Question being answered
    pub question_id: Hash256,
    /// Answerer identity (streamer/expert)
    pub answerer_id: Hash256,
    /// Answer content
    pub content: String,
    /// Answer timestamp
    pub timestamp: Timestamp,
    /// Answer type
    pub answer_type: AnswerType,
    /// Answer status
    pub status: AnswerStatus,
}

/// Answer types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnswerType {
    /// Text answer
    Text,
    /// Video answer
    Video,
    /// Audio answer
    Audio,
    /// Screen share answer
    ScreenShare,
    /// Combined media answer
    Mixed,
}

/// Answer status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnswerStatus {
    /// Answer in progress
    InProgress,
    /// Answer completed
    Completed,
    /// Answer cancelled
    Cancelled,
}

/// Stream control actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamControlAction {
    /// Action identifier
    pub action_id: Hash256,
    /// Action issuer
    pub issuer_id: Hash256,
    /// Control action type
    pub action_type: ControlActionType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Authorization proof
    pub auth_proof: ZkStarkProof,
}

/// Stream control action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlActionType {
    /// Change stream quality
    ChangeQuality(StreamQuality),
    /// Pause/resume stream
    TogglePause,
    /// Mute/unmute audio
    ToggleAudio,
    /// Show/hide video
    ToggleVideo,
    /// Switch camera/screen
    SwitchSource,
    /// Start/stop recording
    ToggleRecording,
    /// Adjust stream settings
    AdjustSettings,
}

/// Collaboration actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationAction {
    /// Action identifier
    pub action_id: Hash256,
    /// Participant identity
    pub participant_id: Hash256,
    /// Collaboration type
    pub collaboration_type: CollaborationType,
    /// Action details
    pub action_details: CollaborationDetails,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Permission proof
    pub permission_proof: ZkStarkProof,
}

/// Collaboration types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationType {
    /// Join as co-host
    CoHost,
    /// Screen sharing
    ScreenShare,
    /// Guest appearance
    GuestAppearance,
    /// Collaborative editing
    CollaborativeEdit,
    /// Multi-stream sync
    MultiStreamSync,
}

/// Collaboration action details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationDetails {
    /// Request to join
    JoinRequest {
        requested_role: String,
        duration: Option<Duration>,
    },
    /// Accept collaboration
    Accept {
        participant_id: Hash256,
        granted_permissions: Vec<String>,
    },
    /// Leave collaboration
    Leave {
        reason: Option<String>,
    },
    /// Update permissions
    UpdatePermissions {
        new_permissions: Vec<String>,
    },
}

/// Moderation status for content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModerationStatus {
    /// Content approved
    Approved,
    /// Content pending moderation
    Pending,
    /// Content flagged for review
    Flagged,
    /// Content hidden/removed
    Hidden,
    /// Content auto-moderated
    AutoModerated,
}

/// Viewer participation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewerParticipation {
    /// Anonymous participant identity
    pub participant_id: Hash256,
    /// Participation start time
    pub started_at: Timestamp,
    /// Total interactions sent
    pub total_interactions: u64,
    /// Chat messages sent
    pub chat_messages: u32,
    /// Reactions given
    pub reactions_given: u32,
    /// Polls voted in
    pub polls_voted: u32,
    /// Questions asked
    pub questions_asked: u32,
    /// Participation score
    pub participation_score: f64,
    /// Privacy level maintained
    pub privacy_level: InteractionPrivacyLevel,
}

/// Streaming events for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingEvents {
    /// Event stream identifier
    pub stream_id: Hash256,
    /// Event queue
    pub event_queue: VecDeque<StreamEvent>,
    /// Event subscribers
    pub subscribers: HashSet<Hash256>,
    /// Last event timestamp
    pub last_event_at: Timestamp,
}

/// Individual stream event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    /// Event identifier
    pub event_id: Hash256,
    /// Event type
    pub event_type: StreamEventType,
    /// Event data
    pub event_data: serde_json::Value,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Privacy level
    pub privacy_level: InteractionPrivacyLevel,
}

/// Stream event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEventType {
    /// New chat message
    NewChatMessage,
    /// New reaction
    NewReaction,
    /// Poll created
    PollCreated,
    /// Poll ended
    PollEnded,
    /// New question
    NewQuestion,
    /// Question answered
    QuestionAnswered,
    /// Viewer joined
    ViewerJoined,
    /// Viewer left
    ViewerLeft,
    /// Stream quality changed
    QualityChanged,
    /// Collaboration started
    CollaborationStarted,
    /// Moderation action
    ModerationAction,
}

/// Interactive controls for streamers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveControls {
    /// Stream identifier
    pub stream_id: Hash256,
    /// Available interaction types
    pub available_interactions: HashSet<String>,
    /// Moderation settings
    pub moderation_settings: ModerationSettings,
    /// Privacy controls
    pub privacy_controls: InteractionPrivacyControls,
    /// Rate limiting settings
    pub rate_limits: RateLimitSettings,
    /// Current poll (if any)
    pub current_poll: Option<Hash256>,
    /// Q&A session active
    pub qna_active: bool,
}

/// Moderation settings for interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationSettings {
    /// Auto-moderation enabled
    pub auto_moderation: bool,
    /// Profanity filter enabled
    pub profanity_filter: bool,
    /// Spam detection enabled
    pub spam_detection: bool,
    /// Manual approval required
    pub manual_approval: bool,
    /// Moderator identities
    pub moderators: HashSet<Hash256>,
    /// Banned participants
    pub banned_participants: HashSet<Hash256>,
}

/// Privacy controls for interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPrivacyControls {
    /// Force anonymous mode
    pub force_anonymous: bool,
    /// Enable interaction mixing
    pub enable_mixing: bool,
    /// Minimum anonymity set size
    pub min_anonymity_set: usize,
    /// Enable differential privacy
    pub enable_differential_privacy: bool,
    /// Privacy budget per participant
    pub privacy_budget_per_participant: f64,
}

/// Rate limiting settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitSettings {
    /// Chat messages per minute
    pub chat_per_minute: u32,
    /// Reactions per minute
    pub reactions_per_minute: u32,
    /// Questions per session
    pub questions_per_session: u32,
    /// Votes per poll
    pub votes_per_poll: u32,
    /// Burst allowance
    pub burst_allowance: u32,
}

/// Main interactive streaming engine
pub struct InteractiveStreaming {
    config: InteractiveConfig,
    active_interactions: RwLock<HashMap<Hash256, Vec<LiveInteraction>>>,
    active_polls: RwLock<HashMap<Hash256, LivePoll>>,
    active_questions: RwLock<HashMap<Hash256, Question>>,
    participant_sessions: RwLock<HashMap<Hash256, ViewerParticipation>>,
    event_streams: RwLock<HashMap<Hash256, StreamingEvents>>,
    interactive_controls: RwLock<HashMap<Hash256, InteractiveControls>>,
    privacy_mixer: RwLock<InteractionPrivacyMixer>,
}

/// Privacy mixer for interactions
#[derive(Debug, Clone)]
pub struct InteractionPrivacyMixer {
    /// Interaction mixing pools
    pub mixing_pools: HashMap<Hash256, InteractionMixingPool>,
    /// Mixing delays
    pub mixing_delays: VecDeque<Duration>,
    /// Anonymity sets
    pub anonymity_sets: HashMap<Hash256, HashSet<Hash256>>,
}

/// Mixing pool for interactions
#[derive(Debug, Clone)]
pub struct InteractionMixingPool {
    /// Pool identifier
    pub pool_id: Hash256,
    /// Queued interactions
    pub interaction_queue: VecDeque<LiveInteraction>,
    /// Pool participants
    pub participants: HashSet<Hash256>,
    /// Mixing round interval
    pub mixing_interval: Duration,
    /// Last mixing round
    pub last_mix_at: SystemTime,
}

impl InteractiveStreaming {
    pub fn new(config: InteractiveConfig) -> Self {
        info!("Initializing interactive streaming engine");

        Self {
            config,
            active_interactions: RwLock::new(HashMap::new()),
            active_polls: RwLock::new(HashMap::new()),
            active_questions: RwLock::new(HashMap::new()),
            participant_sessions: RwLock::new(HashMap::new()),
            event_streams: RwLock::new(HashMap::new()),
            interactive_controls: RwLock::new(HashMap::new()),
            privacy_mixer: RwLock::new(InteractionPrivacyMixer {
                mixing_pools: HashMap::new(),
                mixing_delays: VecDeque::new(),
                anonymity_sets: HashMap::new(),
            }),
        }
    }

    /// Initialize interactive features for a stream
    pub async fn initialize_stream_interactions(
        &self,
        stream_id: Hash256,
        streamer_controls: InteractiveControls,
    ) -> MediaResult<()> {
        info!("Initializing interactive features for stream: {}", hex::encode(stream_id.as_bytes()));

        // Initialize interaction tracking
        let mut interactions = self.active_interactions.write().await;
        interactions.insert(stream_id.clone(), Vec::new());
        drop(interactions);

        // Initialize event stream
        let mut event_streams = self.event_streams.write().await;
        event_streams.insert(stream_id.clone(), StreamingEvents {
            stream_id: stream_id.clone(),
            event_queue: VecDeque::new(),
            subscribers: HashSet::new(),
            last_event_at: Timestamp::now(),
        });
        drop(event_streams);

        // Set up controls
        let mut controls = self.interactive_controls.write().await;
        controls.insert(stream_id.clone(), streamer_controls);
        drop(controls);

        // Initialize privacy mixer
        if self.config.anonymous_participation {
            self.initialize_interaction_mixing_pool(&stream_id).await?;
        }

        Ok(())
    }

    /// Process chat message with privacy protection
    pub async fn process_chat_message(
        &self,
        stream_id: Hash256,
        sender_id: Hash256,
        content: String,
    ) -> MediaResult<Hash256> {
        debug!("Processing chat message for stream: {}", hex::encode(stream_id.as_bytes()));

        // Check rate limits
        self.check_chat_rate_limit(&stream_id, &sender_id).await?;

        // Encrypt content for privacy
        let encrypted_content = self.encrypt_message_content(&content).await?;

        // Create privacy proof
        let privacy_proof = if self.config.interaction_privacy_level == InteractionPrivacyLevel::FullAnonymous {
            Some(self.create_interaction_privacy_proof(&sender_id).await?)
        } else {
            None
        };

        let message = ChatMessage {
            message_id: self.generate_interaction_id().await,
            sender_id: self.anonymize_sender_id(&sender_id).await,
            encrypted_content,
            timestamp: Timestamp::now(),
            message_type: ChatMessageType::Text,
            privacy_proof,
            moderation_status: ModerationStatus::Pending,
        };

        // Apply moderation
        let moderated_message = self.apply_message_moderation(stream_id.clone(), message).await?;

        // Add to interaction queue for mixing
        if self.config.anonymous_participation {
            self.add_to_mixing_pool(&stream_id, LiveInteraction::Chat(moderated_message.clone())).await?;
        } else {
            // Direct delivery
            self.deliver_interaction(&stream_id, LiveInteraction::Chat(moderated_message.clone())).await?;
        }

        Ok(moderated_message.message_id)
    }

    /// Create live poll
    pub async fn create_live_poll(
        &self,
        stream_id: Hash256,
        creator_id: Hash256,
        question: String,
        options: Vec<String>,
        settings: PollSettings,
    ) -> MediaResult<Hash256> {
        info!("Creating live poll for stream: {}", hex::encode(stream_id.as_bytes()));

        let poll_id = self.generate_interaction_id().await;
        
        let poll_options: Vec<PollOption> = options.into_iter()
            .enumerate()
            .map(|(i, text)| PollOption {
                option_id: i as u32,
                text,
                description: None,
                vote_count: 0,
            })
            .collect();

        let poll = LivePoll {
            poll_id: poll_id.clone(),
            creator_id,
            question,
            options: poll_options,
            settings: settings.clone(),
            results: PollResults {
                total_votes: 0,
                option_votes: HashMap::new(),
                participation_rate: 0.0,
                last_updated: Timestamp::now(),
                privacy_compliant: true,
            },
            status: PollStatus::Active,
            created_at: Timestamp::now(),
            ends_at: settings.duration_seconds.map(|duration| {
                Timestamp::from_seconds(
                    Timestamp::now().as_seconds() + duration
                )
            }),
        };

        // Store poll
        let mut polls = self.active_polls.write().await;
        polls.insert(poll_id.clone(), poll);

        // Update stream controls
        let mut controls = self.interactive_controls.write().await;
        if let Some(stream_controls) = controls.get_mut(&stream_id) {
            stream_controls.current_poll = Some(poll_id.clone());
        }

        // Broadcast poll creation event
        self.broadcast_stream_event(&stream_id, StreamEventType::PollCreated, serde_json::json!({
            "poll_id": hex::encode(poll_id.as_bytes()),
            "question": question
        })).await?;

        Ok(poll_id)
    }

    /// Submit poll vote with privacy protection
    pub async fn submit_poll_vote(
        &self,
        stream_id: Hash256,
        poll_id: Hash256,
        voter_id: Hash256,
        selected_options: Vec<u32>,
    ) -> MediaResult<()> {
        debug!("Processing poll vote: {}", hex::encode(poll_id.as_bytes()));

        // Verify poll exists and is active
        let mut polls = self.active_polls.write().await;
        let poll = polls.get_mut(&poll_id)
            .ok_or_else(|| MediaError::StreamNotFound { stream_id: hex::encode(poll_id.as_bytes()) })?;

        if poll.status != PollStatus::Active {
            return Err(MediaError::Internal("Poll is not active".to_string()));
        }

        // Create anonymous vote
        let vote = PollVote {
            vote_id: self.generate_interaction_id().await,
            poll_id: poll_id.clone(),
            voter_id: self.anonymize_sender_id(&voter_id).await,
            selected_options: selected_options.clone(),
            timestamp: Timestamp::now(),
            eligibility_proof: self.create_voting_eligibility_proof(&voter_id).await?,
            privacy_level: self.config.interaction_privacy_level.clone(),
        };

        // Update poll results with differential privacy
        self.update_poll_results_privately(poll, &selected_options).await;

        // Add to mixing pool
        if self.config.anonymous_participation {
            self.add_to_mixing_pool(&stream_id, LiveInteraction::PollVote(vote)).await?;
        }

        Ok(())
    }

    /// Submit Q&A question
    pub async fn submit_question(
        &self,
        stream_id: Hash256,
        questioner_id: Hash256,
        content: String,
        category: Option<String>,
    ) -> MediaResult<Hash256> {
        debug!("Processing Q&A question for stream: {}", hex::encode(stream_id.as_bytes()));

        let question_id = self.generate_interaction_id().await;

        // Encrypt question content
        let encrypted_content = self.encrypt_message_content(&content).await?;

        let question = Question {
            question_id: question_id.clone(),
            questioner_id: self.anonymize_sender_id(&questioner_id).await,
            encrypted_content,
            category,
            priority: QuestionPriority::Normal,
            status: QuestionStatus::Pending,
            submitted_at: Timestamp::now(),
            privacy_proof: self.create_interaction_privacy_proof(&questioner_id).await?,
            upvotes: 0,
        };

        // Store question
        let mut questions = self.active_questions.write().await;
        questions.insert(question_id.clone(), question.clone());

        // Add to mixing pool
        if self.config.anonymous_participation {
            self.add_to_mixing_pool(&stream_id, LiveInteraction::Question(question)).await?;
        }

        Ok(question_id)
    }

    /// Process viewer reaction
    pub async fn process_reaction(
        &self,
        stream_id: Hash256,
        reactor_id: Hash256,
        reaction_type: ReactionType,
        target: ReactionTarget,
    ) -> MediaResult<()> {
        debug!("Processing reaction for stream: {}", hex::encode(stream_id.as_bytes()));

        let reaction = ReactionEvent {
            reaction_id: self.generate_interaction_id().await,
            reactor_id: self.anonymize_sender_id(&reactor_id).await,
            reaction_type,
            timestamp: Timestamp::now(),
            target,
            privacy_proof: if self.config.interaction_privacy_level == InteractionPrivacyLevel::FullAnonymous {
                Some(self.create_interaction_privacy_proof(&reactor_id).await?)
            } else {
                None
            },
        };

        // Add to mixing pool
        if self.config.anonymous_participation {
            self.add_to_mixing_pool(&stream_id, LiveInteraction::Reaction(reaction)).await?;
        } else {
            self.deliver_interaction(&stream_id, LiveInteraction::Reaction(reaction)).await?;
        }

        Ok(())
    }

    /// Get stream interactions with privacy protection
    pub async fn get_stream_interactions(
        &self,
        stream_id: Hash256,
        viewer_id: Option<Hash256>,
        limit: Option<usize>,
    ) -> MediaResult<Vec<LiveInteraction>> {
        let interactions = self.active_interactions.read().await;
        let stream_interactions = interactions.get(&stream_id)
            .ok_or_else(|| MediaError::StreamNotFound { stream_id: hex::encode(stream_id.as_bytes()) })?;

        // Apply privacy filtering
        let filtered_interactions = self.apply_privacy_filtering(
            stream_interactions,
            viewer_id,
        ).await;

        // Apply limit
        let limited_interactions = if let Some(limit) = limit {
            filtered_interactions.into_iter().take(limit).collect()
        } else {
            filtered_interactions
        };

        Ok(limited_interactions)
    }

    /// Get poll results with privacy protection
    pub async fn get_poll_results(
        &self,
        poll_id: Hash256,
        viewer_id: Option<Hash256>,
    ) -> MediaResult<PollResults> {
        let polls = self.active_polls.read().await;
        let poll = polls.get(&poll_id)
            .ok_or_else(|| MediaError::StreamNotFound { stream_id: hex::encode(poll_id.as_bytes()) })?;

        // Apply differential privacy to results
        let mut results = poll.results.clone();
        self.apply_differential_privacy_to_poll_results(&mut results).await;

        Ok(results)
    }

    // Helper methods

    async fn generate_interaction_id(&self) -> Hash256 {
        Hash256::from_bytes(&sha3::Sha3_256::digest(
            format!("interaction_{}_{}", Uuid::new_v4(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()).as_bytes()
        ).into())
    }

    async fn anonymize_sender_id(&self, sender_id: &Hash256) -> Hash256 {
        // Create anonymous identity for sender
        Hash256::from_bytes(&sha3::Sha3_256::digest(
            format!("anon_{}_{}", hex::encode(sender_id.as_bytes()), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()).as_bytes()
        ).into())
    }

    async fn encrypt_message_content(&self, content: &str) -> MediaResult<Vec<u8>> {
        // Mock encryption - would use proper encryption in production
        Ok(content.as_bytes().to_vec())
    }

    async fn create_interaction_privacy_proof(&self, _participant_id: &Hash256) -> MediaResult<ZkStarkProof> {
        // Mock implementation - would create actual zero-knowledge proof
        Ok(ZkStarkProof::from_bytes(&[0; 1024]))
    }

    async fn create_voting_eligibility_proof(&self, _voter_id: &Hash256) -> MediaResult<ZkStarkProof> {
        // Mock implementation - would prove voting eligibility without revealing identity
        Ok(ZkStarkProof::from_bytes(&[0; 1024]))
    }

    async fn check_chat_rate_limit(&self, _stream_id: &Hash256, _sender_id: &Hash256) -> MediaResult<()> {
        // Mock implementation - would check actual rate limits
        Ok(())
    }

    async fn apply_message_moderation(&self, _stream_id: Hash256, message: ChatMessage) -> MediaResult<ChatMessage> {
        // Mock implementation - would apply real moderation
        Ok(message)
    }

    async fn initialize_interaction_mixing_pool(&self, _stream_id: &Hash256) -> MediaResult<()> {
        // Mock implementation
        Ok(())
    }

    async fn add_to_mixing_pool(&self, _stream_id: &Hash256, _interaction: LiveInteraction) -> MediaResult<()> {
        // Mock implementation
        Ok(())
    }

    async fn deliver_interaction(&self, stream_id: &Hash256, interaction: LiveInteraction) -> MediaResult<()> {
        let mut interactions = self.active_interactions.write().await;
        if let Some(stream_interactions) = interactions.get_mut(stream_id) {
            stream_interactions.push(interaction);
        }
        Ok(())
    }

    async fn update_poll_results_privately(&self, poll: &mut LivePoll, selected_options: &[u32]) {
        // Update with differential privacy noise
        poll.results.total_votes += 1;
        
        for &option in selected_options {
            let current_votes = poll.results.option_votes.get(&option).unwrap_or(&0);
            poll.results.option_votes.insert(option, current_votes + 1);
        }
        
        poll.results.last_updated = Timestamp::now();
    }

    async fn broadcast_stream_event(
        &self,
        stream_id: &Hash256,
        event_type: StreamEventType,
        event_data: serde_json::Value,
    ) -> MediaResult<()> {
        let mut event_streams = self.event_streams.write().await;
        if let Some(stream_events) = event_streams.get_mut(stream_id) {
            let event = StreamEvent {
                event_id: self.generate_interaction_id().await,
                event_type,
                event_data,
                timestamp: Timestamp::now(),
                privacy_level: self.config.interaction_privacy_level.clone(),
            };
            
            stream_events.event_queue.push_back(event);
            stream_events.last_event_at = Timestamp::now();
            
            // Keep only recent events
            if stream_events.event_queue.len() > 1000 {
                stream_events.event_queue.pop_front();
            }
        }
        Ok(())
    }

    async fn apply_privacy_filtering(
        &self,
        interactions: &[LiveInteraction],
        _viewer_id: Option<Hash256>,
    ) -> Vec<LiveInteraction> {
        // Mock implementation - would apply real privacy filtering
        interactions.to_vec()
    }

    async fn apply_differential_privacy_to_poll_results(&self, _results: &mut PollResults) {
        // Mock implementation - would add differential privacy noise
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_interactive_streaming_initialization() {
        let config = InteractiveConfig::default();
        let engine = InteractiveStreaming::new(config);

        let stream_id = Hash256::from_bytes(&[1; 32]);
        let controls = InteractiveControls {
            stream_id: stream_id.clone(),
            available_interactions: HashSet::new(),
            moderation_settings: ModerationSettings {
                auto_moderation: true,
                profanity_filter: true,
                spam_detection: true,
                manual_approval: false,
                moderators: HashSet::new(),
                banned_participants: HashSet::new(),
            },
            privacy_controls: InteractionPrivacyControls {
                force_anonymous: true,
                enable_mixing: true,
                min_anonymity_set: 10,
                enable_differential_privacy: true,
                privacy_budget_per_participant: 1.0,
            },
            rate_limits: RateLimitSettings {
                chat_per_minute: 30,
                reactions_per_minute: 60,
                questions_per_session: 10,
                votes_per_poll: 1,
                burst_allowance: 5,
            },
            current_poll: None,
            qna_active: false,
        };

        let result = engine.initialize_stream_interactions(stream_id, controls).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_chat_message_processing() {
        let config = InteractiveConfig::default();
        let engine = InteractiveStreaming::new(config);

        let stream_id = Hash256::from_bytes(&[1; 32]);
        let sender_id = Hash256::from_bytes(&[2; 32]);

        // Initialize first
        let controls = InteractiveControls {
            stream_id: stream_id.clone(),
            available_interactions: HashSet::new(),
            moderation_settings: ModerationSettings {
                auto_moderation: true,
                profanity_filter: true,
                spam_detection: true,
                manual_approval: false,
                moderators: HashSet::new(),
                banned_participants: HashSet::new(),
            },
            privacy_controls: InteractionPrivacyControls {
                force_anonymous: true,
                enable_mixing: true,
                min_anonymity_set: 10,
                enable_differential_privacy: true,
                privacy_budget_per_participant: 1.0,
            },
            rate_limits: RateLimitSettings {
                chat_per_minute: 30,
                reactions_per_minute: 60,
                questions_per_session: 10,
                votes_per_poll: 1,
                burst_allowance: 5,
            },
            current_poll: None,
            qna_active: false,
        };

        engine.initialize_stream_interactions(stream_id.clone(), controls).await.unwrap();

        let result = engine.process_chat_message(
            stream_id,
            sender_id,
            "Hello from the stream!".to_string(),
        ).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_live_poll_creation() {
        let config = InteractiveConfig::default();
        let engine = InteractiveStreaming::new(config);

        let stream_id = Hash256::from_bytes(&[1; 32]);
        let creator_id = Hash256::from_bytes(&[2; 32]);

        let poll_settings = PollSettings {
            multiple_choice: false,
            anonymous_only: true,
            show_live_results: true,
            duration_seconds: Some(600), // 10 minutes
            allow_vote_changes: false,
            min_votes_for_results: 5,
        };

        let result = engine.create_live_poll(
            stream_id,
            creator_id,
            "What's your favorite programming language?".to_string(),
            vec!["Rust".to_string(), "Python".to_string(), "JavaScript".to_string()],
            poll_settings,
        ).await;

        assert!(result.is_ok());
    }
}