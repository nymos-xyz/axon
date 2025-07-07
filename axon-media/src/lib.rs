//! # Axon Media & Streaming Engine
//! 
//! Privacy-preserving media streaming and processing system for the Axon social network.
//! Provides anonymous live streaming, video/audio processing, and interactive media features
//! while maintaining complete user privacy and anonymity.

pub mod error;
pub mod streaming;
pub mod processing;
pub mod transcoding;
pub mod recording;
pub mod interactive;
pub mod privacy_streaming;
pub mod media_analytics;
pub mod advanced_content;

pub use error::{MediaError, MediaResult};
pub use streaming::{
    StreamingServer, StreamingClient, StreamConfig, StreamMetadata,
    StreamQuality, StreamProtocol, LiveStream
};
pub use processing::{
    MediaProcessor, ProcessingJob, MediaFormat, ProcessingConfig,
    MediaMetadata, ProcessingPipeline
};
pub use transcoding::{
    TranscodingEngine, TranscodingProfile, VideoCodec, AudioCodec,
    TranscodingJob, QualitySettings
};
pub use recording::{
    MediaRecorder, RecordingSession, RecordingConfig, PlaybackService,
    RecordingMetadata, ArchiveManager
};
pub use interactive::{
    InteractiveStreaming, LiveInteraction, ViewerParticipation,
    StreamingEvents, InteractiveControls
};
pub use privacy_streaming::{
    PrivacyStreamingEngine, AnonymousViewer, StreamPrivacyLevel,
    PrivacyControls, AnonymousInteraction
};
pub use media_analytics::{
    MediaAnalytics, StreamingMetrics, ViewerAnalytics,
    ContentPerformance, PrivacyMetrics as MediaPrivacyMetrics
};
pub use advanced_content::{
    AdvancedContentEngine, AdvancedContentConfig, RichMediaType, RichMediaContent,
    InteractiveContentType, InteractiveContent, ScheduledContent, CompositionProject,
    ContentPrivacySettings, SchedulingOptions, ComponentPrivacySettings
};

/// Media protocol version
pub const MEDIA_PROTOCOL_VERSION: u32 = 1;

/// Maximum stream bitrate (bits per second)
pub const MAX_STREAM_BITRATE: u64 = 50_000_000; // 50 Mbps

/// Default streaming chunk size
pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024; // 64KB

/// Privacy budget for media analytics
pub const MEDIA_PRIVACY_BUDGET: f64 = 5.0;