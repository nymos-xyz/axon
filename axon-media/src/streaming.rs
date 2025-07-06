//! Core Streaming Infrastructure
//! 
//! This module provides the fundamental streaming server and client implementations
//! for live video/audio streaming with privacy preservation and anonymous viewing.

use crate::error::{MediaError, MediaResult};
use nym_core::NymIdentity;
use nym_crypto::Hash256;
use quid_core::QuIDIdentity;

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, debug, warn, error};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stream configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Stream protocol (RTMP, WebRTC, etc.)
    pub protocol: StreamProtocol,
    /// Maximum bitrate (bits per second)
    pub max_bitrate: u64,
    /// Stream quality levels
    pub quality_levels: Vec<StreamQuality>,
    /// Enable adaptive bitrate
    pub adaptive_bitrate: bool,
    /// Stream timeout (seconds)
    pub timeout_seconds: u64,
    /// Enable stream recording
    pub enable_recording: bool,
    /// Privacy settings
    pub privacy_mode: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            protocol: StreamProtocol::WebRTC,
            max_bitrate: crate::MAX_STREAM_BITRATE,
            quality_levels: vec![
                StreamQuality::Low,
                StreamQuality::Medium,
                StreamQuality::High,
            ],
            adaptive_bitrate: true,
            timeout_seconds: 3600, // 1 hour
            enable_recording: false,
            privacy_mode: true,
        }
    }
}

/// Streaming protocols
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StreamProtocol {
    /// Real-Time Messaging Protocol
    RTMP,
    /// Web Real-Time Communication
    WebRTC,
    /// HTTP Live Streaming
    HLS,
    /// Dynamic Adaptive Streaming over HTTP
    DASH,
    /// Secure Real-time Transport Protocol
    SRTP,
}

/// Stream quality levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StreamQuality {
    Low,      // 480p, 1 Mbps
    Medium,   // 720p, 3 Mbps
    High,     // 1080p, 6 Mbps
    Ultra,    // 4K, 25 Mbps
}

impl StreamQuality {
    pub fn bitrate(&self) -> u64 {
        match self {
            StreamQuality::Low => 1_000_000,
            StreamQuality::Medium => 3_000_000,
            StreamQuality::High => 6_000_000,
            StreamQuality::Ultra => 25_000_000,
        }
    }

    pub fn resolution(&self) -> (u32, u32) {
        match self {
            StreamQuality::Low => (854, 480),
            StreamQuality::Medium => (1280, 720),
            StreamQuality::High => (1920, 1080),
            StreamQuality::Ultra => (3840, 2160),
        }
    }
}

/// Stream metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetadata {
    /// Stream identifier
    pub stream_id: Hash256,
    /// Stream title (may be encrypted)
    pub title: String,
    /// Stream description
    pub description: Option<String>,
    /// Content categories
    pub categories: Vec<String>,
    /// Stream language
    pub language: Option<String>,
    /// Creator identity (anonymous or pseudonymous)
    pub creator_id: Option<Hash256>,
    /// Stream start time
    pub started_at: SystemTime,
    /// Current viewer count (obfuscated for privacy)
    pub viewer_count: u64,
    /// Stream duration
    pub duration: Option<Duration>,
    /// Stream status
    pub status: StreamStatus,
}

/// Stream status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StreamStatus {
    /// Stream is starting up
    Starting,
    /// Stream is live
    Live,
    /// Stream is paused
    Paused,
    /// Stream has ended
    Ended,
    /// Stream encountered an error
    Error(String),
}

/// Live stream representation
#[derive(Debug, Clone)]
pub struct LiveStream {
    /// Stream metadata
    pub metadata: StreamMetadata,
    /// Stream configuration
    pub config: StreamConfig,
    /// Connected viewers
    pub viewers: HashMap<Hash256, ViewerConnection>,
    /// Stream statistics
    pub statistics: StreamStatistics,
    /// Privacy settings
    pub privacy_settings: StreamPrivacySettings,
}

/// Viewer connection information
#[derive(Debug, Clone)]
pub struct ViewerConnection {
    /// Viewer identifier (anonymous)
    pub viewer_id: Hash256,
    /// Connection timestamp
    pub connected_at: SystemTime,
    /// Current quality level
    pub quality_level: StreamQuality,
    /// Viewer location (obfuscated)
    pub region: Option<String>,
    /// Connection latency
    pub latency_ms: u64,
}

/// Stream statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStatistics {
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Current bitrate
    pub current_bitrate: u64,
    /// Frame rate
    pub framerate: f32,
    /// Dropped frames
    pub dropped_frames: u64,
    /// Network quality score (0-100)
    pub network_quality: u8,
    /// Buffer health
    pub buffer_health: BufferHealth,
}

/// Buffer health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferHealth {
    /// Current buffer size (milliseconds)
    pub buffer_size_ms: u64,
    /// Buffer underruns
    pub underruns: u64,
    /// Buffer overruns
    pub overruns: u64,
    /// Target buffer size
    pub target_buffer_ms: u64,
}

/// Stream privacy settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamPrivacySettings {
    /// Enable anonymous viewing
    pub anonymous_viewing: bool,
    /// Hide viewer count
    pub hide_viewer_count: bool,
    /// Enable viewer mixing
    pub enable_viewer_mixing: bool,
    /// Minimum anonymity set size
    pub min_anonymity_set: usize,
    /// Enable geographic obfuscation
    pub geographic_obfuscation: bool,
}

/// Streaming server
pub struct StreamingServer {
    /// Server configuration
    config: StreamingServerConfig,
    /// Active streams
    active_streams: RwLock<HashMap<Hash256, LiveStream>>,
    /// Stream endpoints
    endpoints: RwLock<HashMap<StreamProtocol, String>>,
    /// Server statistics
    statistics: RwLock<ServerStatistics>,
    /// Privacy mixer
    privacy_mixer: RwLock<PrivacyMixer>,
}

/// Streaming server configuration
#[derive(Debug, Clone)]
pub struct StreamingServerConfig {
    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,
    /// Maximum viewers per stream
    pub max_viewers_per_stream: usize,
    /// Supported protocols
    pub supported_protocols: Vec<StreamProtocol>,
    /// Enable privacy features
    pub enable_privacy_features: bool,
    /// Stream inactivity timeout
    pub inactivity_timeout: Duration,
    /// Enable adaptive bitrate
    pub enable_adaptive_bitrate: bool,
}

impl Default for StreamingServerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 10000,
            max_viewers_per_stream: 100000,
            supported_protocols: vec![
                StreamProtocol::WebRTC,
                StreamProtocol::HLS,
                StreamProtocol::DASH,
            ],
            enable_privacy_features: true,
            inactivity_timeout: Duration::from_secs(300),
            enable_adaptive_bitrate: true,
        }
    }
}

/// Server statistics
#[derive(Debug, Clone)]
pub struct ServerStatistics {
    /// Total active streams
    pub active_streams: usize,
    /// Total connected viewers
    pub total_viewers: usize,
    /// Total bandwidth usage (bytes/second)
    pub bandwidth_usage: u64,
    /// Server uptime
    pub uptime: Duration,
    /// Privacy mixer statistics
    pub privacy_statistics: PrivacyStatistics,
}

/// Privacy mixer for viewer anonymity
#[derive(Debug, Clone)]
pub struct PrivacyMixer {
    /// Mixing pools by stream
    pub mixing_pools: HashMap<Hash256, MixingPool>,
    /// Global anonymity set
    pub global_anonymity_set: HashSet<Hash256>,
    /// Mixing parameters
    pub mixing_parameters: MixingParameters,
}

/// Mixing pool for a stream
#[derive(Debug, Clone)]
pub struct MixingPool {
    /// Pool identifier
    pub pool_id: Hash256,
    /// Anonymous viewer identifiers
    pub viewer_ids: HashSet<Hash256>,
    /// Pool creation time
    pub created_at: SystemTime,
    /// Mixing round interval
    pub mixing_interval: Duration,
}

/// Mixing parameters
#[derive(Debug, Clone)]
pub struct MixingParameters {
    /// Minimum pool size for anonymity
    pub min_pool_size: usize,
    /// Mixing round duration
    pub mixing_round_duration: Duration,
    /// Enable temporal mixing
    pub enable_temporal_mixing: bool,
    /// Geographic mixing enabled
    pub enable_geographic_mixing: bool,
}

/// Privacy statistics
#[derive(Debug, Clone)]
pub struct PrivacyStatistics {
    /// Total anonymous viewers
    pub anonymous_viewers: usize,
    /// Average anonymity set size
    pub average_anonymity_set_size: f64,
    /// Privacy violations detected
    pub privacy_violations: u64,
    /// Mixing rounds completed
    pub mixing_rounds_completed: u64,
}

/// Streaming client
pub struct StreamingClient {
    /// Client configuration
    config: StreamingClientConfig,
    /// Current stream connection
    current_stream: RwLock<Option<StreamConnection>>,
    /// Client statistics
    statistics: RwLock<ClientStatistics>,
    /// Privacy settings
    privacy_settings: RwLock<ClientPrivacySettings>,
}

/// Streaming client configuration
#[derive(Debug, Clone)]
pub struct StreamingClientConfig {
    /// Preferred protocol
    pub preferred_protocol: StreamProtocol,
    /// Preferred quality
    pub preferred_quality: StreamQuality,
    /// Enable adaptive quality
    pub enable_adaptive_quality: bool,
    /// Buffer size (milliseconds)
    pub buffer_size_ms: u64,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Enable privacy mode
    pub enable_privacy_mode: bool,
}

/// Stream connection
#[derive(Debug, Clone)]
pub struct StreamConnection {
    /// Connection ID
    pub connection_id: Hash256,
    /// Stream metadata
    pub stream_metadata: StreamMetadata,
    /// Current quality
    pub current_quality: StreamQuality,
    /// Connection timestamp
    pub connected_at: SystemTime,
    /// Protocol used
    pub protocol: StreamProtocol,
    /// Connection statistics
    pub statistics: ConnectionStatistics,
}

/// Connection statistics
#[derive(Debug, Clone)]
pub struct ConnectionStatistics {
    /// Bytes received
    pub bytes_received: u64,
    /// Current latency
    pub latency_ms: u64,
    /// Buffer level
    pub buffer_level_ms: u64,
    /// Quality switches
    pub quality_switches: u32,
    /// Connection quality score
    pub quality_score: u8,
}

/// Client statistics
#[derive(Debug, Clone)]
pub struct ClientStatistics {
    /// Total viewing time
    pub total_viewing_time: Duration,
    /// Streams watched
    pub streams_watched: u32,
    /// Data consumed
    pub data_consumed: u64,
    /// Average quality watched
    pub average_quality: f32,
    /// Privacy metrics
    pub privacy_metrics: ClientPrivacyMetrics,
}

/// Client privacy settings
#[derive(Debug, Clone)]
pub struct ClientPrivacySettings {
    /// Anonymous viewing mode
    pub anonymous_mode: bool,
    /// Viewer mixing enabled
    pub viewer_mixing: bool,
    /// Location obfuscation
    pub location_obfuscation: bool,
    /// Disable viewer analytics
    pub disable_analytics: bool,
}

/// Client privacy metrics
#[derive(Debug, Clone)]
pub struct ClientPrivacyMetrics {
    /// Anonymous sessions
    pub anonymous_sessions: u32,
    /// Privacy level maintained
    pub privacy_level: f32,
    /// Tracking attempts blocked
    pub tracking_blocked: u32,
}

use std::collections::HashSet;

impl StreamingServer {
    pub fn new(config: StreamingServerConfig) -> Self {
        info!("Initializing streaming server with {} protocol support", 
              config.supported_protocols.len());

        Self {
            config,
            active_streams: RwLock::new(HashMap::new()),
            endpoints: RwLock::new(HashMap::new()),
            statistics: RwLock::new(ServerStatistics {
                active_streams: 0,
                total_viewers: 0,
                bandwidth_usage: 0,
                uptime: Duration::from_secs(0),
                privacy_statistics: PrivacyStatistics {
                    anonymous_viewers: 0,
                    average_anonymity_set_size: 0.0,
                    privacy_violations: 0,
                    mixing_rounds_completed: 0,
                },
            }),
            privacy_mixer: RwLock::new(PrivacyMixer {
                mixing_pools: HashMap::new(),
                global_anonymity_set: HashSet::new(),
                mixing_parameters: MixingParameters {
                    min_pool_size: 10,
                    mixing_round_duration: Duration::from_secs(60),
                    enable_temporal_mixing: true,
                    enable_geographic_mixing: true,
                },
            }),
        }
    }

    /// Create a new live stream
    pub async fn create_stream(
        &self,
        creator_id: Option<QuIDIdentity>,
        metadata: StreamMetadata,
        config: StreamConfig,
    ) -> MediaResult<Hash256> {
        info!("Creating new stream: {}", metadata.title);

        // Check capacity
        let active_streams = self.active_streams.read().await;
        if active_streams.len() >= self.config.max_concurrent_streams {
            return Err(MediaError::CapacityExceeded {
                current: active_streams.len(),
                max: self.config.max_concurrent_streams,
            });
        }
        drop(active_streams);

        // Create stream
        let stream = LiveStream {
            metadata: metadata.clone(),
            config,
            viewers: HashMap::new(),
            statistics: StreamStatistics {
                bytes_sent: 0,
                bytes_received: 0,
                current_bitrate: 0,
                framerate: 0.0,
                dropped_frames: 0,
                network_quality: 100,
                buffer_health: BufferHealth {
                    buffer_size_ms: 0,
                    underruns: 0,
                    overruns: 0,
                    target_buffer_ms: 3000,
                },
            },
            privacy_settings: StreamPrivacySettings {
                anonymous_viewing: true,
                hide_viewer_count: false,
                enable_viewer_mixing: true,
                min_anonymity_set: 10,
                geographic_obfuscation: true,
            },
        };

        // Create privacy mixing pool
        if self.config.enable_privacy_features {
            self.create_mixing_pool_for_stream(&metadata.stream_id).await?;
        }

        // Add to active streams
        let mut active_streams = self.active_streams.write().await;
        active_streams.insert(metadata.stream_id.clone(), stream);

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.active_streams = active_streams.len();

        info!("Stream created successfully: {}", hex::encode(metadata.stream_id.as_bytes()));
        Ok(metadata.stream_id)
    }

    /// Add viewer to stream
    pub async fn add_viewer(
        &self,
        stream_id: Hash256,
        viewer_id: Hash256,
        quality_preference: Option<StreamQuality>,
    ) -> MediaResult<ViewerConnection> {
        debug!("Adding viewer to stream: {}", hex::encode(stream_id.as_bytes()));

        let mut active_streams = self.active_streams.write().await;
        let stream = active_streams.get_mut(&stream_id)
            .ok_or_else(|| MediaError::StreamNotFound {
                stream_id: hex::encode(stream_id.as_bytes()),
            })?;

        // Check viewer capacity
        if stream.viewers.len() >= self.config.max_viewers_per_stream {
            return Err(MediaError::CapacityExceeded {
                current: stream.viewers.len(),
                max: self.config.max_viewers_per_stream,
            });
        }

        // Determine quality level
        let quality_level = quality_preference
            .unwrap_or_else(|| {
                // Default to highest available quality
                stream.config.quality_levels.iter()
                    .max_by_key(|q| q.bitrate())
                    .cloned()
                    .unwrap_or(StreamQuality::Medium)
            });

        // Create viewer connection
        let connection = ViewerConnection {
            viewer_id: viewer_id.clone(),
            connected_at: SystemTime::now(),
            quality_level,
            region: None, // Obfuscated for privacy
            latency_ms: 0,
        };

        // Add to stream
        stream.viewers.insert(viewer_id.clone(), connection.clone());

        // Update privacy mixer
        if self.config.enable_privacy_features {
            self.add_viewer_to_mixing_pool(&stream_id, &viewer_id).await?;
        }

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_viewers = active_streams.values()
            .map(|s| s.viewers.len())
            .sum();

        Ok(connection)
    }

    /// Remove viewer from stream
    pub async fn remove_viewer(
        &self,
        stream_id: Hash256,
        viewer_id: Hash256,
    ) -> MediaResult<()> {
        debug!("Removing viewer from stream");

        let mut active_streams = self.active_streams.write().await;
        let stream = active_streams.get_mut(&stream_id)
            .ok_or_else(|| MediaError::StreamNotFound {
                stream_id: hex::encode(stream_id.as_bytes()),
            })?;

        // Remove viewer
        stream.viewers.remove(&viewer_id);

        // Update privacy mixer
        if self.config.enable_privacy_features {
            self.remove_viewer_from_mixing_pool(&stream_id, &viewer_id).await?;
        }

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_viewers = active_streams.values()
            .map(|s| s.viewers.len())
            .sum();

        Ok(())
    }

    /// Get stream metadata with privacy protection
    pub async fn get_stream_metadata(
        &self,
        stream_id: Hash256,
        viewer_id: Option<Hash256>,
    ) -> MediaResult<StreamMetadata> {
        let active_streams = self.active_streams.read().await;
        let stream = active_streams.get(&stream_id)
            .ok_or_else(|| MediaError::StreamNotFound {
                stream_id: hex::encode(stream_id.as_bytes()),
            })?;

        let mut metadata = stream.metadata.clone();

        // Apply privacy protection
        if stream.privacy_settings.hide_viewer_count {
            metadata.viewer_count = self.obfuscate_viewer_count(metadata.viewer_count).await;
        }

        Ok(metadata)
    }

    /// Get server statistics
    pub async fn get_server_statistics(&self) -> ServerStatistics {
        self.statistics.read().await.clone()
    }

    // Helper methods for privacy mixing

    async fn create_mixing_pool_for_stream(&self, stream_id: &Hash256) -> MediaResult<()> {
        let pool_id = Hash256::from_bytes(&sha3::Sha3_256::digest(
            format!("mixing_pool_{}", hex::encode(stream_id.as_bytes())).as_bytes()
        ).into());

        let pool = MixingPool {
            pool_id,
            viewer_ids: HashSet::new(),
            created_at: SystemTime::now(),
            mixing_interval: Duration::from_secs(60),
        };

        let mut mixer = self.privacy_mixer.write().await;
        mixer.mixing_pools.insert(stream_id.clone(), pool);

        Ok(())
    }

    async fn add_viewer_to_mixing_pool(&self, stream_id: &Hash256, viewer_id: &Hash256) -> MediaResult<()> {
        let mut mixer = self.privacy_mixer.write().await;
        if let Some(pool) = mixer.mixing_pools.get_mut(stream_id) {
            pool.viewer_ids.insert(viewer_id.clone());
            mixer.global_anonymity_set.insert(viewer_id.clone());
        }
        Ok(())
    }

    async fn remove_viewer_from_mixing_pool(&self, stream_id: &Hash256, viewer_id: &Hash256) -> MediaResult<()> {
        let mut mixer = self.privacy_mixer.write().await;
        if let Some(pool) = mixer.mixing_pools.get_mut(stream_id) {
            pool.viewer_ids.remove(viewer_id);
        }
        Ok(())
    }

    async fn obfuscate_viewer_count(&self, actual_count: u64) -> u64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // Add noise to viewer count for privacy
        let noise = rng.gen_range(-5..=5);
        (actual_count as i64 + noise).max(0) as u64
    }
}

impl StreamingClient {
    pub fn new(config: StreamingClientConfig) -> Self {
        info!("Initializing streaming client with {:?} protocol preference", config.preferred_protocol);

        Self {
            config,
            current_stream: RwLock::new(None),
            statistics: RwLock::new(ClientStatistics {
                total_viewing_time: Duration::from_secs(0),
                streams_watched: 0,
                data_consumed: 0,
                average_quality: 0.0,
                privacy_metrics: ClientPrivacyMetrics {
                    anonymous_sessions: 0,
                    privacy_level: 1.0,
                    tracking_blocked: 0,
                },
            }),
            privacy_settings: RwLock::new(ClientPrivacySettings {
                anonymous_mode: true,
                viewer_mixing: true,
                location_obfuscation: true,
                disable_analytics: true,
            }),
        }
    }

    /// Connect to a stream
    pub async fn connect_to_stream(
        &self,
        stream_id: Hash256,
        server_endpoint: String,
    ) -> MediaResult<()> {
        info!("Connecting to stream: {}", hex::encode(stream_id.as_bytes()));

        // Create connection
        let connection = StreamConnection {
            connection_id: Hash256::from_bytes(&sha3::Sha3_256::digest(
                format!("connection_{}_{}", 
                       hex::encode(stream_id.as_bytes()),
                       SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()
                ).as_bytes()
            ).into()),
            stream_metadata: StreamMetadata {
                stream_id: stream_id.clone(),
                title: "Loading...".to_string(),
                description: None,
                categories: Vec::new(),
                language: None,
                creator_id: None,
                started_at: SystemTime::now(),
                viewer_count: 0,
                duration: None,
                status: StreamStatus::Starting,
            },
            current_quality: self.config.preferred_quality.clone(),
            connected_at: SystemTime::now(),
            protocol: self.config.preferred_protocol.clone(),
            statistics: ConnectionStatistics {
                bytes_received: 0,
                latency_ms: 0,
                buffer_level_ms: 0,
                quality_switches: 0,
                quality_score: 100,
            },
        };

        // Store connection
        let mut current_stream = self.current_stream.write().await;
        *current_stream = Some(connection);

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.streams_watched += 1;

        // Update privacy metrics if in anonymous mode
        let privacy_settings = self.privacy_settings.read().await;
        if privacy_settings.anonymous_mode {
            stats.privacy_metrics.anonymous_sessions += 1;
        }

        Ok(())
    }

    /// Disconnect from current stream
    pub async fn disconnect(&self) -> MediaResult<()> {
        info!("Disconnecting from stream");

        let mut current_stream = self.current_stream.write().await;
        if let Some(connection) = current_stream.take() {
            // Update statistics
            let viewing_duration = SystemTime::now()
                .duration_since(connection.connected_at)
                .unwrap_or(Duration::from_secs(0));

            let mut stats = self.statistics.write().await;
            stats.total_viewing_time += viewing_duration;
            stats.data_consumed += connection.statistics.bytes_received;
        }

        Ok(())
    }

    /// Get current connection status
    pub async fn get_connection_status(&self) -> Option<StreamConnection> {
        self.current_stream.read().await.clone()
    }

    /// Get client statistics
    pub async fn get_client_statistics(&self) -> ClientStatistics {
        self.statistics.read().await.clone()
    }

    /// Update privacy settings
    pub async fn update_privacy_settings(&self, settings: ClientPrivacySettings) {
        let mut privacy_settings = self.privacy_settings.write().await;
        *privacy_settings = settings;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_streaming_server_creation() {
        let config = StreamingServerConfig::default();
        let server = StreamingServer::new(config);

        let metadata = StreamMetadata {
            stream_id: Hash256::from_bytes(&[1; 32]),
            title: "Test Stream".to_string(),
            description: Some("Test stream description".to_string()),
            categories: vec!["Technology".to_string()],
            language: Some("en".to_string()),
            creator_id: Some(Hash256::from_bytes(&[2; 32])),
            started_at: SystemTime::now(),
            viewer_count: 0,
            duration: None,
            status: StreamStatus::Starting,
        };

        let stream_config = StreamConfig::default();
        let result = server.create_stream(None, metadata, stream_config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_viewer_connection() {
        let config = StreamingServerConfig::default();
        let server = StreamingServer::new(config);

        // Create stream first
        let stream_id = Hash256::from_bytes(&[1; 32]);
        let metadata = StreamMetadata {
            stream_id: stream_id.clone(),
            title: "Test Stream".to_string(),
            description: None,
            categories: Vec::new(),
            language: None,
            creator_id: None,
            started_at: SystemTime::now(),
            viewer_count: 0,
            duration: None,
            status: StreamStatus::Live,
        };

        let stream_config = StreamConfig::default();
        server.create_stream(None, metadata, stream_config).await.unwrap();

        // Add viewer
        let viewer_id = Hash256::from_bytes(&[2; 32]);
        let connection = server.add_viewer(
            stream_id,
            viewer_id,
            Some(StreamQuality::High),
        ).await.unwrap();

        assert_eq!(connection.quality_level, StreamQuality::High);
    }

    #[tokio::test]
    async fn test_streaming_client() {
        let config = StreamingClientConfig {
            preferred_protocol: StreamProtocol::WebRTC,
            preferred_quality: StreamQuality::Medium,
            enable_adaptive_quality: true,
            buffer_size_ms: 3000,
            connection_timeout: Duration::from_secs(30),
            enable_privacy_mode: true,
        };

        let client = StreamingClient::new(config);
        let stream_id = Hash256::from_bytes(&[1; 32]);

        let result = client.connect_to_stream(
            stream_id,
            "wss://streaming.example.com/stream".to_string(),
        ).await;

        assert!(result.is_ok());

        let status = client.get_connection_status().await;
        assert!(status.is_some());
    }
}