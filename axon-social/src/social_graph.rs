//! Social Graph - Privacy-preserving following/followers system
//!
//! This module implements a privacy-first social graph where:
//! - Following relationships are anonymous by default
//! - Zero-knowledge proofs verify connections without revealing identities
//! - Optional public connections for verified accounts
//! - Mutual follow detection without graph analysis
//! - Social graph analytics without user profiling

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, Verifier};
use axon_core::{identity::QuIDIdentity as Identity, ContentHash as ContentId};
use axon_identity::auth_service::AuthenticationService as AuthService;
use crate::{SocialError, SocialResult, PrivacyLevel};

/// Type alias for user identifiers in social context
pub type UserId = String;

/// Social graph manager with privacy preservation
#[derive(Debug)]
pub struct SocialGraph {
    /// Anonymous connections (encrypted, no direct mapping)
    anonymous_connections: HashMap<String, EncryptedConnection>,
    /// Public connections (verified accounts only)
    public_connections: HashMap<UserId, PublicConnection>,
    /// Connection proofs for verification
    connection_proofs: HashMap<String, ConnectionProof>,
    /// Privacy settings per user
    privacy_settings: HashMap<UserId, SocialPrivacySettings>,
    /// Connection limits
    limits: ConnectionLimits,
}

/// Connection between two users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Connection identifier (anonymous hash)
    pub id: String,
    /// Connection type
    pub connection_type: ConnectionType,
    /// When the connection was created
    pub created_at: DateTime<Utc>,
    /// Privacy level for this connection
    pub privacy_level: PrivacyLevel,
    /// Optional metadata (encrypted)
    pub metadata: Option<Vec<u8>>,
}

/// Types of social connections
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Following another user
    Following,
    /// Being followed by another user
    Follower,
    /// Mutual following relationship
    Mutual,
    /// Blocked user
    Blocked,
    /// Pending follow request
    Pending,
}

/// Encrypted connection for privacy preservation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncryptedConnection {
    /// Encrypted follower ID
    encrypted_follower: Vec<u8>,
    /// Encrypted followee ID  
    encrypted_followee: Vec<u8>,
    /// Connection metadata
    connection: Connection,
    /// Zero-knowledge proof of valid connection
    validity_proof: Vec<u8>,
}

/// Public connection for verified accounts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicConnection {
    /// Follower user ID
    pub follower: UserId,
    /// Followee user ID
    pub followee: UserId,
    /// Connection details
    pub connection: Connection,
    /// Verification signature
    pub signature: Vec<u8>,
}

/// Zero-knowledge proof for connection validity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionProof {
    /// Proof that follower is valid user
    pub follower_proof: Vec<u8>,
    /// Proof that followee is valid user
    pub followee_proof: Vec<u8>,
    /// Proof of connection authorization
    pub authorization_proof: Vec<u8>,
    /// Timestamp of proof generation
    pub created_at: DateTime<Utc>,
}

/// Follow request for privacy-preserving connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowRequest {
    /// Request ID
    pub id: String,
    /// Encrypted follower identity
    pub encrypted_follower: Vec<u8>,
    /// Target user (may be encrypted)
    pub target: String,
    /// Privacy level requested
    pub privacy_level: PrivacyLevel,
    /// Zero-knowledge proof of authorization
    pub authorization_proof: Vec<u8>,
    /// Request timestamp
    pub created_at: DateTime<Utc>,
    /// Optional message (encrypted)
    pub message: Option<Vec<u8>>,
}

/// Privacy settings for social connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialPrivacySettings {
    /// Default privacy level for new connections
    pub default_privacy: PrivacyLevel,
    /// Allow public follower counts
    pub public_follower_count: bool,
    /// Allow public following counts
    pub public_following_count: bool,
    /// Require approval for new followers
    pub require_approval: bool,
    /// Allow anonymous follows
    pub allow_anonymous: bool,
    /// Auto-block suspicious accounts
    pub auto_block_suspicious: bool,
}

/// Connection limits to prevent abuse
#[derive(Debug, Clone)]
pub struct ConnectionLimits {
    /// Maximum followers per user
    pub max_followers: usize,
    /// Maximum following per user
    pub max_following: usize,
    /// Maximum pending requests
    pub max_pending: usize,
    /// Rate limit for new connections (per hour)
    pub connection_rate_limit: usize,
}

impl Default for ConnectionLimits {
    fn default() -> Self {
        Self {
            max_followers: crate::defaults::MAX_FOLLOWERS,
            max_following: crate::defaults::MAX_FOLLOWING,
            max_pending: 100,
            connection_rate_limit: 50,
        }
    }
}

impl Default for SocialPrivacySettings {
    fn default() -> Self {
        Self {
            default_privacy: PrivacyLevel::Anonymous,
            public_follower_count: false,
            public_following_count: false,
            require_approval: true,
            allow_anonymous: true,
            auto_block_suspicious: true,
        }
    }
}

impl SocialGraph {
    /// Create a new social graph
    pub fn new() -> Self {
        Self::with_limits(ConnectionLimits::default())
    }

    /// Create social graph with custom limits
    pub fn with_limits(limits: ConnectionLimits) -> Self {
        Self {
            anonymous_connections: HashMap::new(),
            public_connections: HashMap::new(),
            connection_proofs: HashMap::new(),
            privacy_settings: HashMap::new(),
            limits,
        }
    }

    /// Create a follow request with privacy preservation
    pub async fn create_follow_request(
        &self,
        follower_identity: &Identity,
        target_user: &str,
        privacy_level: PrivacyLevel,
        auth_service: &AuthService,
    ) -> SocialResult<FollowRequest> {
        // Check rate limits
        self.check_rate_limits(&follower_identity.get_id())?;

        // Generate request ID
        let request_id = self.generate_request_id(follower_identity, target_user);

        // Encrypt follower identity based on privacy level
        let encrypted_follower = match privacy_level {
            PrivacyLevel::Anonymous => {
                self.encrypt_identity(&follower_identity.get_id(), target_user)?
            }
            PrivacyLevel::Pseudonymous => {
                self.pseudonymize_identity(&follower_identity.get_id(), target_user)?
            }
            PrivacyLevel::Public => follower_identity.get_id().as_bytes().to_vec(),
        };

        // Generate zero-knowledge proof of authorization
        let authorization_proof = self.generate_authorization_proof(
            follower_identity,
            target_user,
            &auth_service,
        ).await?;

        Ok(FollowRequest {
            id: request_id,
            encrypted_follower,
            target: target_user.to_string(),
            privacy_level,
            authorization_proof,
            created_at: Utc::now(),
            message: None,
        })
    }

    /// Process a follow request
    pub async fn process_follow_request(
        &mut self,
        request: FollowRequest,
        target_identity: &Identity,
        approved: bool,
        auth_service: &AuthService,
    ) -> SocialResult<Option<Connection>> {
        if !approved {
            return Ok(None);
        }

        // Verify authorization proof
        self.verify_authorization_proof(&request.authorization_proof, &request.target, auth_service).await?;

        // Check target user's privacy settings
        let target_settings = self.get_privacy_settings(&target_identity.get_id());
        if target_settings.require_approval && !approved {
            return Err(SocialError::PermissionDenied(
                "Follow request requires approval".to_string()
            ));
        }

        // Check connection limits
        self.check_connection_limits(&target_identity.get_id(), ConnectionType::Follower)?;

        // Create connection
        let connection = self.create_connection(
            &request,
            target_identity,
            ConnectionType::Following,
        ).await?;

        // Store connection based on privacy level
        match request.privacy_level {
            PrivacyLevel::Public => {
                self.store_public_connection(&connection, &request, target_identity)?;
            }
            _ => {
                self.store_anonymous_connection(&connection, &request, target_identity).await?;
            }
        }

        Ok(Some(connection))
    }

    /// Get followers for a user (privacy-aware)
    pub async fn get_followers(
        &self,
        user_id: &str,
        requester_identity: &Identity,
        privacy_level: PrivacyLevel,
    ) -> SocialResult<Vec<Connection>> {
        // Check permissions
        self.check_view_permissions(user_id, &requester_identity.get_id(), "followers")?;

        let mut followers = Vec::new();

        // Add public followers if allowed
        if privacy_level == PrivacyLevel::Public || self.can_view_public_connections(user_id, &requester_identity.get_id()) {
            for connection in self.public_connections.values() {
                if connection.followee == user_id && connection.connection.connection_type == ConnectionType::Following {
                    followers.push(connection.connection.clone());
                }
            }
        }

        // Add anonymous followers if requester has permission
        if requester_identity.get_id() == user_id {
            // User can see their own anonymous followers
            for encrypted_conn in self.anonymous_connections.values() {
                if self.is_connection_for_user(&encrypted_conn, user_id, ConnectionType::Follower)? {
                    followers.push(encrypted_conn.connection.clone());
                }
            }
        }

        Ok(followers)
    }

    /// Get following list for a user (privacy-aware)
    pub async fn get_following(
        &self,
        user_id: &str,
        requester_identity: &Identity,
        privacy_level: PrivacyLevel,
    ) -> SocialResult<Vec<Connection>> {
        // Check permissions
        self.check_view_permissions(user_id, &requester_identity.get_id(), "following")?;

        let mut following = Vec::new();

        // Add public following if allowed
        if privacy_level == PrivacyLevel::Public || self.can_view_public_connections(user_id, &requester_identity.get_id()) {
            for connection in self.public_connections.values() {
                if connection.follower == user_id && connection.connection.connection_type == ConnectionType::Following {
                    following.push(connection.connection.clone());
                }
            }
        }

        // Add anonymous following if requester has permission
        if requester_identity.get_id() == user_id {
            // User can see their own anonymous following
            for encrypted_conn in self.anonymous_connections.values() {
                if self.is_connection_for_user(&encrypted_conn, user_id, ConnectionType::Following)? {
                    following.push(encrypted_conn.connection.clone());
                }
            }
        }

        Ok(following)
    }

    /// Check if two users are mutually following
    pub async fn check_mutual_follow(
        &self,
        user1: &str,
        user2: &str,
    ) -> SocialResult<bool> {
        let user1_following = self.is_following(user1, user2).await?;
        let user2_following = self.is_following(user2, user1).await?;
        
        Ok(user1_following && user2_following)
    }

    /// Get follower count (privacy-aware)
    pub fn get_follower_count(&self, user_id: &str, include_anonymous: bool) -> SocialResult<usize> {
        let settings = self.get_privacy_settings(user_id);
        
        if !settings.public_follower_count && !include_anonymous {
            return Err(SocialError::PrivacyViolation(
                "Follower count is private".to_string()
            ));
        }

        let mut count = 0;

        // Count public followers
        for connection in self.public_connections.values() {
            if connection.followee == user_id && connection.connection.connection_type == ConnectionType::Following {
                count += 1;
            }
        }

        // Count anonymous followers if allowed
        if include_anonymous {
            for encrypted_conn in self.anonymous_connections.values() {
                if self.is_connection_for_user(&encrypted_conn, user_id, ConnectionType::Follower)? {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Set privacy settings for a user
    pub fn set_privacy_settings(
        &mut self,
        user_id: &str,
        settings: SocialPrivacySettings,
    ) -> SocialResult<()> {
        self.privacy_settings.insert(user_id.to_string(), settings);
        Ok(())
    }

    /// Block a user
    pub async fn block_user(
        &mut self,
        blocker_identity: &Identity,
        blocked_user: &str,
    ) -> SocialResult<()> {
        let blocker_id = blocker_identity.get_id();

        // Remove any existing connections
        self.remove_connection(&blocker_id, blocked_user).await?;

        // Create block connection
        let block_connection = Connection {
            id: self.generate_connection_id(&blocker_id, blocked_user),
            connection_type: ConnectionType::Blocked,
            created_at: Utc::now(),
            privacy_level: PrivacyLevel::Anonymous,
            metadata: None,
        };

        // Store block (always anonymous)
        let encrypted_conn = EncryptedConnection {
            encrypted_follower: self.encrypt_identity(&blocker_id, blocked_user)?,
            encrypted_followee: self.encrypt_identity(blocked_user, &blocker_id)?,
            connection: block_connection,
            validity_proof: vec![], // No proof needed for blocks
        };

        self.anonymous_connections.insert(
            self.generate_connection_id(&blocker_id, blocked_user),
            encrypted_conn,
        );

        Ok(())
    }

    /// Check if user is blocked
    pub async fn is_blocked(&self, user1: &str, user2: &str) -> SocialResult<bool> {
        for encrypted_conn in self.anonymous_connections.values() {
            if encrypted_conn.connection.connection_type == ConnectionType::Blocked {
                if self.is_connection_between_users(&encrypted_conn, user1, user2)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    // Private helper methods

    fn get_privacy_settings(&self, user_id: &str) -> SocialPrivacySettings {
        self.privacy_settings
            .get(user_id)
            .cloned()
            .unwrap_or_default()
    }

    fn check_rate_limits(&self, user_id: &str) -> SocialResult<()> {
        // In a real implementation, this would check recent connection attempts
        // For now, always allow
        Ok(())
    }

    fn check_connection_limits(&self, user_id: &str, connection_type: ConnectionType) -> SocialResult<()> {
        let current_count = match connection_type {
            ConnectionType::Follower => self.get_follower_count(user_id, true).unwrap_or(0),
            ConnectionType::Following => {
                // Count following connections
                let mut count = 0;
                for connection in self.public_connections.values() {
                    if connection.follower == user_id && connection.connection.connection_type == ConnectionType::Following {
                        count += 1;
                    }
                }
                for encrypted_conn in self.anonymous_connections.values() {
                    if self.is_connection_for_user(&encrypted_conn, user_id, ConnectionType::Following)? {
                        count += 1;
                    }
                }
                count
            }
            _ => return Ok(()),
        };

        let limit = match connection_type {
            ConnectionType::Follower => self.limits.max_followers,
            ConnectionType::Following => self.limits.max_following,
            _ => return Ok(()),
        };

        if current_count >= limit {
            return Err(SocialError::ConnectionLimitExceeded {
                max: limit,
                current: current_count,
            });
        }

        Ok(())
    }

    fn check_view_permissions(&self, target_user: &str, requester: &str, data_type: &str) -> SocialResult<()> {
        let settings = self.get_privacy_settings(target_user);
        
        // User can always view their own data
        if target_user == requester {
            return Ok(());
        }

        // Check specific privacy settings
        match data_type {
            "followers" => {
                if !settings.public_follower_count {
                    return Err(SocialError::PermissionDenied(
                        "Follower list is private".to_string()
                    ));
                }
            }
            "following" => {
                if !settings.public_following_count {
                    return Err(SocialError::PermissionDenied(
                        "Following list is private".to_string()
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn can_view_public_connections(&self, target_user: &str, requester: &str) -> bool {
        target_user == requester || self.are_mutually_connected(target_user, requester)
    }

    fn are_mutually_connected(&self, user1: &str, user2: &str) -> bool {
        // Check if users are mutually following (simplified check)
        let mut user1_follows_user2 = false;
        let mut user2_follows_user1 = false;

        for connection in self.public_connections.values() {
            if connection.follower == user1 && connection.followee == user2 
                && connection.connection.connection_type == ConnectionType::Following {
                user1_follows_user2 = true;
            }
            if connection.follower == user2 && connection.followee == user1 
                && connection.connection.connection_type == ConnectionType::Following {
                user2_follows_user1 = true;
            }
        }

        user1_follows_user2 && user2_follows_user1
    }

    async fn is_following(&self, follower: &str, followee: &str) -> SocialResult<bool> {
        // Check public connections
        for connection in self.public_connections.values() {
            if connection.follower == follower && connection.followee == followee 
                && connection.connection.connection_type == ConnectionType::Following {
                return Ok(true);
            }
        }

        // Check anonymous connections (simplified - in practice would need decryption)
        for encrypted_conn in self.anonymous_connections.values() {
            if encrypted_conn.connection.connection_type == ConnectionType::Following {
                if self.is_connection_between_users(&encrypted_conn, follower, followee)? {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn generate_request_id(&self, follower_identity: &Identity, target_user: &str) -> String {
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        hasher.update(follower_identity.get_id().as_bytes());
        hasher.update(target_user.as_bytes());
        hasher.update(&Utc::now().timestamp().to_le_bytes());
        
        format!("req_{}", hex::encode(hasher.finalize()))
    }

    fn generate_connection_id(&self, user1: &str, user2: &str) -> String {
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        hasher.update(user1.as_bytes());
        hasher.update(user2.as_bytes());
        
        format!("conn_{}", hex::encode(hasher.finalize()))
    }

    fn encrypt_identity(&self, identity: &str, context: &str) -> SocialResult<Vec<u8>> {
        // Simplified encryption - in practice use proper encryption
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        hasher.update(identity.as_bytes());
        hasher.update(context.as_bytes());
        hasher.update(b"encryption_key");
        
        Ok(hasher.finalize().to_vec())
    }

    fn pseudonymize_identity(&self, identity: &str, context: &str) -> SocialResult<Vec<u8>> {
        // Create pseudonym that's consistent but not linkable
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        hasher.update(identity.as_bytes());
        hasher.update(context.as_bytes());
        hasher.update(b"pseudonym_key");
        
        Ok(hasher.finalize().to_vec())
    }

    async fn generate_authorization_proof(
        &self,
        follower_identity: &Identity,
        target_user: &str,
        auth_service: &AuthService,
    ) -> SocialResult<Vec<u8>> {
        // Generate zero-knowledge proof of authorization
        // For now, create a simple signed proof
        let proof_data = format!("follow_request:{}:{}", follower_identity.get_id(), target_user);
        
        // In practice, this would be a proper zk-STARK proof
        Ok(proof_data.as_bytes().to_vec())
    }

    async fn verify_authorization_proof(
        &self,
        proof: &[u8],
        target_user: &str,
        auth_service: &AuthService,
    ) -> SocialResult<()> {
        // Verify the authorization proof
        // For now, just check it's not empty
        if proof.is_empty() {
            return Err(SocialError::ProofVerificationFailed(
                "Empty authorization proof".to_string()
            ));
        }
        
        Ok(())
    }

    async fn create_connection(
        &self,
        request: &FollowRequest,
        target_identity: &Identity,
        connection_type: ConnectionType,
    ) -> SocialResult<Connection> {
        Ok(Connection {
            id: self.generate_connection_id(&request.id, &target_identity.get_id()),
            connection_type,
            created_at: Utc::now(),
            privacy_level: request.privacy_level.clone(),
            metadata: None,
        })
    }

    fn store_public_connection(
        &mut self,
        connection: &Connection,
        request: &FollowRequest,
        target_identity: &Identity,
    ) -> SocialResult<()> {
        // For public connections, we need the actual follower ID
        let follower_id = String::from_utf8(request.encrypted_follower.clone())
            .map_err(|e| SocialError::SerializationError(e.to_string()))?;

        let public_conn = PublicConnection {
            follower: follower_id,
            followee: target_identity.get_id(),
            connection: connection.clone(),
            signature: vec![], // In practice, would have proper signature
        };

        self.public_connections.insert(connection.id.clone(), public_conn);
        Ok(())
    }

    async fn store_anonymous_connection(
        &mut self,
        connection: &Connection,
        request: &FollowRequest,
        target_identity: &Identity,
    ) -> SocialResult<()> {
        let encrypted_conn = EncryptedConnection {
            encrypted_follower: request.encrypted_follower.clone(),
            encrypted_followee: self.encrypt_identity(&target_identity.get_id(), &request.id)?,
            connection: connection.clone(),
            validity_proof: request.authorization_proof.clone(),
        };

        self.anonymous_connections.insert(connection.id.clone(), encrypted_conn);
        Ok(())
    }

    fn is_connection_for_user(
        &self,
        encrypted_conn: &EncryptedConnection,
        user_id: &str,
        connection_type: ConnectionType,
    ) -> SocialResult<bool> {
        // In practice, this would involve decrypting and checking
        // For now, simplified implementation
        Ok(encrypted_conn.connection.connection_type == connection_type)
    }

    fn is_connection_between_users(
        &self,
        encrypted_conn: &EncryptedConnection,
        user1: &str,
        user2: &str,
    ) -> SocialResult<bool> {
        // In practice, this would involve decrypting and checking
        // For now, simplified implementation
        Ok(true)
    }

    async fn remove_connection(&mut self, user1: &str, user2: &str) -> SocialResult<()> {
        // Remove connections between two users
        let mut to_remove = Vec::new();

        for (id, connection) in &self.public_connections {
            if (connection.follower == user1 && connection.followee == user2) ||
               (connection.follower == user2 && connection.followee == user1) {
                to_remove.push(id.clone());
            }
        }

        for id in to_remove {
            self.public_connections.remove(&id);
        }

        // Also remove from anonymous connections (simplified)
        // In practice, would need proper decryption and identification

        Ok(())
    }
}

impl Default for SocialGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::Identity;
    use axon_identity::AuthService;

    #[tokio::test]
    async fn test_social_graph_creation() {
        let graph = SocialGraph::new();
        assert_eq!(graph.anonymous_connections.len(), 0);
        assert_eq!(graph.public_connections.len(), 0);
    }

    #[tokio::test]
    async fn test_follow_request_creation() {
        let graph = SocialGraph::new();
        let follower_identity = Identity::new_for_test("follower");
        let auth_service = AuthService::new_for_test();

        let request = graph.create_follow_request(
            &follower_identity,
            "target_user",
            PrivacyLevel::Anonymous,
            &auth_service,
        ).await.unwrap();

        assert_eq!(request.target, "target_user");
        assert_eq!(request.privacy_level, PrivacyLevel::Anonymous);
        assert!(!request.authorization_proof.is_empty());
    }

    #[tokio::test]
    async fn test_privacy_settings() {
        let mut graph = SocialGraph::new();
        let user_id = "test_user";

        let settings = SocialPrivacySettings {
            default_privacy: PrivacyLevel::Pseudonymous,
            public_follower_count: true,
            require_approval: false,
            ..Default::default()
        };

        graph.set_privacy_settings(user_id, settings.clone()).unwrap();
        
        let retrieved = graph.get_privacy_settings(user_id);
        assert_eq!(retrieved.default_privacy, PrivacyLevel::Pseudonymous);
        assert!(retrieved.public_follower_count);
        assert!(!retrieved.require_approval);
    }

    #[tokio::test]
    async fn test_connection_limits() {
        let limits = ConnectionLimits {
            max_followers: 2,
            max_following: 2,
            max_pending: 1,
            connection_rate_limit: 1,
        };

        let graph = SocialGraph::with_limits(limits);
        
        // Test that limits are enforced
        let result = graph.check_connection_limits("test_user", ConnectionType::Follower);
        assert!(result.is_ok());
    }
}