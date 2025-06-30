//! Content structures for Axon protocol

use crate::{
    crypto::{AxonSignature, AxonVerifyingKey},
    types::{ContentHash, ContentType, Timestamp, VisibilityLevel},
    AxonError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Individual post content
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Post {
    pub id: ContentHash,
    pub author: AxonVerifyingKey,
    pub content: PostContent,
    pub metadata: PostMetadata,
    pub interactions: PostInteractions,
    pub signature: AxonSignature,
}

/// Post content data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PostContent {
    pub text: Option<String>,
    pub media_hashes: Vec<ContentHash>,
    pub shared_content: Option<ContentHash>,
    pub poll_data: Option<PollData>,
    pub content_type: ContentType,
}

/// Post metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PostMetadata {
    pub created_at: Timestamp,
    pub edited_at: Option<Timestamp>,
    pub visibility: VisibilityLevel,
    pub tags: Vec<String>,
    pub mentions: Vec<AxonVerifyingKey>,
    pub reply_to: Option<ContentHash>,
    pub edit_history: Vec<EditRecord>,
}

/// Edit history record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditRecord {
    pub timestamp: Timestamp,
    pub content_hash: ContentHash,
    pub edit_reason: Option<String>,
}

/// Post interactions (anonymous engagement)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PostInteractions {
    pub replies: Vec<ContentHash>,
    pub like_count_commitment: [u8; 32], // Pedersen commitment
    pub share_count_commitment: [u8; 32],
    pub view_count_commitment: [u8; 32],
    pub engagement_proof: Vec<u8>, // zk-STARK proof placeholder
}

/// Poll data structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PollData {
    pub question: String,
    pub options: Vec<String>,
    pub ends_at: Timestamp,
    pub multiple_choice: bool,
    pub vote_commitments: HashMap<usize, [u8; 32]>, // Option index -> vote count commitment
}

/// User profile structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserProfile {
    pub owner: AxonVerifyingKey,
    pub display_name: String,
    pub bio: Option<String>,
    pub avatar_hash: Option<ContentHash>,
    pub banner_hash: Option<ContentHash>,
    pub links: Vec<String>,
    pub posts: Vec<ContentHash>,
    pub following: Vec<AxonVerifyingKey>,
    pub followers: Vec<AxonVerifyingKey>,
    pub blocked: Vec<[u8; 32]>, // Encrypted QuID list
    pub created_at: Timestamp,
    pub last_updated: Timestamp,
    pub signature: AxonSignature,
}

/// Content feed structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentFeed {
    pub posts: Vec<Post>,
    pub has_more: bool,
    pub next_cursor: Option<String>,
    pub generated_at: Timestamp,
}

/// Anonymous engagement record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnonymousEngagement {
    pub content_hash: ContentHash,
    pub engagement_type: EngagementType,
    pub proof: Vec<u8>, // zk-STARK proof of valid engagement
    pub nullifier: [u8; 32], // Prevents double engagement
}

/// Types of engagement
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EngagementType {
    Like,
    Share,
    Reply,
    View,
}

impl Post {
    /// Verify the post signature
    pub fn verify_signature(&self) -> Result<()> {
        let content_bytes = bincode::serialize(&(
            &self.content,
            &self.metadata,
            &self.interactions,
        ))
        .map_err(|e| AxonError::SerializationError(e.to_string()))?;
        
        self.author.verify(&content_bytes, &self.signature)
    }

    /// Calculate content hash for this post
    pub fn calculate_hash(&self) -> ContentHash {
        let content_bytes = bincode::serialize(self)
            .expect("Post serialization should not fail");
        crate::crypto::hash_content(&content_bytes)
    }
}

impl UserProfile {
    /// Verify the profile signature
    pub fn verify_signature(&self) -> Result<()> {
        let content_bytes = bincode::serialize(&(
            &self.display_name,
            &self.bio,
            &self.avatar_hash,
            &self.banner_hash,
            &self.links,
            &self.last_updated,
        ))
        .map_err(|e| AxonError::SerializationError(e.to_string()))?;
        
        self.owner.verify(&content_bytes, &self.signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::AxonSigningKey;

    #[test]
    fn test_post_creation_and_verification() {
        let signing_key = AxonSigningKey::generate();
        let verifying_key = signing_key.verifying_key();
        
        let content = PostContent {
            text: Some("Hello, Axon!".to_string()),
            media_hashes: vec![],
            shared_content: None,
            poll_data: None,
            content_type: ContentType::Text,
        };
        
        let metadata = PostMetadata {
            created_at: Timestamp::now(),
            edited_at: None,
            visibility: VisibilityLevel::Public,
            tags: vec!["test".to_string()],
            mentions: vec![],
            reply_to: None,
            edit_history: vec![],
        };
        
        let interactions = PostInteractions {
            replies: vec![],
            like_count_commitment: [0u8; 32],
            share_count_commitment: [0u8; 32],
            view_count_commitment: [0u8; 32],
            engagement_proof: vec![],
        };
        
        let content_bytes = bincode::serialize(&(&content, &metadata, &interactions)).unwrap();
        let signature = signing_key.sign(&content_bytes);
        
        let post = Post {
            id: crate::crypto::hash_content(&content_bytes),
            author: verifying_key,
            content,
            metadata,
            interactions,
            signature,
        };
        
        assert!(post.verify_signature().is_ok());
    }
}