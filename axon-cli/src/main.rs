//! Axon CLI application

use clap::{Parser, Subcommand};
use axon_core::{
    types::{DomainName, DomainType, Timestamp, VisibilityLevel},
    crypto::AxonSigningKey,
    content::{Post, PostContent, PostMetadata, PostInteractions},
};
use anyhow::Result;

#[derive(Parser)]
#[command(name = "axon")]
#[command(about = "Axon protocol command-line interface")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Identity management commands
    Identity {
        #[command(subcommand)]
        action: IdentityAction,
    },
    /// Domain management commands
    Domain {
        #[command(subcommand)]
        action: DomainAction,
    },
    /// Content operations
    Content {
        #[command(subcommand)]
        action: ContentAction,
    },
    /// Protocol information
    Info,
}

#[derive(Subcommand)]
enum IdentityAction {
    /// Generate a new identity
    Generate,
}

#[derive(Subcommand)]
enum DomainAction {
    /// Check domain availability
    Check { name: String },
}

#[derive(Subcommand)]
enum ContentAction {
    /// Create new content
    Create { text: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Identity { action } => handle_identity_command(action).await,
        Commands::Domain { action } => handle_domain_command(action).await,
        Commands::Content { action } => handle_content_command(action).await,
        Commands::Info => handle_info_command().await,
    }
}

async fn handle_identity_command(action: IdentityAction) -> Result<()> {
    match action {
        IdentityAction::Generate => {
            println!("Generating new identity...");
            
            let signing_key = AxonSigningKey::generate();
            let verifying_key = signing_key.verifying_key();
            
            println!("Generated new identity:");
            println!("  Public Key: {}", hex::encode(verifying_key.to_bytes()));
            println!("  Key bytes: {} bytes", verifying_key.to_bytes().len());
            
            // Test signing
            let data = b"test message";
            let signature = signing_key.sign(data);
            let verification_result = verifying_key.verify(data, &signature);
            
            println!("  Signature test: {}", if verification_result.is_ok() { "PASSED" } else { "FAILED" });
        }
    }
    
    Ok(())
}

async fn handle_domain_command(action: DomainAction) -> Result<()> {
    match action {
        DomainAction::Check { name } => {
            println!("Checking domain availability: {}.axon", name);
            
            let domain_name = DomainName::new(name)?;
            println!("Domain name: {}", domain_name);
            println!("Full domain: {}", domain_name.full_domain());
            println!("Status: Available (demo mode)");
        }
    }
    
    Ok(())
}

async fn handle_content_command(action: ContentAction) -> Result<()> {
    match action {
        ContentAction::Create { text } => {
            println!("Creating content...");
            
            let signing_key = AxonSigningKey::generate();
            
            let content = PostContent {
                text: Some(text.clone()),
                media_hashes: vec![],
                shared_content: None,
                poll_data: None,
                content_type: axon_core::types::ContentType::Text,
            };
            
            let metadata = PostMetadata {
                created_at: Timestamp::now(),
                edited_at: None,
                visibility: VisibilityLevel::Public,
                tags: vec!["cli".to_string()],
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
            
            let content_bytes = bincode::serialize(&(&content, &metadata, &interactions))?;
            let signature = signing_key.sign(&content_bytes);
            
            let post = Post {
                id: axon_core::crypto::hash_content(&content_bytes),
                author: signing_key.verifying_key(),
                content,
                metadata,
                interactions,
                signature,
            };
            
            println!("Content created successfully!");
            println!("  ID: {}", post.id);
            println!("  Author: {}", hex::encode(post.author.to_bytes()));
            println!("  Text: {}", text);
            println!("  Created: {}", post.metadata.created_at.as_secs());
            println!("  Visibility: {:?}", post.metadata.visibility);
            
            // Verify post signature
            let verification_result = post.verify_signature();
            println!("  Signature verification: {}", if verification_result.is_ok() { "PASSED" } else { "FAILED" });
        }
    }
    
    Ok(())
}

async fn handle_info_command() -> Result<()> {
    println!("Axon Protocol Information");
    println!("========================");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!("Protocol Version: {}", axon_core::PROTOCOL_VERSION);
    println!("Max Content Size: {} bytes", axon_core::MAX_CONTENT_SIZE);
    println!("Max Domain Length: {} characters", axon_core::MAX_DOMAIN_LENGTH);
    println!("Min Domain Length: {} characters", axon_core::MIN_DOMAIN_LENGTH);
    println!();
    println!("Available Commands:");
    println!("  identity generate - Generate a new cryptographic identity");
    println!("  domain check <name> - Check .axon domain availability");
    println!("  content create <text> - Create and sign new content");
    println!("  info - Show this information");
    println!();
    println!("Examples:");
    println!("  axon identity generate");
    println!("  axon domain check mydomain");
    println!("  axon content create \"Hello, Axon network!\"");
    
    Ok(())
}