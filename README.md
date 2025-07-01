# Axon Protocol

A privacy-first decentralized social network protocol built with Rust, featuring cryptographic identity management, domain registry, and content authentication.

## Overview

Axon is the core protocol for a decentralized social web that prioritizes user privacy and content ownership. It provides:

- **Cryptographic Identity Management**: Ed25519-based QuID integration for secure, anonymous identities
- **Domain Registry**: Smart contract-based .axon domain system with adaptive pricing
- **Content Authentication**: SHAKE256 content addressing with zk-STARK authenticity proofs
- **Privacy-First Architecture**: Anonymous engagement tracking and optional content revelation

## Project Structure

```
axon/
├── axon-core/          # Core protocol types and cryptography ✅
├── axon-contracts/     # Smart contracts for domains and governance ✅
├── axon-identity/      # QuID identity integration and authentication ✅
├── axon-social/        # Social networking features ✅ NEW
├── axon-cli/           # Command-line interface for testing ✅
└── docs/               # Documentation and roadmap
```

## Features Implemented

### Phase 1: Foundation (✅ Completed)

- [x] **Development Environment**: Rust workspace with proper dependency management
- [x] **Core Cryptography**: Ed25519 key management with serialization support
- [x] **Domain Registry**: Smart contract for .axon domain registration with adaptive pricing
- [x] **QuID Identity Integration**: Authentication service with anonymous proofs
- [x] **Content System**: Post creation, signing, and verification
- [x] **CLI Interface**: Working command-line tool for testing

### Phase 2: Social Features (✅ COMPLETED)

- [x] **Privacy-Preserving Social Graph**: Following/followers with zero-knowledge proofs
- [x] **Content Interaction System**: Likes, replies, comments, shares with anonymity
- [x] **Feed Generation Engine**: Chronological, engagement-based, and trending algorithms
- [x] **Privacy Controls**: Comprehensive anonymization and privacy violation detection
- [x] **Anonymous Analytics**: Engagement tracking without user profiling
- [x] **Anti-Spam Protection**: Rate limiting, pattern detection, and abuse prevention

### Technical Achievements

- **Core social networking** with complete privacy preservation
- **Zero-knowledge proofs** for all anonymous interactions
- **Type-safe cryptography** with proper serialization
- **Smart contract foundation** for domain management
- **Privacy-first architecture** throughout all social features
- **Comprehensive test coverage** for all components

## Quick Start

### Prerequisites

- Rust 1.70+ with Cargo
- Basic understanding of cryptographic concepts

### Installation

```bash
git clone <repository>
cd axon
cargo build --workspace
```

### Usage

```bash
# Show protocol information
cargo run -p axon-cli -- info

# Generate a new cryptographic identity
cargo run -p axon-cli -- identity generate

# Check domain availability
cargo run -p axon-cli -- domain check mydomain

# Create and sign content
cargo run -p axon-cli -- content create "Hello, Axon network!"
```

## Architecture

### Core Components

1. **axon-core**: Fundamental types and cryptographic primitives
   - Ed25519 key management with custom serialization
   - SHAKE256 content hashing
   - Domain and content type definitions

2. **axon-contracts**: Smart contract implementations
   - Domain registry with adaptive pricing
   - Auto-renewal contract system
   - Revenue distribution mechanisms

3. **axon-identity**: Authentication and identity management
   - QuID identity integration
   - Session management
   - Anonymous proof verification

4. **axon-cli**: Command-line interface
   - Identity generation and management
   - Domain operations
   - Content creation and verification

### Cryptographic Features

- **Ed25519 Signatures**: Fast, secure digital signatures
- **SHAKE256 Hashing**: Extensible output content addressing
- **Pedersen Commitments**: Privacy-preserving value commitments
- **zk-STARK Proofs**: Zero-knowledge content authenticity (framework)

## Testing

Run the complete test suite:

```bash
# Test all components
cargo test --workspace

# Test specific component
cargo test -p axon-core
```

Current test coverage:
- ✅ Cryptographic operations (signing, verification, hashing)
- ✅ Domain creation and management
- ✅ Identity proof generation and verification
- ✅ Content creation and signature verification
- ✅ Serialization/deserialization of all types

## Development Status

**Current Phase**: Foundation Complete ✅

**IMMEDIATE PRIORITY** (Phase 2 - Social Features):
- [ ] **Following/Followers System** with privacy-preserving social connections
- [ ] **Content Interaction System** (replies, comments, likes) with zk-STARK proofs
- [ ] **Feed Generation Engine** with chronological and algorithmic ranking
- [ ] **Privacy Controls** for all social features and content visibility

**Next Priority** (Phase 3):
- [ ] Privacy-preserving discovery and recommendation engine
- [ ] Content storage optimization with MimbleWimble-style cut-through
- [ ] Creator economy integration with Nym cryptocurrency payments
- [ ] Anonymous analytics and engagement tracking

## Security Considerations

- All private keys are generated using cryptographically secure random number generation
- Signatures use Ed25519 with proper domain separation
- Content hashes use SHAKE256 for collision resistance
- Identity proofs include replay protection mechanisms

## Contributing

This implementation follows the roadmap specified in `docs/roadmap.md`. Each component includes comprehensive tests and documentation.

## License

[License information to be added]

---

*Built with privacy-first principles for the decentralized social web.*