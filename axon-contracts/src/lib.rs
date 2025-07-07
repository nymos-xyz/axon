//! # Axon Smart Contracts
//! 
//! Smart contract implementations for domain registry, creator economy,
//! and governance on the Axon protocol.
//!
//! ## Features
//! 
//! - **Domain Registry**: Core domain registration with dynamic pricing
//! - **Adaptive Pricing**: Market-based pricing with ML predictions  
//! - **Governance**: Quadratic voting with delegation and time-locks
//! - **Auto-Renewal**: Automated domain renewal with escrow
//! - **Creator Economy**: Subscription and monetization features
//! - **Revenue Distribution**: Automatic token burning and fee allocation
//! - **Integration**: Fully integrated registry with all features

pub mod domain_registry;
pub mod creator_economy;
pub mod governance;
pub mod pricing;
pub mod auto_renewal;
pub mod quid_axon_registry;
pub mod integrated_registry;

pub use domain_registry::{DomainRegistryContract, RevenueDistribution, DynamicFeeCalculation};
pub use creator_economy::CreatorEconomyContract;
pub use governance::{GovernanceContract, GovernanceConfig, ProposalType, ProposalAction, GovernanceEvent};
pub use pricing::{AdaptivePricingContract, MarketIndicators, PriceTrendPrediction};
pub use auto_renewal::{AutoRenewalContract, AutoRenewalEvent};
pub use quid_axon_registry::{QuidAxonRegistry, NymverseDomainConfig, NymverseDomainType};
pub use integrated_registry::{
    IntegratedRegistryContract, IntegratedRegistryEvent, SystemStatistics, 
    SystemHealthScore, RevenueSystem, MarketAnalysisEngine
};