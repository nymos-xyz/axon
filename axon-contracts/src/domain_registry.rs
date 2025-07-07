//! Domain registry smart contract implementation

use axon_core::{
    domain::{DomainRecord, DomainRegistrationRequest, DomainPricing},
    types::{DomainName, DomainType, Timestamp},
    crypto::AxonVerifyingKey,
    Result, AxonError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, debug};

/// Domain registry smart contract state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainRegistryContract {
    /// Registered domains
    domains: HashMap<DomainName, DomainRecord>,
    /// Domain reservations (pending registration)
    reservations: HashMap<DomainName, DomainReservation>,
    /// Current pricing configuration
    pricing: DomainPricing,
    /// Contract administrator
    admin: AxonVerifyingKey,
    /// Total domains registered
    total_registered: u64,
    /// Contract creation time
    created_at: Timestamp,
}

/// Domain reservation for pending registrations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainReservation {
    pub domain_name: DomainName,
    pub requester: AxonVerifyingKey,
    pub request: DomainRegistrationRequest,
    pub reserved_at: Timestamp,
    pub expires_at: Timestamp,
    pub payment_received: bool,
}

/// Domain registry events
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DomainRegistryEvent {
    DomainRegistered {
        domain: DomainName,
        owner: AxonVerifyingKey,
        expires_at: Timestamp,
    },
    DomainRenewed {
        domain: DomainName,
        new_expiry: Timestamp,
    },
    DomainTransferred {
        domain: DomainName,
        from: AxonVerifyingKey,
        to: AxonVerifyingKey,
    },
    DomainExpired {
        domain: DomainName,
        owner: AxonVerifyingKey,
    },
    PricingUpdated {
        updated_at: Timestamp,
    },
}

impl DomainRegistryContract {
    /// Create a new domain registry contract
    pub fn new(admin: AxonVerifyingKey, initial_pricing: DomainPricing) -> Self {
        Self {
            domains: HashMap::new(),
            reservations: HashMap::new(),
            pricing: initial_pricing,
            admin,
            total_registered: 0,
            created_at: Timestamp::now(),
        }
    }

    /// Register a new domain
    pub fn register_domain(
        &mut self,
        request: DomainRegistrationRequest,
    ) -> Result<DomainRegistryEvent> {
        // Verify the registration request
        request.verify_signature()?;

        // Check if domain is available
        if self.domains.contains_key(&request.domain_name) {
            return Err(AxonError::InvalidDomain(
                "Domain already registered".to_string(),
            ));
        }

        // Check if domain is reserved by someone else
        if let Some(reservation) = self.reservations.get(&request.domain_name) {
            if reservation.requester != request.owner {
                return Err(AxonError::InvalidDomain(
                    "Domain reserved by another user".to_string(),
                ));
            }
        }

        // Calculate registration cost
        let cost = self.calculate_registration_cost(&request.domain_type, request.duration_years);
        
        // Verify payment (placeholder - would integrate with actual payment system)
        if !self.verify_payment(&request.payment_proof, cost) {
            return Err(AxonError::InvalidDomain(
                "Payment verification failed".to_string(),
            ));
        }

        // Create domain record
        let expires_at = Timestamp(Timestamp::now().0 + (request.duration_years as u64 * 365 * 24 * 3600));
        
        // Create temporary signing key for domain record signature (in real implementation, would use proper key management)
        let temp_key = axon_core::crypto::AxonSigningKey::generate();
        let domain_signature = temp_key.sign(&[0u8; 32]); // Placeholder
        
        let domain_record = DomainRecord {
            domain_name: request.domain_name.clone(),
            owner: request.owner.clone(),
            content_root_hash: request.initial_content_hash,
            registered_at: Timestamp::now(),
            expires_at,
            domain_type: request.domain_type.clone(),
            auto_renewal: request.auto_renewal,
            transfer_rules: axon_core::domain::TransferPolicy {
                transferable: true,
                requires_approval: false,
                authorized_transferees: vec![],
                transfer_cooldown_days: 0,
            },
            metadata: request.metadata,
            signature: domain_signature,
        };

        // Store domain record
        self.domains.insert(request.domain_name.clone(), domain_record);
        self.total_registered += 1;

        // Remove reservation if exists
        self.reservations.remove(&request.domain_name);

        Ok(DomainRegistryEvent::DomainRegistered {
            domain: request.domain_name,
            owner: request.owner,
            expires_at,
        })
    }

    /// Renew an existing domain
    pub fn renew_domain(
        &mut self,
        domain_name: &DomainName,
        owner: &AxonVerifyingKey,
        years: u32,
        payment_proof: Vec<u8>,
    ) -> Result<DomainRegistryEvent> {
        let domain = self.domains.get_mut(domain_name)
            .ok_or(AxonError::DomainNotFound)?;

        // Verify ownership
        if &domain.owner != owner {
            return Err(AxonError::PermissionDenied);
        }

        // Calculate renewal cost
        let cost = domain.calculate_renewal_cost(&self.pricing, years);

        // Verify payment
        let payment_valid = self.verify_payment(&payment_proof, cost);
        if !payment_valid {
            return Err(AxonError::InvalidDomain(
                "Payment verification failed".to_string(),
            ));
        }

        // Update expiration date
        let additional_time = years as u64 * 365 * 24 * 3600;
        domain.expires_at = Timestamp(domain.expires_at.0 + additional_time);

        Ok(DomainRegistryEvent::DomainRenewed {
            domain: domain_name.clone(),
            new_expiry: domain.expires_at,
        })
    }

    /// Transfer domain ownership
    pub fn transfer_domain(
        &mut self,
        domain_name: &DomainName,
        from: &AxonVerifyingKey,
        to: &AxonVerifyingKey,
    ) -> Result<DomainRegistryEvent> {
        let domain = self.domains.get_mut(domain_name)
            .ok_or(AxonError::DomainNotFound)?;

        // Verify ownership
        if &domain.owner != from {
            return Err(AxonError::PermissionDenied);
        }

        // Check transfer policy
        if !domain.transfer_rules.transferable {
            return Err(AxonError::PermissionDenied);
        }

        // Update ownership
        domain.owner = to.clone();

        Ok(DomainRegistryEvent::DomainTransferred {
            domain: domain_name.clone(),
            from: from.clone(),
            to: to.clone(),
        })
    }

    /// Get domain information
    pub fn get_domain(&self, domain_name: &DomainName) -> Option<&DomainRecord> {
        self.domains.get(domain_name)
    }

    /// Check if domain is available
    pub fn is_available(&self, domain_name: &DomainName) -> bool {
        !self.domains.contains_key(domain_name) && 
        !self.reservations.contains_key(domain_name)
    }

    /// Reserve a domain name temporarily
    pub fn reserve_domain(
        &mut self,
        domain_name: DomainName,
        requester: AxonVerifyingKey,
        request: DomainRegistrationRequest,
    ) -> Result<()> {
        if !self.is_available(&domain_name) {
            return Err(AxonError::InvalidDomain(
                "Domain not available".to_string(),
            ));
        }

        let reservation = DomainReservation {
            domain_name: domain_name.clone(),
            requester,
            request,
            reserved_at: Timestamp::now(),
            expires_at: Timestamp(Timestamp::now().0 + 1800), // 30 minutes
            payment_received: false,
        };

        self.reservations.insert(domain_name, reservation);
        Ok(())
    }

    /// Clean up expired reservations
    pub fn cleanup_expired_reservations(&mut self) -> u32 {
        let now = Timestamp::now();
        let initial_count = self.reservations.len();
        
        self.reservations.retain(|_, reservation| {
            reservation.expires_at.0 > now.0
        });
        
        (initial_count - self.reservations.len()) as u32
    }

    /// Mark expired domains
    pub fn process_expired_domains(&mut self) -> Vec<DomainRegistryEvent> {
        let mut events = Vec::new();
        let now = Timestamp::now();

        for (domain_name, domain) in &self.domains {
            if domain.expires_at.0 <= now.0 {
                events.push(DomainRegistryEvent::DomainExpired {
                    domain: domain_name.clone(),
                    owner: domain.owner.clone(),
                });
            }
        }

        events
    }

    /// Update pricing configuration
    pub fn update_pricing(
        &mut self,
        new_pricing: DomainPricing,
        admin: &AxonVerifyingKey,
    ) -> Result<DomainRegistryEvent> {
        if admin != &self.admin {
            return Err(AxonError::PermissionDenied);
        }

        self.pricing = new_pricing;
        
        Ok(DomainRegistryEvent::PricingUpdated {
            updated_at: Timestamp::now(),
        })
    }

    /// Calculate registration cost for a domain type and duration
    fn calculate_registration_cost(&self, domain_type: &DomainType, years: u32) -> u64 {
        let base_price = match domain_type {
            DomainType::Standard => self.pricing.base_prices.standard,
            DomainType::Premium => self.pricing.base_prices.premium,
            DomainType::Vanity => self.pricing.base_prices.vanity,
            DomainType::Organization => self.pricing.base_prices.organization,
            DomainType::Community => self.pricing.base_prices.community,
        };

        let adjusted_price = (base_price as f32 * 
            self.pricing.network_health_multiplier * 
            self.pricing.demand_multiplier) as u64;
        
        // Apply multi-year discount
        let duration_multiplier = match years {
            1 => 1.0,
            2 => 1.9,    // 5% discount
            3 => 2.7,    // 10% discount
            4 => 3.4,    // 15% discount
            5..=10 => (years as f32 * 0.8), // 20% discount
            _ => (years as f32 * 0.75), // 25% discount for 10+ years
        };
        
        (adjusted_price as f32 * duration_multiplier) as u64
    }
    
    /// Calculate and distribute registration revenue
    fn process_registration_revenue(&mut self, amount: u64) -> RevenueDistribution {
        // Default revenue distribution percentages
        let development_share = (amount as f32 * 0.25) as u64;  // 25% to development
        let ecosystem_share = (amount as f32 * 0.20) as u64;    // 20% to ecosystem
        let burn_amount = (amount as f32 * 0.15) as u64;        // 15% burned (deflationary)
        let validator_share = (amount as f32 * 0.30) as u64;    // 30% to validators
        let creator_share = (amount as f32 * 0.10) as u64;      // 10% to creator economy
        
        RevenueDistribution {
            total_amount: amount,
            development_fund: development_share,
            ecosystem_fund: ecosystem_share,
            burned_tokens: burn_amount,
            validator_rewards: validator_share,
            creator_rewards: creator_share,
            timestamp: Timestamp::now(),
        }
    }

    /// Verify payment proof (placeholder implementation)
    fn verify_payment(&self, _payment_proof: &[u8], _amount: u64) -> bool {
        // In a real implementation, this would verify the payment proof
        // against the blockchain payment system
        true
    }

    /// Get contract statistics
    pub fn get_stats(&self) -> DomainRegistryStats {
        let mut type_counts = HashMap::new();
        let mut expired_count = 0;
        let now = Timestamp::now();

        for domain in self.domains.values() {
            *type_counts.entry(domain.domain_type.clone()).or_insert(0) += 1;
            if domain.expires_at.0 <= now.0 {
                expired_count += 1;
            }
        }

        DomainRegistryStats {
            total_registered: self.total_registered,
            active_domains: self.domains.len() as u64 - expired_count,
            expired_domains: expired_count,
            pending_reservations: self.reservations.len() as u64,
            domains_by_type: type_counts,
            pricing: self.pricing.clone(),
            total_revenue: 0, // Would be tracked separately
            burned_tokens: 0, // Would be tracked separately  
            revenue_distribution: Vec::new(), // Would be tracked separately
        }
    }
    
    /// Calculate registration fee with dynamic pricing
    pub fn calculate_dynamic_fee(
        &self,
        domain_type: &DomainType,
        domain_name: &str,
        years: u32,
        user: &AxonVerifyingKey,
        bulk_count: Option<u32>,
    ) -> DynamicFeeCalculation {
        let base_cost = self.calculate_registration_cost(domain_type, years);
        let mut final_cost = base_cost;
        let mut applied_discounts = Vec::new();
        let mut applied_surcharges = Vec::new();
        
        // Length-based pricing
        if domain_name.len() <= 3 {
            let surcharge = (base_cost as f32 * 0.5) as u64; // 50% surcharge for short domains
            final_cost += surcharge;
            applied_surcharges.push(FeeAdjustment {
                reason: "Short domain premium".to_string(),
                amount: surcharge,
                percentage: 50.0,
            });
        }
        
        // Bulk registration discount
        if let Some(count) = bulk_count {
            if count >= 10 {
                let discount = (final_cost as f32 * 0.15) as u64; // 15% bulk discount
                final_cost = final_cost.saturating_sub(discount);
                applied_discounts.push(FeeAdjustment {
                    reason: "Bulk registration discount".to_string(),
                    amount: discount,
                    percentage: 15.0,
                });
            }
        }
        
        // Premium keyword detection
        let premium_keywords = ["crypto", "nft", "defi", "web3", "dao", "ai", "ml"];
        for keyword in &premium_keywords {
            if domain_name.to_lowercase().contains(keyword) {
                let surcharge = (base_cost as f32 * 0.25) as u64; // 25% premium
                final_cost += surcharge;
                applied_surcharges.push(FeeAdjustment {
                    reason: format!("Premium keyword: {}", keyword),
                    amount: surcharge,
                    percentage: 25.0,
                });
                break;
            }
        }
        
        DynamicFeeCalculation {
            base_cost,
            final_cost,
            discounts: applied_discounts,
            surcharges: applied_surcharges,
            effective_discount_rate: if final_cost < base_cost {
                ((base_cost - final_cost) as f32 / base_cost as f32) * 100.0
            } else {
                0.0
            },
            calculated_at: Timestamp::now(),
        }
    }
}

/// Domain registry statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainRegistryStats {
    pub total_registered: u64,
    pub active_domains: u64,
    pub expired_domains: u64,
    pub pending_reservations: u64,
    pub domains_by_type: HashMap<DomainType, u64>,
    pub pricing: DomainPricing,
    pub total_revenue: u64,
    pub burned_tokens: u64,
    pub revenue_distribution: Vec<RevenueDistribution>,
}

/// Revenue distribution record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RevenueDistribution {
    pub total_amount: u64,
    pub development_fund: u64,
    pub ecosystem_fund: u64,
    pub burned_tokens: u64,
    pub validator_rewards: u64,
    pub creator_rewards: u64,
    pub timestamp: Timestamp,
}

/// Enhanced domain registry with revenue tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnhancedDomainRegistryContract {
    pub base_registry: DomainRegistryContract,
    pub revenue_tracker: RevenueTracker,
    pub burn_schedule: TokenBurnSchedule,
    pub fee_calculator: DynamicFeeCalculator,
}

/// Revenue tracking system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RevenueTracker {
    pub total_revenue: u64,
    pub distributed_revenue: u64,
    pub burned_tokens: u64,
    pub revenue_history: Vec<RevenueDistribution>,
    pub monthly_targets: HashMap<String, u64>, // YYYY-MM -> target revenue
    pub last_distribution: Option<Timestamp>,
}

/// Token burning schedule
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenBurnSchedule {
    pub total_burned: u64,
    pub burn_rate: f32, // Percentage of revenue to burn
    pub burn_events: Vec<BurnEvent>,
    pub next_burn: Option<Timestamp>,
    pub burn_interval_seconds: u64,
}

/// Token burn event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnEvent {
    pub amount: u64,
    pub reason: BurnReason,
    pub block_height: u64,
    pub timestamp: Timestamp,
    pub transaction_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BurnReason {
    ScheduledBurn,
    ExcessRevenue,
    GovernanceDecision,
    NetworkUpgrade,
}

/// Dynamic fee calculator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicFeeCalculator {
    pub base_multiplier: f32,
    pub network_congestion_factor: f32,
    pub demand_surge_multiplier: f32,
    pub loyalty_discounts: HashMap<AxonVerifyingKey, f32>, // User -> discount %
    pub bulk_registration_discounts: Vec<BulkDiscount>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BulkDiscount {
    pub min_domains: u32,
    pub discount_percentage: f32,
    pub max_discount_amount: u64,
}

/// Dynamic fee calculation result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicFeeCalculation {
    pub base_cost: u64,
    pub final_cost: u64,
    pub discounts: Vec<FeeAdjustment>,
    pub surcharges: Vec<FeeAdjustment>,
    pub effective_discount_rate: f32,
    pub calculated_at: Timestamp,
}

/// Fee adjustment (discount or surcharge)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeeAdjustment {
    pub reason: String,
    pub amount: u64,
    pub percentage: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{
        crypto::AxonSigningKey,
        domain::{DomainMetadata, VerificationStatus, DomainTypePricing},
    };

    #[test]
    fn test_domain_registration() {
        let admin_key = AxonSigningKey::generate();
        let user_key = AxonSigningKey::generate();
        
        let pricing = DomainPricing {
            base_prices: DomainTypePricing {
                standard: 1000,
                premium: 5000,
                vanity: 2000,
                organization: 3000,
                community: 1500,
            },
            network_health_multiplier: 1.0,
            demand_multiplier: 1.0,
            updated_at: Timestamp::now(),
        };

        let mut registry = DomainRegistryContract::new(admin_key.verifying_key(), pricing);
        
        let domain_name = DomainName::new("testdomain".to_string()).unwrap();
        let request = DomainRegistrationRequest {
            domain_name: domain_name.clone(),
            domain_type: DomainType::Standard,
            duration_years: 1,
            owner: user_key.verifying_key(),
            initial_content_hash: axon_core::crypto::hash_content(b"initial content"),
            auto_renewal: None,
            metadata: DomainMetadata {
                description: Some("Test domain".to_string()),
                keywords: vec![],
                contact_info: None,
                category: None,
                reputation_score: 0,
                verification_status: VerificationStatus::Unverified,
            },
            payment_proof: vec![],
            signature: user_key.sign(&[]), // Placeholder
        };

        let result = registry.register_domain(request);
        assert!(result.is_ok());
        
        assert!(!registry.is_available(&domain_name));
        assert!(registry.get_domain(&domain_name).is_some());
    }

    #[test]
    fn test_domain_reservation() {
        let admin_key = AxonSigningKey::generate();
        let user_key = AxonSigningKey::generate();
        
        let pricing = DomainPricing {
            base_prices: DomainTypePricing {
                standard: 1000,
                premium: 5000,
                vanity: 2000,
                organization: 3000,
                community: 1500,
            },
            network_health_multiplier: 1.0,
            demand_multiplier: 1.0,
            updated_at: Timestamp::now(),
        };

        let mut registry = DomainRegistryContract::new(admin_key.verifying_key(), pricing);
        
        let domain_name = DomainName::new("reserved".to_string()).unwrap();
        
        // Create a placeholder request
        let request = DomainRegistrationRequest {
            domain_name: domain_name.clone(),
            domain_type: DomainType::Standard,
            duration_years: 1,
            owner: user_key.verifying_key(),
            initial_content_hash: axon_core::crypto::hash_content(b"content"),
            auto_renewal: None,
            metadata: DomainMetadata {
                description: None,
                keywords: vec![],
                contact_info: None,
                category: None,
                reputation_score: 0,
                verification_status: VerificationStatus::Unverified,
            },
            payment_proof: vec![],
            signature: user_key.sign(&[]),
        };

        let result = registry.reserve_domain(
            domain_name.clone(),
            user_key.verifying_key(),
            request,
        );
        
        assert!(result.is_ok());
        assert!(!registry.is_available(&domain_name));
    }
    
    #[test]
    fn test_dynamic_fee_calculation() {
        let admin_key = AxonSigningKey::generate();
        let user_key = AxonSigningKey::generate();
        
        let pricing = DomainPricing {
            base_prices: DomainTypePricing {
                standard: 1000,
                premium: 5000,
                vanity: 2000,
                organization: 3000,
                community: 1500,
            },
            network_health_multiplier: 1.0,
            demand_multiplier: 1.0,
            updated_at: Timestamp::now(),
        };

        let registry = DomainRegistryContract::new(admin_key.verifying_key(), pricing);
        
        // Test short domain premium
        let calculation = registry.calculate_dynamic_fee(
            &DomainType::Standard,
            "ai",
            1,
            &user_key.verifying_key(),
            None,
        );
        
        assert!(calculation.final_cost > calculation.base_cost);
        assert!(!calculation.surcharges.is_empty());
        
        // Test bulk discount
        let bulk_calculation = registry.calculate_dynamic_fee(
            &DomainType::Standard,
            "testdomain",
            1,
            &user_key.verifying_key(),
            Some(15), // 15 domains
        );
        
        assert!(bulk_calculation.final_cost < bulk_calculation.base_cost);
        assert!(!bulk_calculation.discounts.is_empty());
    }
    
    #[test]
    fn test_revenue_distribution() {
        let admin_key = AxonSigningKey::generate();
        let pricing = DomainPricing {
            base_prices: DomainTypePricing {
                standard: 1000,
                premium: 5000,
                vanity: 2000,
                organization: 3000,
                community: 1500,
            },
            network_health_multiplier: 1.0,
            demand_multiplier: 1.0,
            updated_at: Timestamp::now(),
        };

        let mut registry = DomainRegistryContract::new(admin_key.verifying_key(), pricing);
        
        let distribution = registry.process_registration_revenue(10000);
        
        assert_eq!(distribution.total_amount, 10000);
        assert_eq!(distribution.development_fund, 2500);  // 25%
        assert_eq!(distribution.ecosystem_fund, 2000);    // 20%
        assert_eq!(distribution.burned_tokens, 1500);     // 15%
        assert_eq!(distribution.validator_rewards, 3000); // 30%
        assert_eq!(distribution.creator_rewards, 1000);   // 10%
        
        // Should sum to total
        let total_distributed = distribution.development_fund + 
                               distribution.ecosystem_fund + 
                               distribution.burned_tokens + 
                               distribution.validator_rewards + 
                               distribution.creator_rewards;
        assert_eq!(total_distributed, distribution.total_amount);
    }
}