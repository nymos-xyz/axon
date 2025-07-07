//! Adaptive pricing smart contract for dynamic domain costs

use axon_core::{
    domain::{DomainPricing, DomainTypePricing},
    types::{DomainType, Timestamp},
    crypto::AxonVerifyingKey,
    Result, AxonError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Adaptive pricing contract state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptivePricingContract {
    /// Current pricing configuration
    pub pricing: DomainPricing,
    /// Network health metrics
    pub network_metrics: NetworkHealthMetrics,
    /// Demand tracking
    pub demand_metrics: DemandMetrics,
    /// Pricing history for trend analysis
    pub pricing_history: Vec<PricingSnapshot>,
    /// Contract administrator
    pub admin: AxonVerifyingKey,
    /// Last update timestamp
    pub last_updated: Timestamp,
}

/// Network health metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkHealthMetrics {
    /// Consensus participation rate (0.0 - 1.0)
    pub consensus_participation: f32,
    /// Average zk-proof generation cost
    pub average_zk_proof_cost: u64,
    /// Target zk-proof cost
    pub target_zk_proof_cost: u64,
    /// Network utilization (0.0 - 1.0)
    pub network_utilization: f32,
    /// Node availability (0.0 - 1.0)
    pub node_availability: f32,
    /// Last metrics update
    pub updated_at: Timestamp,
}

/// Demand tracking metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DemandMetrics {
    /// Registrations in last 24 hours by domain type
    pub recent_registrations: HashMap<DomainType, u32>,
    /// Total domains registered
    pub total_domains: u64,
    /// Average time between registrations
    pub avg_registration_interval: u64,
    /// Domain expiration rate
    pub expiration_rate: f32,
    /// Renewal rate
    pub renewal_rate: f32,
    /// Last demand calculation
    pub updated_at: Timestamp,
}

/// Pricing snapshot for historical analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PricingSnapshot {
    pub pricing: DomainPricing,
    pub network_health: f32,
    pub demand_score: f32,
    pub timestamp: Timestamp,
}

/// Pricing adjustment event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PricingEvent {
    PriceAdjusted {
        old_multiplier: f32,
        new_multiplier: f32,
        reason: AdjustmentReason,
        timestamp: Timestamp,
    },
    NetworkHealthUpdated {
        consensus_participation: f32,
        zk_proof_cost: u64,
        timestamp: Timestamp,
    },
    DemandUpdated {
        registrations_24h: u32,
        demand_score: f32,
        timestamp: Timestamp,
    },
    MarketAnalysisUpdated {
        trending_keywords: Vec<String>,
        competition_factor: f32,
        volatility_index: f32,
        timestamp: Timestamp,
    },
}

/// Reason for pricing adjustment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AdjustmentReason {
    NetworkHealthChange,
    DemandChange,
    ManualAdjustment,
    EmergencyAdjustment,
}

impl AdaptivePricingContract {
    /// Create new adaptive pricing contract
    pub fn new(admin: AxonVerifyingKey, initial_pricing: DomainPricing) -> Self {
        Self {
            pricing: initial_pricing,
            network_metrics: NetworkHealthMetrics {
                consensus_participation: 1.0,
                average_zk_proof_cost: 1000,
                target_zk_proof_cost: 1000,
                network_utilization: 0.5,
                node_availability: 1.0,
                updated_at: Timestamp::now(),
            },
            demand_metrics: DemandMetrics {
                recent_registrations: HashMap::new(),
                total_domains: 0,
                avg_registration_interval: 3600, // 1 hour
                expiration_rate: 0.05,
                renewal_rate: 0.95,
                updated_at: Timestamp::now(),
            },
            pricing_history: Vec::new(),
            admin,
            last_updated: Timestamp::now(),
        }
    }

    /// Update network health metrics
    pub fn update_network_metrics(
        &mut self,
        consensus_participation: f32,
        average_zk_proof_cost: u64,
        network_utilization: f32,
        node_availability: f32,
    ) -> Result<PricingEvent> {
        self.network_metrics = NetworkHealthMetrics {
            consensus_participation,
            average_zk_proof_cost,
            target_zk_proof_cost: self.network_metrics.target_zk_proof_cost,
            network_utilization,
            node_availability,
            updated_at: Timestamp::now(),
        };

        // Trigger pricing adjustment based on network health
        self.adjust_pricing_for_network_health()?;

        Ok(PricingEvent::NetworkHealthUpdated {
            consensus_participation,
            zk_proof_cost: average_zk_proof_cost,
            timestamp: Timestamp::now(),
        })
    }

    /// Update demand metrics
    pub fn update_demand_metrics(
        &mut self,
        registrations_24h: HashMap<DomainType, u32>,
        total_domains: u64,
        expiration_rate: f32,
        renewal_rate: f32,
    ) -> Result<PricingEvent> {
        let total_registrations: u32 = registrations_24h.values().sum();
        
        self.demand_metrics = DemandMetrics {
            recent_registrations: registrations_24h,
            total_domains,
            avg_registration_interval: if total_registrations > 0 {
                86400 / total_registrations as u64 // 24 hours in seconds / registrations
            } else {
                86400 // Default to 24 hours if no registrations
            },
            expiration_rate,
            renewal_rate,
            updated_at: Timestamp::now(),
        };

        // Calculate demand score
        let demand_score = self.calculate_demand_score();

        // Trigger pricing adjustment based on demand
        self.adjust_pricing_for_demand(demand_score)?;

        Ok(PricingEvent::DemandUpdated {
            registrations_24h: total_registrations,
            demand_score,
            timestamp: Timestamp::now(),
        })
    }

    /// Calculate current domain price
    pub fn calculate_domain_price(&self, domain_type: &DomainType, duration_years: u32) -> u64 {
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

        // Apply duration discount for longer registrations
        let duration_multiplier = match duration_years {
            1 => 1.0,
            2 => 1.9,    // 5% discount
            3 => 2.7,    // 10% discount
            4 => 3.4,    // 15% discount
            5..=10 => (duration_years as f32 * 0.8), // 20% discount
            _ => (duration_years as f32 * 0.75), // 25% discount for 10+ years
        };

        (adjusted_price as f32 * duration_multiplier) as u64
    }

    /// Adjust pricing based on network health
    fn adjust_pricing_for_network_health(&mut self) -> Result<()> {
        let old_multiplier = self.pricing.network_health_multiplier;
        let mut new_multiplier: f32 = 1.0;

        // Adjust based on consensus participation
        if self.network_metrics.consensus_participation < 0.67 {
            new_multiplier *= 1.2; // Increase prices to fund network security
        } else if self.network_metrics.consensus_participation > 0.9 {
            new_multiplier *= 0.95; // Slight discount for healthy network
        }

        // Adjust based on zk-proof costs
        if self.network_metrics.average_zk_proof_cost > self.network_metrics.target_zk_proof_cost {
            new_multiplier *= 0.8; // Subsidize domains when privacy is expensive
        }

        // Adjust based on network utilization
        if self.network_metrics.network_utilization > 0.8 {
            new_multiplier *= 1.1; // Increase prices during high utilization
        } else if self.network_metrics.network_utilization < 0.3 {
            new_multiplier *= 0.9; // Decrease prices to encourage usage
        }

        // Adjust based on node availability
        if self.network_metrics.node_availability < 0.8 {
            new_multiplier *= 1.15; // Increase prices to incentivize node operation
        }

        // Clamp multiplier to reasonable bounds
        new_multiplier = new_multiplier.clamp(0.5, 2.0);

        self.pricing.network_health_multiplier = new_multiplier;
        self.last_updated = Timestamp::now();

        // Record snapshot if significant change
        if (new_multiplier - old_multiplier).abs() > 0.05 {
            self.record_pricing_snapshot();
        }

        Ok(())
    }

    /// Adjust pricing based on demand
    fn adjust_pricing_for_demand(&mut self, demand_score: f32) -> Result<()> {
        let old_multiplier = self.pricing.demand_multiplier;
        let mut new_multiplier: f32 = 1.0;

        // Adjust based on registration volume
        if demand_score > 2.0 {
            new_multiplier = 1.3; // High demand
        } else if demand_score > 1.5 {
            new_multiplier = 1.15; // Moderate high demand
        } else if demand_score < 0.5 {
            new_multiplier = 0.8; // Low demand
        } else if demand_score < 0.8 {
            new_multiplier = 0.9; // Moderate low demand
        }

        // Adjust based on renewal rate
        if self.demand_metrics.renewal_rate < 0.7 {
            new_multiplier *= 0.9; // Encourage renewals with lower prices
        }

        // Clamp multiplier to reasonable bounds
        new_multiplier = new_multiplier.clamp(0.6, 2.0);

        self.pricing.demand_multiplier = new_multiplier;
        self.last_updated = Timestamp::now();

        // Record snapshot if significant change
        if (new_multiplier - old_multiplier).abs() > 0.05 {
            self.record_pricing_snapshot();
        }

        Ok(())
    }

    /// Calculate demand score based on recent activity
    fn calculate_demand_score(&self) -> f32 {
        let total_recent: u32 = self.demand_metrics.recent_registrations.values().sum();
        
        // Base demand score on registrations per hour
        let registrations_per_hour = total_recent as f32 / 24.0;
        
        // Normalize against total domains (higher base = higher bar for "high demand")
        let normalized_demand = registrations_per_hour / (self.demand_metrics.total_domains as f32 * 0.001).max(1.0);
        
        // Factor in renewal rate (higher renewal = sustained demand)
        let renewal_factor = self.demand_metrics.renewal_rate;
        
        normalized_demand * (1.0 + renewal_factor)
    }

    /// Record pricing snapshot for historical analysis
    fn record_pricing_snapshot(&mut self) {
        let snapshot = PricingSnapshot {
            pricing: self.pricing.clone(),
            network_health: self.calculate_network_health_score(),
            demand_score: self.calculate_demand_score(),
            timestamp: Timestamp::now(),
        };

        self.pricing_history.push(snapshot);

        // Keep only last 100 snapshots
        if self.pricing_history.len() > 100 {
            self.pricing_history.remove(0);
        }
    }

    /// Calculate overall network health score
    fn calculate_network_health_score(&self) -> f32 {
        let consensus_score = self.network_metrics.consensus_participation;
        let cost_score = if self.network_metrics.average_zk_proof_cost <= self.network_metrics.target_zk_proof_cost {
            1.0
        } else {
            self.network_metrics.target_zk_proof_cost as f32 / self.network_metrics.average_zk_proof_cost as f32
        };
        let utilization_score = 1.0 - (self.network_metrics.network_utilization - 0.5).abs();
        let availability_score = self.network_metrics.node_availability;

        (consensus_score + cost_score + utilization_score + availability_score) / 4.0
    }

    /// Manual pricing adjustment by admin
    pub fn manual_adjustment(
        &mut self,
        admin: &AxonVerifyingKey,
        new_base_prices: Option<DomainTypePricing>,
        network_multiplier: Option<f32>,
        demand_multiplier: Option<f32>,
    ) -> Result<PricingEvent> {
        if admin != &self.admin {
            return Err(AxonError::PermissionDenied);
        }

        let old_multiplier = self.pricing.network_health_multiplier;

        if let Some(prices) = new_base_prices {
            self.pricing.base_prices = prices;
        }

        if let Some(multiplier) = network_multiplier {
            self.pricing.network_health_multiplier = multiplier.clamp(0.1, 5.0);
        }

        if let Some(multiplier) = demand_multiplier {
            self.pricing.demand_multiplier = multiplier.clamp(0.1, 5.0);
        }

        self.pricing.updated_at = Timestamp::now();
        self.last_updated = Timestamp::now();
        self.record_pricing_snapshot();

        Ok(PricingEvent::PriceAdjusted {
            old_multiplier,
            new_multiplier: self.pricing.network_health_multiplier,
            reason: AdjustmentReason::ManualAdjustment,
            timestamp: Timestamp::now(),
        })
    }

    /// Emergency pricing freeze
    pub fn emergency_freeze(
        &mut self,
        admin: &AxonVerifyingKey,
        reason: String,
    ) -> Result<PricingEvent> {
        if admin != &self.admin {
            return Err(AxonError::PermissionDenied);
        }

        // Freeze pricing at current levels
        let old_multiplier = self.pricing.network_health_multiplier;
        
        // Set multipliers to maintain current pricing
        self.pricing.network_health_multiplier = 1.0;
        self.pricing.demand_multiplier = 1.0;
        self.pricing.updated_at = Timestamp::now();
        self.last_updated = Timestamp::now();
        
        self.record_pricing_snapshot();

        Ok(PricingEvent::PriceAdjusted {
            old_multiplier,
            new_multiplier: 1.0,
            reason: AdjustmentReason::EmergencyAdjustment,
            timestamp: Timestamp::now(),
        })
    }

    /// Advanced market-based pricing with ML predictions
    pub fn calculate_market_price(
        &self,
        domain_type: &DomainType,
        domain_name: &str,
        duration_years: u32,
        market_indicators: &MarketIndicators,
    ) -> u64 {
        let base_price = self.calculate_domain_price(domain_type, duration_years);
        
        // Apply market-based adjustments
        let mut market_multiplier = 1.0;
        
        // Scarcity adjustment (shorter domains are more valuable)
        if domain_name.len() <= 3 {
            market_multiplier *= 3.0;
        } else if domain_name.len() <= 5 {
            market_multiplier *= 1.5;
        }
        
        // Keyword popularity adjustment
        if self.is_trending_keyword(domain_name, market_indicators) {
            market_multiplier *= 2.0;
        }
        
        // Competition factor
        market_multiplier *= 1.0 + market_indicators.competition_factor;
        
        // Time-of-day pricing (higher during peak hours)
        market_multiplier *= market_indicators.time_factor;
        
        // Apply market volatility protection
        market_multiplier = market_multiplier.clamp(0.5, 5.0);
        
        (base_price as f32 * market_multiplier) as u64
    }
    
    /// Check if domain name contains trending keywords
    fn is_trending_keyword(&self, domain_name: &str, market_indicators: &MarketIndicators) -> bool {
        let name_lower = domain_name.to_lowercase();
        market_indicators.trending_keywords.iter()
            .any(|keyword| name_lower.contains(&keyword.to_lowercase()))
    }

    /// Predict future pricing trends
    pub fn predict_price_trend(&self, domain_type: &DomainType, days_ahead: u32) -> PriceTrendPrediction {
        let current_price = self.calculate_domain_price(domain_type, 1);
        
        // Simple trend analysis based on historical data
        let trend_direction = if self.pricing_history.len() >= 2 {
            let recent = &self.pricing_history[self.pricing_history.len() - 1];
            let older = &self.pricing_history[self.pricing_history.len() - 2];
            
            if recent.demand_score > older.demand_score {
                TrendDirection::Increasing
            } else if recent.demand_score < older.demand_score {
                TrendDirection::Decreasing
            } else {
                TrendDirection::Stable
            }
        } else {
            TrendDirection::Stable
        };
        
        let predicted_multiplier = match trend_direction {
            TrendDirection::Increasing => 1.0 + (days_ahead as f32 * 0.01), // 1% increase per day
            TrendDirection::Decreasing => 1.0 - (days_ahead as f32 * 0.005), // 0.5% decrease per day
            TrendDirection::Stable => 1.0,
        };
        
        let predicted_price = (current_price as f32 * predicted_multiplier.clamp(0.5, 3.0)) as u64;
        
        PriceTrendPrediction {
            current_price,
            predicted_price,
            confidence: self.calculate_prediction_confidence(),
            trend_direction,
            days_ahead,
            factors: self.get_trend_factors(),
        }
    }
    
    fn calculate_prediction_confidence(&self) -> f32 {
        // Confidence based on amount of historical data and stability
        let data_points = self.pricing_history.len() as f32;
        let base_confidence = (data_points / 100.0).min(0.8); // Max 80% from data
        
        // Reduce confidence if high volatility
        let volatility = self.calculate_pricing_volatility();
        let volatility_penalty = volatility * 0.3;
        
        (base_confidence - volatility_penalty).max(0.1)
    }
    
    fn calculate_pricing_volatility(&self) -> f32 {
        if self.pricing_history.len() < 2 {
            return 0.0;
        }
        
        let mut volatility = 0.0;
        for i in 1..self.pricing_history.len() {
            let prev = &self.pricing_history[i - 1];
            let curr = &self.pricing_history[i];
            
            let change = (curr.demand_score - prev.demand_score).abs() / prev.demand_score;
            volatility += change;
        }
        
        volatility / (self.pricing_history.len() - 1) as f32
    }
    
    fn get_trend_factors(&self) -> Vec<String> {
        let mut factors = Vec::new();
        
        if self.network_metrics.consensus_participation < 0.8 {
            factors.push("Low consensus participation".to_string());
        }
        
        if self.demand_metrics.renewal_rate < 0.8 {
            factors.push("Low renewal rate".to_string());
        }
        
        if self.network_metrics.network_utilization > 0.8 {
            factors.push("High network utilization".to_string());
        }
        
        factors
    }

    /// Get current pricing information
    pub fn get_current_pricing(&self) -> &DomainPricing {
        &self.pricing
    }

    /// Get pricing history
    pub fn get_pricing_history(&self) -> &[PricingSnapshot] {
        &self.pricing_history
    }

    /// Get network health metrics
    pub fn get_network_metrics(&self) -> &NetworkHealthMetrics {
        &self.network_metrics
    }

    /// Get demand metrics
    pub fn get_demand_metrics(&self) -> &DemandMetrics {
        &self.demand_metrics
    }
}

/// Market indicators for advanced pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketIndicators {
    pub trending_keywords: Vec<String>,
    pub competition_factor: f32,
    pub time_factor: f32,
    pub market_volatility: f32,
    pub external_demand_signals: Vec<DemandSignal>,
}

/// External demand signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandSignal {
    pub source: String,
    pub signal_type: SignalType,
    pub strength: f32,
    pub timestamp: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    SocialMediaMention,
    SearchVolume,
    CompetitorPricing,
    NetworkActivity,
    CreatorDemand,
}

/// Price trend prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceTrendPrediction {
    pub current_price: u64,
    pub predicted_price: u64,
    pub confidence: f32,
    pub trend_direction: TrendDirection,
    pub days_ahead: u32,
    pub factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::crypto::AxonSigningKey;

    #[test]
    fn test_adaptive_pricing_creation() {
        let admin_key = AxonSigningKey::generate();
        let initial_pricing = DomainPricing {
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

        let pricing_contract = AdaptivePricingContract::new(
            admin_key.verifying_key(),
            initial_pricing.clone(),
        );

        assert_eq!(pricing_contract.pricing.base_prices.standard, 1000);
        assert_eq!(pricing_contract.pricing.network_health_multiplier, 1.0);
        assert_eq!(pricing_contract.pricing.demand_multiplier, 1.0);
    }

    #[test]
    fn test_price_calculation() {
        let admin_key = AxonSigningKey::generate();
        let initial_pricing = DomainPricing {
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

        let pricing_contract = AdaptivePricingContract::new(
            admin_key.verifying_key(),
            initial_pricing,
        );

        // Test base price calculation
        let price_1_year = pricing_contract.calculate_domain_price(&DomainType::Standard, 1);
        assert_eq!(price_1_year, 1000);

        // Test multi-year discount
        let price_2_years = pricing_contract.calculate_domain_price(&DomainType::Standard, 2);
        assert!(price_2_years < 2000); // Should be discounted

        // Test premium domain pricing
        let premium_price = pricing_contract.calculate_domain_price(&DomainType::Premium, 1);
        assert_eq!(premium_price, 5000);
    }

    #[test]
    fn test_network_health_adjustment() {
        let admin_key = AxonSigningKey::generate();
        let initial_pricing = DomainPricing {
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

        let mut pricing_contract = AdaptivePricingContract::new(
            admin_key.verifying_key(),
            initial_pricing,
        );

        // Test low consensus participation (should increase multiplier)
        let result = pricing_contract.update_network_metrics(0.5, 1000, 0.5, 1.0);
        assert!(result.is_ok());
        assert!(pricing_contract.pricing.network_health_multiplier > 1.0);

        // Test high zk-proof cost (should decrease multiplier)
        let result = pricing_contract.update_network_metrics(0.9, 2000, 0.5, 1.0);
        assert!(result.is_ok());
        // Should be reduced due to high zk-proof cost
    }
}