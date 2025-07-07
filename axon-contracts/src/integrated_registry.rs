//! Integrated domain registry with advanced features
//! 
//! This contract integrates:
//! - Domain registration with adaptive pricing
//! - Revenue distribution and token burning
//! - Governance integration
//! - Auto-renewal system
//! - Creator economy features

use crate::{
    domain_registry::{DomainRegistryContract, RevenueDistribution, DynamicFeeCalculation},
    pricing::{AdaptivePricingContract, MarketIndicators, PriceTrendPrediction},
    governance::{GovernanceContract, ProposalAction, GovernanceEvent},
    auto_renewal::{AutoRenewalContract, AutoRenewalEvent},
    creator_economy::CreatorEconomyContract,
};

use axon_core::{
    domain::{DomainRegistrationRequest, DomainRecord, DomainPricing},
    types::{DomainName, DomainType, Timestamp},
    crypto::AxonVerifyingKey,
    Result, AxonError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, debug};

/// Fully integrated domain registry system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegratedRegistryContract {
    /// Core domain registry
    pub domain_registry: DomainRegistryContract,
    /// Adaptive pricing system
    pub pricing_engine: AdaptivePricingContract,
    /// Governance system
    pub governance: GovernanceContract,
    /// Auto-renewal system
    pub auto_renewal: AutoRenewalContract,
    /// Creator economy features
    pub creator_economy: CreatorEconomyContract,
    /// Revenue tracking and distribution
    pub revenue_system: RevenueSystem,
    /// Market analysis engine
    pub market_engine: MarketAnalysisEngine,
    /// Integration statistics
    pub integration_stats: IntegrationStats,
}

/// Advanced revenue distribution system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RevenueSystem {
    /// Total revenue collected
    pub total_revenue: u64,
    /// Total tokens burned
    pub total_burned: u64,
    /// Revenue distribution history
    pub distribution_history: Vec<RevenueDistribution>,
    /// Burn schedule configuration
    pub burn_config: BurnConfiguration,
    /// Revenue allocation rules
    pub allocation_rules: RevenueAllocationRules,
    /// Monthly revenue targets
    pub monthly_targets: HashMap<String, MonthlyTarget>,
}

/// Token burning configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnConfiguration {
    /// Percentage of revenue to burn
    pub burn_percentage: f64,
    /// Minimum burn amount per transaction
    pub min_burn_amount: u64,
    /// Maximum burn amount per transaction
    pub max_burn_amount: u64,
    /// Burn frequency (seconds)
    pub burn_interval: u64,
    /// Last burn timestamp
    pub last_burn: Option<Timestamp>,
    /// Emergency burn multiplier
    pub emergency_multiplier: f64,
}

/// Revenue allocation rules
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RevenueAllocationRules {
    /// Development fund percentage
    pub development_percentage: f64,
    /// Ecosystem fund percentage
    pub ecosystem_percentage: f64,
    /// Validator rewards percentage
    pub validator_percentage: f64,
    /// Creator rewards percentage
    pub creator_percentage: f64,
    /// Token burn percentage
    pub burn_percentage: f64,
    /// Governance treasury percentage
    pub governance_percentage: f64,
}

/// Monthly revenue targets and tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonthlyTarget {
    /// Target revenue for the month
    pub target_amount: u64,
    /// Actual revenue collected
    pub actual_amount: u64,
    /// Number of domains registered
    pub domains_registered: u32,
    /// Average domain price
    pub average_price: u64,
    /// Month identifier (YYYY-MM)
    pub month: String,
}

/// Market analysis engine for pricing optimization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketAnalysisEngine {
    /// Current market indicators
    pub market_indicators: MarketIndicators,
    /// Trending domain keywords
    pub trending_keywords: Vec<TrendingKeyword>,
    /// Price history analysis
    pub price_analysis: PriceAnalysisData,
    /// Competitor pricing data
    pub competitor_data: Vec<CompetitorPricing>,
    /// Market predictions
    pub predictions: Vec<MarketPrediction>,
}

/// Trending keyword data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrendingKeyword {
    pub keyword: String,
    pub trend_score: f64,
    pub search_volume: u64,
    pub price_impact: f64,
    pub detected_at: Timestamp,
}

/// Price analysis data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriceAnalysisData {
    pub average_price_30d: u64,
    pub price_volatility: f64,
    pub demand_trend: DemandTrend,
    pub seasonal_factors: Vec<SeasonalFactor>,
    pub price_elasticity: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DemandTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeasonalFactor {
    pub month: u8,
    pub multiplier: f64,
    pub confidence: f64,
}

/// Competitor pricing information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompetitorPricing {
    pub competitor_name: String,
    pub domain_type: String,
    pub price: u64,
    pub features: Vec<String>,
    pub last_updated: Timestamp,
}

/// Market prediction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketPrediction {
    pub prediction_type: PredictionType,
    pub timeframe_days: u32,
    pub predicted_value: f64,
    pub confidence: f64,
    pub factors: Vec<String>,
    pub created_at: Timestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PredictionType {
    PriceMovement,
    DemandVolume,
    MarketShare,
    RevenueTarget,
}

/// Integration statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegrationStats {
    pub total_transactions: u64,
    pub successful_registrations: u64,
    pub failed_registrations: u64,
    pub governance_proposals: u32,
    pub auto_renewals: u32,
    pub creator_subscriptions: u32,
    pub average_response_time_ms: u64,
    pub system_uptime_percentage: f64,
}

/// Integrated registry events
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IntegratedRegistryEvent {
    DomainRegistered {
        domain: DomainName,
        owner: AxonVerifyingKey,
        price_paid: u64,
        revenue_distribution: RevenueDistribution,
        timestamp: Timestamp,
    },
    PriceAdjustment {
        old_price: u64,
        new_price: u64,
        reason: String,
        affected_domains: u32,
    },
    RevenueDistributed {
        total_amount: u64,
        burned_amount: u64,
        distribution: RevenueDistribution,
    },
    GovernanceProposal {
        proposal_id: u64,
        proposal_type: String,
        impact_assessment: ImpactAssessment,
    },
    MarketAnalysisUpdate {
        trending_keywords: Vec<String>,
        price_predictions: Vec<PriceTrendPrediction>,
        market_health_score: f64,
    },
}

/// Impact assessment for governance proposals
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub financial_impact: f64,
    pub user_impact: UserImpactLevel,
    pub technical_complexity: TechnicalComplexity,
    pub risk_level: RiskLevel,
    pub estimated_implementation_time: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UserImpactLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TechnicalComplexity {
    Simple,
    Moderate,
    Complex,
    HighlyComplex,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl IntegratedRegistryContract {
    /// Create new integrated registry system
    pub fn new(
        admin: AxonVerifyingKey,
        initial_pricing: DomainPricing,
        governance_config: crate::governance::GovernanceConfig,
        emergency_council: std::collections::HashSet<AxonVerifyingKey>,
    ) -> Self {
        Self {
            domain_registry: DomainRegistryContract::new(admin.clone(), initial_pricing.clone()),
            pricing_engine: AdaptivePricingContract::new(admin.clone(), initial_pricing),
            governance: GovernanceContract::new(governance_config, emergency_council),
            auto_renewal: AutoRenewalContract::new(),
            creator_economy: CreatorEconomyContract::new(),
            revenue_system: RevenueSystem {
                total_revenue: 0,
                total_burned: 0,
                distribution_history: Vec::new(),
                burn_config: BurnConfiguration {
                    burn_percentage: 0.15, // 15% burn rate
                    min_burn_amount: 100,
                    max_burn_amount: 100_000,
                    burn_interval: 86400, // Daily burns
                    last_burn: None,
                    emergency_multiplier: 2.0,
                },
                allocation_rules: RevenueAllocationRules {
                    development_percentage: 0.25,
                    ecosystem_percentage: 0.20,
                    validator_percentage: 0.30,
                    creator_percentage: 0.10,
                    burn_percentage: 0.15,
                    governance_percentage: 0.05, // Treasury reserves
                },
                monthly_targets: HashMap::new(),
            },
            market_engine: MarketAnalysisEngine {
                market_indicators: MarketIndicators {
                    trending_keywords: Vec::new(),
                    competition_factor: 1.0,
                    time_factor: 1.0,
                    market_volatility: 0.1,
                    external_demand_signals: Vec::new(),
                },
                trending_keywords: Vec::new(),
                price_analysis: PriceAnalysisData {
                    average_price_30d: 1000,
                    price_volatility: 0.1,
                    demand_trend: DemandTrend::Stable,
                    seasonal_factors: Vec::new(),
                    price_elasticity: 1.0,
                },
                competitor_data: Vec::new(),
                predictions: Vec::new(),
            },
            integration_stats: IntegrationStats {
                total_transactions: 0,
                successful_registrations: 0,
                failed_registrations: 0,
                governance_proposals: 0,
                auto_renewals: 0,
                creator_subscriptions: 0,
                average_response_time_ms: 100,
                system_uptime_percentage: 99.9,
            },
        }
    }

    /// Register domain with full integration
    pub fn register_domain_integrated(
        &mut self,
        request: DomainRegistrationRequest,
        market_analysis: bool,
    ) -> Result<IntegratedRegistryEvent> {
        let start_time = std::time::Instant::now();
        
        // Calculate dynamic pricing
        let fee_calculation = self.domain_registry.calculate_dynamic_fee(
            &request.domain_type,
            request.domain_name.as_str(),
            request.duration_years,
            &request.owner,
            None,
        );

        // Apply market-based pricing if requested
        let final_price = if market_analysis {
            self.pricing_engine.calculate_market_price(
                &request.domain_type,
                request.domain_name.as_str(),
                request.duration_years,
                &self.market_engine.market_indicators,
            )
        } else {
            fee_calculation.final_cost
        };

        // Register the domain
        let registration_result = self.domain_registry.register_domain(request.clone());

        match registration_result {
            Ok(_) => {
                // Process revenue distribution
                let revenue_distribution = self.process_revenue_distribution(final_price)?;
                
                // Update market analysis
                self.update_market_data(&request.domain_name, final_price);
                
                // Update statistics
                self.integration_stats.total_transactions += 1;
                self.integration_stats.successful_registrations += 1;
                self.integration_stats.average_response_time_ms = 
                    (self.integration_stats.average_response_time_ms + start_time.elapsed().as_millis() as u64) / 2;

                info!(
                    "Domain {} registered for {} tokens with revenue distribution",
                    request.domain_name.as_str(),
                    final_price
                );

                Ok(IntegratedRegistryEvent::DomainRegistered {
                    domain: request.domain_name,
                    owner: request.owner,
                    price_paid: final_price,
                    revenue_distribution,
                    timestamp: Timestamp::now(),
                })
            }
            Err(e) => {
                self.integration_stats.failed_registrations += 1;
                warn!("Domain registration failed: {:?}", e);
                Err(e)
            }
        }
    }

    /// Process comprehensive revenue distribution
    fn process_revenue_distribution(&mut self, amount: u64) -> Result<RevenueDistribution> {
        let rules = &self.revenue_system.allocation_rules;
        
        let development_share = (amount as f64 * rules.development_percentage) as u64;
        let ecosystem_share = (amount as f64 * rules.ecosystem_percentage) as u64;
        let validator_share = (amount as f64 * rules.validator_percentage) as u64;
        let creator_share = (amount as f64 * rules.creator_percentage) as u64;
        let burn_amount = (amount as f64 * rules.burn_percentage) as u64;
        let governance_share = (amount as f64 * rules.governance_percentage) as u64;

        let distribution = RevenueDistribution {
            total_amount: amount,
            development_fund: development_share,
            ecosystem_fund: ecosystem_share,
            burned_tokens: burn_amount,
            validator_rewards: validator_share,
            creator_rewards: creator_share,
            timestamp: Timestamp::now(),
        };

        // Update revenue tracking
        self.revenue_system.total_revenue += amount;
        self.revenue_system.total_burned += burn_amount;
        self.revenue_system.distribution_history.push(distribution.clone());

        // Schedule token burn if needed
        self.schedule_token_burn(burn_amount)?;

        info!(
            "Revenue distributed: {} total, {} burned, {} to development, {} to ecosystem",
            amount, burn_amount, development_share, ecosystem_share
        );

        Ok(distribution)
    }

    /// Schedule token burning
    fn schedule_token_burn(&mut self, amount: u64) -> Result<()> {
        let config = &mut self.revenue_system.burn_config;
        let now = Timestamp::now();

        // Check if it's time for scheduled burn
        let should_burn = match config.last_burn {
            Some(last) => now.0 >= last.0 + config.burn_interval,
            None => true,
        };

        if should_burn && amount >= config.min_burn_amount {
            let burn_amount = amount.min(config.max_burn_amount);
            
            // Execute burn (placeholder - would integrate with token contract)
            self.execute_token_burn(burn_amount)?;
            
            config.last_burn = Some(now);
            
            info!("Scheduled token burn: {} tokens", burn_amount);
        }

        Ok(())
    }

    /// Execute token burn
    fn execute_token_burn(&self, amount: u64) -> Result<()> {
        // Placeholder for actual token burning logic
        info!("Burning {} tokens", amount);
        Ok(())
    }

    /// Update market analysis data
    fn update_market_data(&mut self, domain_name: &DomainName, price: u64) {
        // Update price analysis
        let analysis = &mut self.market_engine.price_analysis;
        analysis.average_price_30d = (analysis.average_price_30d + price) / 2;
        
        // Detect trending keywords
        let name = domain_name.as_str().to_lowercase();
        let potential_keywords = ["ai", "crypto", "nft", "defi", "web3", "dao", "meta"];
        
        for &keyword in &potential_keywords {
            if name.contains(keyword) {
                // Update or add trending keyword
                if let Some(trending) = self.market_engine.trending_keywords
                    .iter_mut()
                    .find(|t| t.keyword == keyword) {
                    trending.trend_score += 0.1;
                    trending.search_volume += 1;
                } else {
                    self.market_engine.trending_keywords.push(TrendingKeyword {
                        keyword: keyword.to_string(),
                        trend_score: 1.0,
                        search_volume: 1,
                        price_impact: 1.2,
                        detected_at: Timestamp::now(),
                    });
                }
                break;
            }
        }
    }

    /// Create governance proposal for pricing changes
    pub fn create_pricing_proposal(
        &mut self,
        proposer: AxonVerifyingKey,
        title: String,
        description: String,
        new_base_prices: Option<axon_core::domain::DomainTypePricing>,
        network_multiplier: Option<f32>,
        demand_multiplier: Option<f32>,
        proposer_voting_power: u64,
    ) -> Result<GovernanceEvent> {
        let actions = vec![ProposalAction::UpdatePricing {
            contract_address: "pricing_engine".to_string(),
            new_parameters: {
                let mut params = HashMap::new();
                if let Some(prices) = &new_base_prices {
                    params.insert("standard_price".to_string(), prices.standard.to_string());
                    params.insert("premium_price".to_string(), prices.premium.to_string());
                }
                if let Some(multiplier) = network_multiplier {
                    params.insert("network_multiplier".to_string(), multiplier.to_string());
                }
                if let Some(multiplier) = demand_multiplier {
                    params.insert("demand_multiplier".to_string(), multiplier.to_string());
                }
                params
            },
        }];

        let metadata = crate::governance::ProposalMetadata {
            category: "pricing".to_string(),
            urgency: crate::governance::UrgencyLevel::Medium,
            estimated_impact: crate::governance::ImpactLevel::High,
            required_expertise: vec!["economics".to_string(), "pricing".to_string()],
            external_links: Vec::new(),
            discussion_forum: Some("https://forum.axon.network/pricing".to_string()),
        };

        let result = self.governance.create_proposal(
            proposer,
            title,
            description,
            crate::governance::ProposalType::ParameterUpdate,
            actions,
            metadata,
            proposer_voting_power,
        );

        if result.is_ok() {
            self.integration_stats.governance_proposals += 1;
        }

        result
    }

    /// Get comprehensive system statistics
    pub fn get_system_statistics(&self) -> SystemStatistics {
        let domain_stats = self.domain_registry.get_stats();
        let governance_stats = self.governance.get_governance_stats();
        let pricing_metrics = self.pricing_engine.get_current_pricing();
        
        SystemStatistics {
            domain_registry: domain_stats,
            governance: governance_stats,
            revenue_system: self.revenue_system.clone(),
            market_analysis: self.market_engine.price_analysis.clone(),
            integration_stats: self.integration_stats.clone(),
            pricing_metrics: pricing_metrics.clone(),
            system_health: self.calculate_system_health(),
        }
    }

    /// Calculate overall system health score
    fn calculate_system_health(&self) -> SystemHealthScore {
        let registration_success_rate = if self.integration_stats.total_transactions > 0 {
            self.integration_stats.successful_registrations as f64 / self.integration_stats.total_transactions as f64
        } else {
            1.0
        };

        let revenue_health = if self.revenue_system.total_revenue > 0 {
            1.0 - (self.revenue_system.total_burned as f64 / self.revenue_system.total_revenue as f64).min(0.5)
        } else {
            1.0
        };

        let governance_participation = if self.governance.proposals.len() > 0 {
            let active_proposals = self.governance.get_active_proposals().len() as f64;
            (active_proposals / self.governance.proposals.len() as f64).min(1.0)
        } else {
            0.5
        };

        let overall_score = (registration_success_rate + revenue_health + governance_participation) / 3.0;

        SystemHealthScore {
            overall_score,
            registration_success_rate,
            revenue_health_score: revenue_health,
            governance_participation_rate: governance_participation,
            uptime_percentage: self.integration_stats.system_uptime_percentage,
            last_calculated: Timestamp::now(),
        }
    }

    /// Predict future trends
    pub fn generate_market_predictions(&mut self, days_ahead: u32) -> Vec<MarketPrediction> {
        let mut predictions = Vec::new();

        // Price movement prediction
        let price_trend = self.pricing_engine.predict_price_trend(&DomainType::Standard, days_ahead);
        predictions.push(MarketPrediction {
            prediction_type: PredictionType::PriceMovement,
            timeframe_days: days_ahead,
            predicted_value: price_trend.predicted_price as f64,
            confidence: price_trend.confidence as f64,
            factors: price_trend.factors,
            created_at: Timestamp::now(),
        });

        // Demand volume prediction based on trending keywords
        let demand_multiplier = self.market_engine.trending_keywords.iter()
            .map(|k| k.trend_score)
            .fold(0.0, |acc, score| acc + score) / self.market_engine.trending_keywords.len() as f64;
        
        predictions.push(MarketPrediction {
            prediction_type: PredictionType::DemandVolume,
            timeframe_days: days_ahead,
            predicted_value: demand_multiplier,
            confidence: 0.7,
            factors: vec!["keyword_trends".to_string(), "historical_demand".to_string()],
            created_at: Timestamp::now(),
        });

        self.market_engine.predictions = predictions.clone();
        predictions
    }
}

/// Comprehensive system statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemStatistics {
    pub domain_registry: crate::domain_registry::DomainRegistryStats,
    pub governance: crate::governance::GovernanceStats,
    pub revenue_system: RevenueSystem,
    pub market_analysis: PriceAnalysisData,
    pub integration_stats: IntegrationStats,
    pub pricing_metrics: DomainPricing,
    pub system_health: SystemHealthScore,
}

/// System health scoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemHealthScore {
    pub overall_score: f64,
    pub registration_success_rate: f64,
    pub revenue_health_score: f64,
    pub governance_participation_rate: f64,
    pub uptime_percentage: f64,
    pub last_calculated: Timestamp,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::crypto::AxonSigningKey;
    use axon_core::domain::{DomainMetadata, VerificationStatus, DomainTypePricing};

    #[test]
    fn test_integrated_registry_creation() {
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

        let governance_config = crate::governance::GovernanceConfig {
            proposal_threshold: 10_000,
            quorum_threshold: 0.1,
            voting_period: 604800,
            timelock_period: 172800,
            execution_window: 259200,
            quadratic_voting: true,
            max_voting_power: 100_000,
            delegation_fee: 0.01,
        };

        let registry = IntegratedRegistryContract::new(
            admin_key.verifying_key(),
            pricing,
            governance_config,
            std::collections::HashSet::new(),
        );

        assert_eq!(registry.revenue_system.burn_config.burn_percentage, 0.15);
        assert_eq!(registry.market_engine.price_analysis.average_price_30d, 1000);
        assert_eq!(registry.integration_stats.total_transactions, 0);
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

        let governance_config = crate::governance::GovernanceConfig {
            proposal_threshold: 10_000,
            quorum_threshold: 0.1,
            voting_period: 604800,
            timelock_period: 172800,
            execution_window: 259200,
            quadratic_voting: true,
            max_voting_power: 100_000,
            delegation_fee: 0.01,
        };

        let mut registry = IntegratedRegistryContract::new(
            admin_key.verifying_key(),
            pricing,
            governance_config,
            std::collections::HashSet::new(),
        );

        let distribution = registry.process_revenue_distribution(10000).unwrap();

        assert_eq!(distribution.total_amount, 10000);
        assert_eq!(distribution.development_fund, 2500);  // 25%
        assert_eq!(distribution.ecosystem_fund, 2000);    // 20%
        assert_eq!(distribution.burned_tokens, 1500);     // 15%
        assert_eq!(distribution.validator_rewards, 3000); // 30%
        assert_eq!(distribution.creator_rewards, 1000);   // 10%

        assert_eq!(registry.revenue_system.total_revenue, 10000);
        assert_eq!(registry.revenue_system.total_burned, 1500);
    }

    #[test]
    fn test_system_health_calculation() {
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

        let governance_config = crate::governance::GovernanceConfig {
            proposal_threshold: 10_000,
            quorum_threshold: 0.1,
            voting_period: 604800,
            timelock_period: 172800,
            execution_window: 259200,
            quadratic_voting: true,
            max_voting_power: 100_000,
            delegation_fee: 0.01,
        };

        let mut registry = IntegratedRegistryContract::new(
            admin_key.verifying_key(),
            pricing,
            governance_config,
            std::collections::HashSet::new(),
        );

        // Simulate some activity
        registry.integration_stats.total_transactions = 100;
        registry.integration_stats.successful_registrations = 95;
        registry.revenue_system.total_revenue = 100_000;
        registry.revenue_system.total_burned = 15_000;

        let health = registry.calculate_system_health();

        assert_eq!(health.registration_success_rate, 0.95);
        assert!(health.overall_score > 0.5);
        assert!(health.revenue_health_score > 0.5);
    }
}