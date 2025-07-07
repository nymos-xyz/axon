//! Community Funding and Crowdfunding System

use crate::error::{CreatorEconomyError, CreatorEconomyResult};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::time::Duration;

/// Community funding system
#[derive(Debug)]
pub struct CommunityFunding {
    active_campaigns: HashMap<String, CrowdfundingCampaign>,
    anonymous_backers: HashMap<String, AnonymousBacker>,
    funding_analytics: CampaignAnalytics,
}

/// Crowdfunding campaign
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrowdfundingCampaign {
    pub campaign_id: String,
    pub creator_id: String,
    pub title: String,
    pub description: String,
    pub funding_goal: FundingGoal,
    pub current_funding: u64,
    pub backer_count: u32,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub campaign_status: CampaignStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CampaignStatus {
    Planning,
    Active,
    Successful,
    Failed,
    Cancelled,
}

/// Anonymous backer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymousBacker {
    pub backer_id: String,
    pub total_contributions: u64,
    pub campaigns_backed: u32,
    pub privacy_level: PrivacyLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyLevel {
    Anonymous,
    Pseudonymous,
    Public,
}

/// Funding goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingGoal {
    pub target_amount: u64,
    pub minimum_funding: u64,
    pub stretch_goals: Vec<StretchGoal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StretchGoal {
    pub amount: u64,
    pub description: String,
    pub rewards: Vec<String>,
}

/// Campaign analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignAnalytics {
    pub total_campaigns: u32,
    pub success_rate: f64,
    pub average_funding_amount: f64,
    pub total_funds_raised: u64,
}

impl CommunityFunding {
    pub fn new() -> Self {
        Self {
            active_campaigns: HashMap::new(),
            anonymous_backers: HashMap::new(),
            funding_analytics: CampaignAnalytics::new(),
        }
    }

    pub async fn create_campaign(&mut self, campaign: CrowdfundingCampaign) -> CreatorEconomyResult<String> {
        let campaign_id = campaign.campaign_id.clone();
        self.active_campaigns.insert(campaign_id.clone(), campaign);
        Ok(campaign_id)
    }

    pub async fn back_campaign(&mut self, campaign_id: &str, backer: AnonymousBacker, amount: u64) -> CreatorEconomyResult<()> {
        if let Some(campaign) = self.active_campaigns.get_mut(campaign_id) {
            campaign.current_funding += amount;
            campaign.backer_count += 1;
        }
        
        self.anonymous_backers.insert(backer.backer_id.clone(), backer);
        Ok(())
    }
}

impl CampaignAnalytics {
    fn new() -> Self {
        Self {
            total_campaigns: 0,
            success_rate: 0.0,
            average_funding_amount: 0.0,
            total_funds_raised: 0,
        }
    }
}