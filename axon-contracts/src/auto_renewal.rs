//! Auto-renewal smart contract for domain management

use axon_core::{
    domain::{AutoRenewalConfig, FundingSource},
    crypto::AxonVerifyingKey,
    types::{DomainName, Timestamp},
    Result, AxonError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Auto-renewal contract state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoRenewalContract {
    /// Domain renewal configurations
    pub renewals: HashMap<DomainName, DomainRenewalConfig>,
    /// Escrow accounts for renewal funding
    pub escrow_accounts: HashMap<AxonVerifyingKey, EscrowAccount>,
    /// Pending renewal queue
    pub renewal_queue: Vec<PendingRenewal>,
    /// Contract configuration
    pub config: RenewalContractConfig,
    /// Last processing timestamp
    pub last_processed: Timestamp,
}

/// Domain-specific renewal configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainRenewalConfig {
    pub domain: DomainName,
    pub owner: AxonVerifyingKey,
    pub config: AutoRenewalConfig,
    pub next_renewal_check: Timestamp,
    pub renewal_attempts: u32,
    pub last_successful_renewal: Option<Timestamp>,
    pub created_at: Timestamp,
}

/// Escrow account for renewal funding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EscrowAccount {
    pub owner: AxonVerifyingKey,
    pub balance: u64,
    pub reserved_balance: u64, // Amount reserved for pending renewals
    pub domains: Vec<DomainName>,
    pub created_at: Timestamp,
    pub last_updated: Timestamp,
}

/// Pending renewal in processing queue
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingRenewal {
    pub domain: DomainName,
    pub owner: AxonVerifyingKey,
    pub renewal_cost: u64,
    pub funding_source: FundingSource,
    pub scheduled_for: Timestamp,
    pub attempts: u32,
    pub created_at: Timestamp,
}

/// Contract configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenewalContractConfig {
    /// Maximum renewal attempts before giving up
    pub max_renewal_attempts: u32,
    /// Days before expiration to attempt renewal
    pub renewal_window_days: u32,
    /// Processing fee for auto-renewal service
    pub processing_fee_percentage: f32,
    /// Minimum escrow balance required
    pub min_escrow_balance: u64,
    /// Last configuration update
    pub updated_at: Timestamp,
}

/// Auto-renewal events
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AutoRenewalEvent {
    RenewalConfigured {
        domain: DomainName,
        owner: AxonVerifyingKey,
        config: AutoRenewalConfig,
        timestamp: Timestamp,
    },
    RenewalScheduled {
        domain: DomainName,
        scheduled_for: Timestamp,
        estimated_cost: u64,
        timestamp: Timestamp,
    },
    RenewalExecuted {
        domain: DomainName,
        cost: u64,
        funding_source: FundingSource,
        timestamp: Timestamp,
    },
    RenewalFailed {
        domain: DomainName,
        reason: RenewalFailureReason,
        attempts: u32,
        timestamp: Timestamp,
    },
    EscrowDeposit {
        account: AxonVerifyingKey,
        amount: u64,
        new_balance: u64,
        timestamp: Timestamp,
    },
    EscrowWithdrawal {
        account: AxonVerifyingKey,
        amount: u64,
        new_balance: u64,
        timestamp: Timestamp,
    },
}

/// Reasons for renewal failure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RenewalFailureReason {
    InsufficientFunds,
    PaymentFailed,
    DomainNotFound,
    ConfigurationError,
    NetworkError,
    MaxAttemptsReached,
}

impl AutoRenewalContract {
    /// Create new auto-renewal contract
    pub fn new() -> Self {
        Self {
            renewals: HashMap::new(),
            escrow_accounts: HashMap::new(),
            renewal_queue: Vec::new(),
            config: RenewalContractConfig {
                max_renewal_attempts: 3,
                renewal_window_days: 30,
                processing_fee_percentage: 0.05, // 5% processing fee
                min_escrow_balance: 1000,
                updated_at: Timestamp::now(),
            },
            last_processed: Timestamp::now(),
        }
    }

    /// Configure auto-renewal for a domain
    pub fn configure_renewal(
        &mut self,
        domain: DomainName,
        owner: AxonVerifyingKey,
        config: AutoRenewalConfig,
    ) -> Result<AutoRenewalEvent> {
        // Validate configuration
        if config.renewal_duration_years == 0 || config.renewal_duration_years > 10 {
            return Err(AxonError::InvalidDomain(
                "Invalid renewal duration".to_string(),
            ));
        }

        // Check funding source availability
        self.validate_funding_source(&config.funding_source, &owner)?;

        // Calculate next renewal check (30 days before expiration)
        let next_check = Timestamp(Timestamp::now().0 + (self.config.renewal_window_days as u64 * 24 * 3600));

        let renewal_config = DomainRenewalConfig {
            domain: domain.clone(),
            owner: owner.clone(),
            config: config.clone(),
            next_renewal_check: next_check,
            renewal_attempts: 0,
            last_successful_renewal: None,
            created_at: Timestamp::now(),
        };

        self.renewals.insert(domain.clone(), renewal_config);

        Ok(AutoRenewalEvent::RenewalConfigured {
            domain,
            owner,
            config,
            timestamp: Timestamp::now(),
        })
    }

    /// Deposit funds to escrow account
    pub fn deposit_escrow(
        &mut self,
        account: AxonVerifyingKey,
        amount: u64,
        payment_proof: Vec<u8>,
    ) -> Result<AutoRenewalEvent> {
        // Verify payment proof (placeholder)
        if !self.verify_payment(&payment_proof, amount) {
            return Err(AxonError::InvalidDomain(
                "Payment verification failed".to_string(),
            ));
        }

        let escrow = self.escrow_accounts.entry(account.clone()).or_insert_with(|| {
            EscrowAccount {
                owner: account.clone(),
                balance: 0,
                reserved_balance: 0,
                domains: Vec::new(),
                created_at: Timestamp::now(),
                last_updated: Timestamp::now(),
            }
        });

        escrow.balance += amount;
        escrow.last_updated = Timestamp::now();

        Ok(AutoRenewalEvent::EscrowDeposit {
            account,
            amount,
            new_balance: escrow.balance,
            timestamp: Timestamp::now(),
        })
    }

    /// Withdraw funds from escrow account
    pub fn withdraw_escrow(
        &mut self,
        account: AxonVerifyingKey,
        amount: u64,
        owner_signature: Vec<u8>, // Signature verification placeholder
    ) -> Result<AutoRenewalEvent> {
        let escrow = self.escrow_accounts.get_mut(&account)
            .ok_or(AxonError::InvalidDomain("Escrow account not found".to_string()))?;

        let available_balance = escrow.balance - escrow.reserved_balance;
        if available_balance < amount {
            return Err(AxonError::InvalidDomain("Insufficient available balance".to_string()));
        }

        // Verify owner signature (placeholder)
        if !self.verify_signature(&owner_signature, &account) {
            return Err(AxonError::AuthenticationFailed);
        }

        escrow.balance -= amount;
        escrow.last_updated = Timestamp::now();

        Ok(AutoRenewalEvent::EscrowWithdrawal {
            account,
            amount,
            new_balance: escrow.balance,
            timestamp: Timestamp::now(),
        })
    }

    /// Process pending renewals
    pub fn process_renewals(&mut self) -> Result<Vec<AutoRenewalEvent>> {
        let mut events = Vec::new();
        let now = Timestamp::now();

        // Check for domains that need renewal scheduling
        for (domain_name, renewal_config) in &mut self.renewals {
            if now.0 >= renewal_config.next_renewal_check.0 {
                // Schedule renewal
                let estimated_cost = self.estimate_renewal_cost(domain_name)?;
                
                let pending_renewal = PendingRenewal {
                    domain: domain_name.clone(),
                    owner: renewal_config.owner.clone(),
                    renewal_cost: estimated_cost,
                    funding_source: renewal_config.config.funding_source.clone(),
                    scheduled_for: now,
                    attempts: 0,
                    created_at: now,
                };

                self.renewal_queue.push(pending_renewal);

                events.push(AutoRenewalEvent::RenewalScheduled {
                    domain: domain_name.clone(),
                    scheduled_for: now,
                    estimated_cost,
                    timestamp: now,
                });

                // Update next renewal check for next cycle
                renewal_config.next_renewal_check = Timestamp(
                    now.0 + (renewal_config.config.renewal_duration_years as u64 * 365 * 24 * 3600)
                );
            }
        }

        // Process renewal queue
        let mut processed_renewals = Vec::new();
        for (index, pending_renewal) in self.renewal_queue.iter_mut().enumerate() {
            if now.0 >= pending_renewal.scheduled_for.0 {
                match self.execute_renewal(pending_renewal) {
                    Ok(event) => {
                        events.push(event);
                        processed_renewals.push(index);
                    }
                    Err(_) => {
                        pending_renewal.attempts += 1;
                        if pending_renewal.attempts >= self.config.max_renewal_attempts {
                            events.push(AutoRenewalEvent::RenewalFailed {
                                domain: pending_renewal.domain.clone(),
                                reason: RenewalFailureReason::MaxAttemptsReached,
                                attempts: pending_renewal.attempts,
                                timestamp: now,
                            });
                            processed_renewals.push(index);
                        } else {
                            // Reschedule for later
                            pending_renewal.scheduled_for = Timestamp(now.0 + 3600); // Retry in 1 hour
                        }
                    }
                }
            }
        }

        // Remove processed renewals from queue (in reverse order to maintain indices)
        for &index in processed_renewals.iter().rev() {
            self.renewal_queue.remove(index);
        }

        self.last_processed = now;
        Ok(events)
    }

    /// Execute a single renewal
    fn execute_renewal(&mut self, pending_renewal: &PendingRenewal) -> Result<AutoRenewalEvent> {
        // Reserve funds based on funding source
        match &pending_renewal.funding_source {
            FundingSource::Account(account) => {
                let escrow = self.escrow_accounts.get_mut(account)
                    .ok_or(AxonError::InvalidDomain("Escrow account not found".to_string()))?;
                
                let available_balance = escrow.balance - escrow.reserved_balance;
                if available_balance < pending_renewal.renewal_cost {
                    return Err(AxonError::InvalidDomain("Insufficient funds".to_string()));
                }

                // Reserve funds for renewal
                escrow.reserved_balance += pending_renewal.renewal_cost;
            }
            FundingSource::DomainEarnings => {
                // Check domain earnings (placeholder)
                if !self.has_sufficient_domain_earnings(&pending_renewal.domain, pending_renewal.renewal_cost) {
                    return Err(AxonError::InvalidDomain("Insufficient domain earnings".to_string()));
                }
            }
            FundingSource::Escrow => {
                // Use default escrow account logic
                let escrow = self.escrow_accounts.get_mut(&pending_renewal.owner)
                    .ok_or(AxonError::InvalidDomain("Escrow account not found".to_string()))?;
                
                let available_balance = escrow.balance - escrow.reserved_balance;
                if available_balance < pending_renewal.renewal_cost {
                    return Err(AxonError::InvalidDomain("Insufficient escrow funds".to_string()));
                }

                escrow.reserved_balance += pending_renewal.renewal_cost;
            }
        }

        // Execute payment (placeholder - would integrate with domain registry)
        self.process_renewal_payment(pending_renewal)?;

        // Update renewal configuration
        if let Some(renewal_config) = self.renewals.get_mut(&pending_renewal.domain) {
            renewal_config.last_successful_renewal = Some(Timestamp::now());
            renewal_config.renewal_attempts = 0;
        }

        Ok(AutoRenewalEvent::RenewalExecuted {
            domain: pending_renewal.domain.clone(),
            cost: pending_renewal.renewal_cost,
            funding_source: pending_renewal.funding_source.clone(),
            timestamp: Timestamp::now(),
        })
    }

    /// Validate funding source configuration
    fn validate_funding_source(&self, funding_source: &FundingSource, owner: &AxonVerifyingKey) -> Result<()> {
        match funding_source {
            FundingSource::Account(account) => {
                if let Some(escrow) = self.escrow_accounts.get(account) {
                    if escrow.balance < self.config.min_escrow_balance {
                        return Err(AxonError::InvalidDomain(
                            "Insufficient escrow balance".to_string(),
                        ));
                    }
                } else {
                    return Err(AxonError::InvalidDomain(
                        "Escrow account not found".to_string(),
                    ));
                }
            }
            FundingSource::Escrow => {
                if let Some(escrow) = self.escrow_accounts.get(owner) {
                    if escrow.balance < self.config.min_escrow_balance {
                        return Err(AxonError::InvalidDomain(
                            "Insufficient escrow balance".to_string(),
                        ));
                    }
                } else {
                    return Err(AxonError::InvalidDomain(
                        "Owner escrow account not found".to_string(),
                    ));
                }
            }
            FundingSource::DomainEarnings => {
                // Would validate domain has earnings capability
                // Placeholder implementation
            }
        }
        Ok(())
    }

    /// Estimate renewal cost for a domain
    fn estimate_renewal_cost(&self, _domain: &DomainName) -> Result<u64> {
        // Placeholder - would integrate with pricing contract
        Ok(1000)
    }

    /// Check if domain has sufficient earnings for renewal
    fn has_sufficient_domain_earnings(&self, _domain: &DomainName, _required: u64) -> bool {
        // Placeholder - would check domain earnings
        true
    }

    /// Process renewal payment
    fn process_renewal_payment(&mut self, _pending_renewal: &PendingRenewal) -> Result<()> {
        // Placeholder - would integrate with domain registry contract
        Ok(())
    }

    /// Verify payment proof
    fn verify_payment(&self, _payment_proof: &[u8], _amount: u64) -> bool {
        // Placeholder payment verification
        true
    }

    /// Verify signature
    fn verify_signature(&self, _signature: &[u8], _account: &AxonVerifyingKey) -> bool {
        // Placeholder signature verification
        true
    }

    /// Get renewal configuration for a domain
    pub fn get_renewal_config(&self, domain: &DomainName) -> Option<&DomainRenewalConfig> {
        self.renewals.get(domain)
    }

    /// Get escrow account information
    pub fn get_escrow_account(&self, account: &AxonVerifyingKey) -> Option<&EscrowAccount> {
        self.escrow_accounts.get(account)
    }

    /// Get pending renewals
    pub fn get_pending_renewals(&self) -> &[PendingRenewal] {
        &self.renewal_queue
    }

    /// Remove renewal configuration
    pub fn remove_renewal_config(
        &mut self,
        domain: &DomainName,
        owner: &AxonVerifyingKey,
    ) -> Result<()> {
        if let Some(config) = self.renewals.get(domain) {
            if &config.owner != owner {
                return Err(AxonError::PermissionDenied);
            }
            self.renewals.remove(domain);
            Ok(())
        } else {
            Err(AxonError::DomainNotFound)
        }
    }
}

impl Default for AutoRenewalContract {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::crypto::AxonSigningKey;

    #[test]
    fn test_auto_renewal_contract_creation() {
        let contract = AutoRenewalContract::new();
        assert_eq!(contract.config.max_renewal_attempts, 3);
        assert_eq!(contract.config.renewal_window_days, 30);
        assert!(contract.renewals.is_empty());
        assert!(contract.escrow_accounts.is_empty());
    }

    #[test]
    fn test_escrow_deposit_withdrawal() {
        let mut contract = AutoRenewalContract::new();
        let user_key = AxonSigningKey::generate();
        let account = user_key.verifying_key();

        // Test deposit
        let deposit_result = contract.deposit_escrow(account.clone(), 5000, vec![]);
        assert!(deposit_result.is_ok());

        let escrow = contract.get_escrow_account(&account).unwrap();
        assert_eq!(escrow.balance, 5000);

        // Test withdrawal
        let withdraw_result = contract.withdraw_escrow(account.clone(), 2000, vec![]);
        assert!(withdraw_result.is_ok());

        let escrow = contract.get_escrow_account(&account).unwrap();
        assert_eq!(escrow.balance, 3000);
    }

    #[test]
    fn test_renewal_configuration() {
        let mut contract = AutoRenewalContract::new();
        let user_key = AxonSigningKey::generate();
        let account = user_key.verifying_key();
        let domain = DomainName::new("testdomain".to_string()).unwrap();

        // Setup escrow account first
        let _deposit_result = contract.deposit_escrow(account.clone(), 10000, vec![]);

        let config = AutoRenewalConfig {
            enabled: true,
            renewal_duration_years: 1,
            max_renewal_price: 2000,
            funding_source: FundingSource::Escrow,
        };

        let result = contract.configure_renewal(domain.clone(), account.clone(), config.clone());
        assert!(result.is_ok());

        let renewal_config = contract.get_renewal_config(&domain).unwrap();
        assert_eq!(renewal_config.config.renewal_duration_years, 1);
        assert_eq!(renewal_config.owner, account);
    }
}