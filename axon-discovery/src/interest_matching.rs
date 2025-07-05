use crate::{
    error::{DiscoveryError, Result},
    types::*,
};
use std::collections::HashMap;
use tracing::{info, debug};
use chrono::Utc;

pub struct InterestMatcher {
    interest_categories: HashMap<String, InterestCategory>,
    similarity_cache: HashMap<String, f64>,
    config: DiscoveryConfig,
}

#[derive(Debug, Clone)]
struct InterestCategory {
    name: String,
    subcategories: Vec<String>,
    semantic_embeddings: Vec<f64>,
    popularity_score: f64,
}

impl InterestMatcher {
    pub fn new(config: DiscoveryConfig) -> Self {
        info!("Initializing privacy-preserving interest matching system");
        
        let mut interest_categories = HashMap::new();
        
        let categories = vec![
            ("technology", vec!["programming", "ai", "blockchain", "privacy"], vec![0.8, 0.2, 0.9, 0.7]),
            ("social", vec!["networking", "community", "discussion", "sharing"], vec![0.6, 0.8, 0.5, 0.7]),
            ("content", vec!["media", "articles", "videos", "podcasts"], vec![0.7, 0.6, 0.8, 0.5]),
            ("privacy", vec!["anonymity", "security", "encryption", "decentralization"], vec![0.9, 0.8, 0.7, 0.85]),
        ];
        
        for (name, subcats, embeddings) in categories {
            interest_categories.insert(name.to_string(), InterestCategory {
                name: name.to_string(),
                subcategories: subcats.into_iter().map(|s| s.to_string()).collect(),
                semantic_embeddings: embeddings,
                popularity_score: 0.5,
            });
        }
        
        Self {
            interest_categories,
            similarity_cache: HashMap::new(),
            config,
        }
    }

    pub async fn match_interests_to_content(
        &self,
        user_interests: &[Interest],
        content_metadata: &HashMap<String, String>,
    ) -> Result<f64> {
        info!("Matching user interests to content with privacy preservation");

        let content_interests = self.extract_content_interests(content_metadata)?;
        
        let similarity_score = self.compute_privacy_preserving_similarity(
            user_interests,
            &content_interests,
        ).await?;

        Ok(similarity_score)
    }

    pub async fn find_similar_interests(
        &self,
        target_interests: &[Interest],
        candidate_pool: &[Interest],
        anonymity_threshold: usize,
    ) -> Result<Vec<Interest>> {
        info!("Finding similar interests with k-anonymity guarantee");

        if candidate_pool.len() < anonymity_threshold {
            return Err(DiscoveryError::InterestMatchingError(
                format!("Candidate pool too small for k-anonymity (need {}, got {})", 
                        anonymity_threshold, candidate_pool.len())
            ));
        }

        let mut similarities = Vec::new();
        
        for candidate in candidate_pool {
            let similarity = self.compute_interest_similarity(target_interests, &[candidate.clone()]).await?;
            similarities.push((candidate.clone(), similarity));
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let k_anonymous_results = similarities.into_iter()
            .take(anonymity_threshold.max(10))
            .map(|(interest, _)| interest)
            .collect();

        Ok(k_anonymous_results)
    }

    pub async fn create_anonymous_interest_profile(
        &self,
        interests: &[Interest],
        privacy_level: PrivacyLevel,
    ) -> Result<Vec<Interest>> {
        info!("Creating anonymous interest profile");

        let mut anonymous_interests = Vec::new();
        
        for interest in interests {
            let mut anonymous_interest = interest.clone();
            
            match privacy_level {
                PrivacyLevel::Anonymous => {
                    anonymous_interest.privacy_masked = true;
                    anonymous_interest.weight = self.add_privacy_noise(interest.weight, 0.1);
                    anonymous_interest = self.generalize_interest_category(anonymous_interest)?;
                },
                PrivacyLevel::Pseudonymous => {
                    anonymous_interest.weight = self.add_privacy_noise(interest.weight, 0.05);
                },
                PrivacyLevel::Private => {
                },
            }
            
            anonymous_interests.push(anonymous_interest);
        }

        Ok(anonymous_interests)
    }

    pub async fn compute_interest_diversity_score(&self, interests: &[Interest]) -> Result<f64> {
        if interests.is_empty() {
            return Ok(0.0);
        }

        let categories: std::collections::HashSet<_> = interests.iter()
            .map(|i| &i.category)
            .collect();
        
        let diversity_score = categories.len() as f64 / self.interest_categories.len() as f64;
        
        Ok(diversity_score.min(1.0))
    }

    async fn compute_privacy_preserving_similarity(
        &self,
        user_interests: &[Interest],
        content_interests: &[Interest],
    ) -> Result<f64> {
        let base_similarity = self.compute_interest_similarity(user_interests, content_interests).await?;
        
        let noisy_similarity = self.add_privacy_noise(base_similarity, 0.05);
        
        Ok(noisy_similarity.clamp(0.0, 1.0))
    }

    async fn compute_interest_similarity(
        &self,
        interests_a: &[Interest],
        interests_b: &[Interest],
    ) -> Result<f64> {
        if interests_a.is_empty() || interests_b.is_empty() {
            return Ok(0.0);
        }

        let mut total_similarity = 0.0;
        let mut total_weight = 0.0;

        for interest_a in interests_a {
            for interest_b in interests_b {
                let category_similarity = self.compute_category_similarity(&interest_a.category, &interest_b.category)?;
                
                let subcategory_similarity = if let (Some(sub_a), Some(sub_b)) = (&interest_a.subcategory, &interest_b.subcategory) {
                    self.compute_subcategory_similarity(sub_a, sub_b)?
                } else {
                    0.5
                };

                let weight_similarity = 1.0 - (interest_a.weight - interest_b.weight).abs();
                
                let combined_similarity = 0.5 * category_similarity + 0.3 * subcategory_similarity + 0.2 * weight_similarity;
                
                let combined_weight = interest_a.weight * interest_b.weight;
                
                total_similarity += combined_similarity * combined_weight;
                total_weight += combined_weight;
            }
        }

        let final_similarity = if total_weight > 0.0 {
            total_similarity / total_weight
        } else {
            0.0
        };

        Ok(final_similarity)
    }

    fn compute_category_similarity(&self, category_a: &str, category_b: &str) -> Result<f64> {
        if category_a == category_b {
            return Ok(1.0);
        }

        let cache_key = format!("{}:{}", category_a, category_b);
        if let Some(&cached_similarity) = self.similarity_cache.get(&cache_key) {
            return Ok(cached_similarity);
        }

        let similarity = if let (Some(cat_a), Some(cat_b)) = (
            self.interest_categories.get(category_a),
            self.interest_categories.get(category_b)
        ) {
            self.compute_semantic_similarity(&cat_a.semantic_embeddings, &cat_b.semantic_embeddings)
        } else {
            0.0
        };

        Ok(similarity)
    }

    fn compute_subcategory_similarity(&self, sub_a: &str, sub_b: &str) -> Result<f64> {
        if sub_a == sub_b {
            Ok(1.0)
        } else {
            let edit_distance = self.levenshtein_distance(sub_a, sub_b) as f64;
            let max_len = sub_a.len().max(sub_b.len()) as f64;
            Ok(1.0 - (edit_distance / max_len))
        }
    }

    fn compute_semantic_similarity(&self, embedding_a: &[f64], embedding_b: &[f64]) -> f64 {
        if embedding_a.len() != embedding_b.len() {
            return 0.0;
        }

        let dot_product: f64 = embedding_a.iter()
            .zip(embedding_b.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f64 = embedding_a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = embedding_b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        if len1 == 0 { return len2; }
        if len2 == 0 { return len1; }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    fn extract_content_interests(&self, content_metadata: &HashMap<String, String>) -> Result<Vec<Interest>> {
        let mut interests = Vec::new();

        if let Some(tags) = content_metadata.get("tags") {
            let tag_list: Vec<&str> = tags.split(',').collect();
            
            for tag in tag_list {
                let trimmed_tag = tag.trim();
                if let Some(category) = self.map_tag_to_category(trimmed_tag) {
                    interests.push(Interest {
                        category: category.clone(),
                        subcategory: Some(trimmed_tag.to_string()),
                        weight: 0.5,
                        privacy_masked: false,
                    });
                }
            }
        }

        if let Some(content_type) = content_metadata.get("content_type") {
            if let Some(category) = self.map_content_type_to_category(content_type) {
                interests.push(Interest {
                    category: category.clone(),
                    subcategory: Some(content_type.clone()),
                    weight: 0.7,
                    privacy_masked: false,
                });
            }
        }

        Ok(interests)
    }

    fn map_tag_to_category(&self, tag: &str) -> Option<String> {
        for (category_name, category) in &self.interest_categories {
            if category.subcategories.contains(&tag.to_string()) {
                return Some(category_name.clone());
            }
        }
        None
    }

    fn map_content_type_to_category(&self, content_type: &str) -> Option<String> {
        match content_type.to_lowercase().as_str() {
            "post" | "article" | "blog" => Some("content".to_string()),
            "video" | "audio" | "podcast" => Some("content".to_string()),
            "discussion" | "comment" | "reply" => Some("social".to_string()),
            "code" | "programming" | "tech" => Some("technology".to_string()),
            _ => None,
        }
    }

    fn add_privacy_noise(&self, value: f64, noise_scale: f64) -> f64 {
        let noise = (rand::random::<f64>() - 0.5) * noise_scale * 2.0;
        (value + noise).clamp(0.0, 1.0)
    }

    fn generalize_interest_category(&self, mut interest: Interest) -> Result<Interest> {
        if interest.privacy_masked {
            if let Some(category) = self.interest_categories.get(&interest.category) {
                if !category.subcategories.is_empty() {
                    interest.subcategory = None;
                }
            }
            
            let generalization_mapping = HashMap::from([
                ("programming", "technology"),
                ("ai", "technology"),
                ("blockchain", "technology"),
                ("networking", "social"),
                ("community", "social"),
                ("media", "content"),
                ("articles", "content"),
            ]);
            
            if let Some(generalized) = generalization_mapping.get(interest.category.as_str()) {
                interest.category = generalized.to_string();
            }
        }

        Ok(interest)
    }
}