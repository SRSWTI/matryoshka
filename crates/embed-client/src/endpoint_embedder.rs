use anyhow::{Context, Result};
use reqwest::blocking::Client;
use reqwest::header::CONNECTION;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub trait Embedder {
    fn model(&self) -> &str;
    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[derive(Debug, Clone)]
pub struct EndpointEmbedder {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl EndpointEmbedder {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

impl Embedder for EndpointEmbedder {
    fn model(&self) -> &str {
        &self.model
    }

    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let response: EmbeddingResponse = self
            .client
            .post(format!("{}/v1/embeddings", self.base_url))
            .header(CONNECTION, "close")
            .bearer_auth(&self.api_key)
            .json(&EmbeddingRequest {
                model: self.model.clone(),
                input: inputs.to_vec(),
                encoding_format: "float",
            })
            .send()
            .context("failed to call embeddings endpoint")?
            .error_for_status()
            .context("embeddings endpoint returned an error")?
            .text()
            .context("failed to read embeddings response body")
            .and_then(|body| {
                serde_json::from_str::<EmbeddingResponse>(&body)
                    .context("failed to parse embeddings response")
            })?;

        let mut rows = response.data;
        rows.sort_by_key(|row| row.index);
        Ok(rows
            .into_iter()
            .map(|row| normalize(row.embedding))
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct DeterministicEmbedder {
    model: String,
    dimensions: usize,
}

impl DeterministicEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self {
            model: format!("deterministic-{dimensions}"),
            dimensions,
        }
    }
}

impl Default for DeterministicEmbedder {
    fn default() -> Self {
        Self::new(96)
    }
}

impl Embedder for DeterministicEmbedder {
    fn model(&self) -> &str {
        &self.model
    }

    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(inputs
            .iter()
            .map(|input| deterministic_vector(input, self.dimensions))
            .collect())
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    encoding_format: &'static str,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

fn deterministic_vector(text: &str, dimensions: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimensions];
    for token in text.split(|ch: char| !ch.is_alphanumeric() && ch != '_') {
        if token.is_empty() {
            continue;
        }
        let mut hasher = Sha256::new();
        hasher.update(token.to_lowercase().as_bytes());
        let digest = hasher.finalize();
        let index =
            u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize % dimensions;
        vector[index] += 1.0;
    }
    normalize(vector)
}

pub fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

pub fn cosine(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}
