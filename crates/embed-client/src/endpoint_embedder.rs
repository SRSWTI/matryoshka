use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use reqwest::header::CONNECTION;
use serde::{Deserialize, Serialize};

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
            .context("failed to call embeddings endpoint")
            .and_then(response_with_body_on_error)?
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

fn response_with_body_on_error(
    response: reqwest::blocking::Response,
) -> Result<reqwest::blocking::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response
        .text()
        .unwrap_or_else(|err| format!("<failed to read error body: {err}>"));
    Err(anyhow!("embeddings endpoint returned {status}: {body}"))
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
