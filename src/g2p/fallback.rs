//! Neural grapheme-to-phoneme for words the lexicon doesn't know.
//!
//! Port of `EnglishFallbackNetwork` + `BARTModel` in `packages/kokoro-swift`,
//! reading the *same* Apache-2.0 `us_bart.safetensors` weights, so an unknown
//! word is pronounced identically on macOS and everywhere else.
//!
//! The model is tiny — `d_model` 128, one encoder layer, one decoder layer, one
//! attention head, a 63-symbol vocabulary — so this is a few hundred lines of
//! plain arithmetic rather than a tensor-framework dependency. On real Bisque
//! narration it fires on ~1.8% of tokens (technical identifiers, mostly), which
//! is why the whole thing is loaded lazily.
//!
//! Faithfulness note: upstream's `generate` runs greedy decoding with **no**
//! causal mask and re-decodes the whole prefix each step. That is reproduced
//! exactly. A "fix" here would be a cross-platform pronunciation divergence.

use anyhow::{anyhow, Context, Result};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;

const MAX_LENGTH: usize = 50;
const UNKNOWN_TOKEN_ID: usize = 3;
/// BART offsets positions by 2 (slots 0 and 1 are reserved).
const POSITION_OFFSET: usize = 2;

#[derive(Debug, Deserialize)]
pub struct BartConfig {
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub grapheme_chars: String,
    pub phoneme_chars: String,
}

/// Row-major dense matrix.
#[derive(Clone)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
    fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }
    fn add_assign(&mut self, other: &Mat) {
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += *b;
        }
    }
}

struct Linear {
    /// `[out, in]`, PyTorch convention: `y = x · Wᵀ + b`.
    weight: Mat,
    bias: Option<Vec<f32>>,
}

impl Linear {
    fn apply(&self, x: &Mat) -> Mat {
        let out_dim = self.weight.rows;
        let mut out = Mat::zeros(x.rows, out_dim);
        for r in 0..x.rows {
            let xr = x.row(r);
            let dst = out.row_mut(r);
            for o in 0..out_dim {
                let w = self.weight.row(o);
                let mut sum = self.bias.as_ref().map_or(0.0, |b| b[o]);
                for (xv, wv) in xr.iter().zip(w) {
                    sum += xv * wv;
                }
                dst[o] = sum;
            }
        }
        out
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    fn apply(&self, x: &Mat) -> Mat {
        const EPS: f32 = 1e-5;
        let mut out = Mat::zeros(x.rows, x.cols);
        for r in 0..x.rows {
            let xr = x.row(r);
            let mean = xr.iter().sum::<f32>() / x.cols as f32;
            let var = xr.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.cols as f32;
            let denom = (var + EPS).sqrt();
            let dst = out.row_mut(r);
            for c in 0..x.cols {
                dst[c] = (xr[c] - mean) / denom * self.weight[c] + self.bias[c];
            }
        }
        out
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
}

impl Attention {
    fn apply(&self, query: &Mat, key_value: &Mat) -> Mat {
        let d_model = query.cols;
        let head_dim = d_model / self.heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = self.q.apply(query);
        let k = self.k.apply(key_value);
        let v = self.v.apply(key_value);

        let mut context = Mat::zeros(query.rows, d_model);
        for h in 0..self.heads {
            let off = h * head_dim;
            for i in 0..q.rows {
                // Scores over every key (no mask — see the module docs).
                let mut scores = vec![0.0f32; k.rows];
                for j in 0..k.rows {
                    let mut s = 0.0;
                    for d in 0..head_dim {
                        s += q.row(i)[off + d] * k.row(j)[off + d];
                    }
                    scores[j] = s * scale;
                }
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    total += *s;
                }
                let dst = context.row_mut(i);
                for (j, w) in scores.iter().enumerate() {
                    let w = w / total;
                    for d in 0..head_dim {
                        dst[off + d] += w * v.row(j)[off + d];
                    }
                }
            }
        }
        self.o.apply(&context)
    }
}

struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn apply(&self, x: &Mat) -> Mat {
        let mut h = self.fc1.apply(x);
        for v in h.data.iter_mut() {
            // Exact erf-based GELU, matching MLX's default.
            *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
        }
        self.fc2.apply(&h)
    }
}

/// Abramowitz & Stegun 7.1.26 — well inside f32 precision for GELU.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

struct EncoderLayer {
    attn: Attention,
    attn_norm: LayerNorm,
    ffn: FeedForward,
    ffn_norm: LayerNorm,
}

struct DecoderLayer {
    self_attn: Attention,
    self_attn_norm: LayerNorm,
    cross_attn: Attention,
    cross_attn_norm: LayerNorm,
    ffn: FeedForward,
    ffn_norm: LayerNorm,
}

pub struct Fallback {
    config: BartConfig,
    shared: Mat,
    encoder_positions: Mat,
    decoder_positions: Mat,
    encoder_norm: LayerNorm,
    decoder_norm: LayerNorm,
    encoder: EncoderLayer,
    decoder: DecoderLayer,
    logit_bias: Vec<f32>,
    grapheme_to_token: HashMap<char, usize>,
    token_to_phoneme: HashMap<usize, char>,
}

impl Fallback {
    pub fn load(config: BartConfig, weights_bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(weights_bytes).context("parsing BART safetensors")?;
        let mat = |name: &str| -> Result<Mat> {
            let t = st.tensor(name).map_err(|e| anyhow!("{name}: {e}"))?;
            let shape = t.shape();
            let (rows, cols) = match shape.len() {
                2 => (shape[0], shape[1]),
                1 => (1, shape[0]),
                n => return Err(anyhow!("{name}: unexpected rank {n}")),
            };
            let data: Vec<f32> = t
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(Mat { rows, cols, data })
        };
        let vec1 = |name: &str| -> Result<Vec<f32>> { Ok(mat(name)?.data) };
        let linear = |prefix: &str, with_bias: bool| -> Result<Linear> {
            Ok(Linear {
                weight: mat(&format!("{prefix}.weight"))?,
                bias: if with_bias {
                    Some(vec1(&format!("{prefix}.bias"))?)
                } else {
                    None
                },
            })
        };
        let norm = |prefix: &str| -> Result<LayerNorm> {
            Ok(LayerNorm {
                weight: vec1(&format!("{prefix}.weight"))?,
                bias: vec1(&format!("{prefix}.bias"))?,
            })
        };
        let attention = |prefix: &str, heads: usize| -> Result<Attention> {
            Ok(Attention {
                q: linear(&format!("{prefix}.q_proj"), true)?,
                k: linear(&format!("{prefix}.k_proj"), true)?,
                v: linear(&format!("{prefix}.v_proj"), true)?,
                o: linear(&format!("{prefix}.out_proj"), true)?,
                heads,
            })
        };

        let grapheme_to_token = config
            .grapheme_chars
            .chars()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();
        let token_to_phoneme = config
            .phoneme_chars
            .chars()
            .enumerate()
            .map(|(i, c)| (i, c))
            .collect();

        Ok(Fallback {
            shared: mat("model.shared.weight")?,
            encoder_positions: mat("model.encoder.embed_positions.weight")?,
            decoder_positions: mat("model.decoder.embed_positions.weight")?,
            encoder_norm: norm("model.encoder.layernorm_embedding")?,
            decoder_norm: norm("model.decoder.layernorm_embedding")?,
            encoder: EncoderLayer {
                attn: attention(
                    "model.encoder.layers.0.self_attn",
                    config.encoder_attention_heads,
                )?,
                attn_norm: norm("model.encoder.layers.0.self_attn_layer_norm")?,
                ffn: FeedForward {
                    fc1: linear("model.encoder.layers.0.fc1", true)?,
                    fc2: linear("model.encoder.layers.0.fc2", true)?,
                },
                ffn_norm: norm("model.encoder.layers.0.final_layer_norm")?,
            },
            decoder: DecoderLayer {
                self_attn: attention(
                    "model.decoder.layers.0.self_attn",
                    config.decoder_attention_heads,
                )?,
                self_attn_norm: norm("model.decoder.layers.0.self_attn_layer_norm")?,
                cross_attn: attention(
                    "model.decoder.layers.0.encoder_attn",
                    config.decoder_attention_heads,
                )?,
                cross_attn_norm: norm("model.decoder.layers.0.encoder_attn_layer_norm")?,
                ffn: FeedForward {
                    fc1: linear("model.decoder.layers.0.fc1", true)?,
                    fc2: linear("model.decoder.layers.0.fc2", true)?,
                },
                ffn_norm: norm("model.decoder.layers.0.final_layer_norm")?,
            },
            logit_bias: vec1("final_logits_bias")?,
            grapheme_to_token,
            token_to_phoneme,
            config,
        })
    }

    fn embed(&self, ids: &[usize], positions: &Mat) -> Mat {
        let d = self.config.d_model;
        let mut h = Mat::zeros(ids.len(), d);
        for (i, id) in ids.iter().enumerate() {
            let token = self.shared.row((*id).min(self.shared.rows - 1));
            let pos = positions.row((i + POSITION_OFFSET).min(positions.rows - 1));
            let dst = h.row_mut(i);
            for c in 0..d {
                dst[c] = token[c] + pos[c];
            }
        }
        h
    }

    fn encode(&self, ids: &[usize]) -> Mat {
        let mut h = self
            .encoder_norm
            .apply(&self.embed(ids, &self.encoder_positions));
        let mut attn = self.encoder.attn.apply(&h, &h);
        attn.add_assign(&h);
        h = self.encoder.attn_norm.apply(&attn);
        let mut ffn = self.encoder.ffn.apply(&h);
        ffn.add_assign(&h);
        self.encoder.ffn_norm.apply(&ffn)
    }

    /// Logits for the last decoder position.
    fn decode_last(&self, ids: &[usize], encoded: &Mat) -> Vec<f32> {
        let mut h = self
            .decoder_norm
            .apply(&self.embed(ids, &self.decoder_positions));
        let mut attn = self.decoder.self_attn.apply(&h, &h);
        attn.add_assign(&h);
        h = self.decoder.self_attn_norm.apply(&attn);

        let mut cross = self.decoder.cross_attn.apply(&h, encoded);
        cross.add_assign(&h);
        h = self.decoder.cross_attn_norm.apply(&cross);

        let mut ffn = self.decoder.ffn.apply(&h);
        ffn.add_assign(&h);
        h = self.decoder.ffn_norm.apply(&ffn);

        // The LM head is the tied embedding matrix.
        let last = h.row(h.rows - 1);
        (0..self.shared.rows)
            .map(|v| {
                let row = self.shared.row(v);
                let dot: f32 = last.iter().zip(row).map(|(a, b)| a * b).sum();
                dot + self.logit_bias.get(v).copied().unwrap_or(0.0)
            })
            .collect()
    }

    /// Pronounce one out-of-vocabulary word. `None` when the model produces
    /// nothing usable — the caller must not turn that into silence.
    pub fn phonemize(&self, word: &str) -> Option<(String, i32)> {
        let mut ids = vec![self.config.bos_token_id];
        ids.extend(word.chars().map(|c| {
            self.grapheme_to_token
                .get(&c)
                .copied()
                .unwrap_or(UNKNOWN_TOKEN_ID)
        }));
        ids.push(self.config.eos_token_id);

        let encoded = self.encode(&ids);
        let mut decoder_input = vec![self.config.bos_token_id];
        let mut generated: Vec<usize> = Vec::new();

        for step in 0..MAX_LENGTH {
            if step == MAX_LENGTH - 1 {
                break;
            }
            let logits = self.decode_last(&decoder_input, &encoded);
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)?;
            if next == self.config.eos_token_id {
                break;
            }
            generated.push(next);
            decoder_input.push(next);
        }

        let phonemes: String = generated
            .iter()
            .filter(|t| **t > UNKNOWN_TOKEN_ID)
            .filter_map(|t| self.token_to_phoneme.get(t))
            .collect();
        if phonemes.is_empty() {
            return None;
        }
        Some((phonemes, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_erf_matches_known_values() {
        // erf(0)=0, erf(1)≈0.8427, erf(-1)≈-0.8427
        assert!((erf(0.0)).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-4);
        assert!((erf(-1.0) + 0.842_700_8).abs() < 1e-4);
        assert!((erf(3.0) - 0.999_978).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_zero_means_and_unit_scales() {
        let ln = LayerNorm {
            weight: vec![1.0; 4],
            bias: vec![0.0; 4],
        };
        let x = Mat {
            rows: 1,
            cols: 4,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let out = ln.apply(&x);
        let mean: f32 = out.data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
        let var: f32 = out.data.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3, "variance should be ~1, got {var}");
    }

    #[test]
    fn linear_applies_weight_transpose_plus_bias() {
        // y = x · Wᵀ + b, with W = [[1,2],[3,4]], x = [1,1], b = [10,20]
        let lin = Linear {
            weight: Mat {
                rows: 2,
                cols: 2,
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
            bias: Some(vec![10.0, 20.0]),
        };
        let x = Mat {
            rows: 1,
            cols: 2,
            data: vec![1.0, 1.0],
        };
        let y = lin.apply(&x);
        assert_eq!(y.data, vec![13.0, 27.0]);
    }

    #[test]
    fn attention_over_one_key_returns_that_value() {
        // Identity projections, single key: softmax over one score is 1.0, so
        // the context is exactly the value.
        let identity = || Linear {
            weight: Mat {
                rows: 2,
                cols: 2,
                data: vec![1.0, 0.0, 0.0, 1.0],
            },
            bias: None,
        };
        let attn = Attention {
            q: identity(),
            k: identity(),
            v: identity(),
            o: identity(),
            heads: 1,
        };
        let q = Mat {
            rows: 1,
            cols: 2,
            data: vec![0.3, 0.7],
        };
        let kv = Mat {
            rows: 1,
            cols: 2,
            data: vec![5.0, -2.0],
        };
        let out = attn.apply(&q, &kv);
        assert!((out.data[0] - 5.0).abs() < 1e-5);
        assert!((out.data[1] + 2.0).abs() < 1e-5);
    }
}
