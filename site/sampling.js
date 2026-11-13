(function attachSamplingLab(root, factory) {
  const api = factory();

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }

  if (root) {
    root.SamplingLab = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function createSamplingLab() {
  'use strict';

  // Fixed logits make the learning demo reproducible and easy to inspect.
  const CANDIDATES = Object.freeze([
    Object.freeze({ token: 'cache', logit: 2.9 }),
    Object.freeze({ token: 'token', logit: 2.4 }),
    Object.freeze({ token: 'GPU', logit: 1.7 }),
    Object.freeze({ token: 'batch', logit: 1.1 }),
    Object.freeze({ token: 'attention', logit: 0.4 }),
    Object.freeze({ token: 'latency', logit: -0.2 }),
  ]);

  function softmax(logits, temperature) {
    if (!Array.isArray(logits) || logits.length === 0) {
      throw new TypeError('logits must be a non-empty array');
    }
    if (logits.some((logit) => !Number.isFinite(logit))) {
      throw new TypeError('every logit must be a finite number');
    }
    if (!Number.isFinite(temperature) || temperature <= 0) {
      throw new RangeError('temperature must be a finite number greater than 0');
    }

    const maxLogit = Math.max(...logits);
    const weights = logits.map((logit) => Math.exp((logit - maxLogit) / temperature));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    return weights.map((weight) => weight / total);
  }

  function candidateDistribution(temperature) {
    return softmax(CANDIDATES.map((candidate) => candidate.logit), temperature);
  }

  function createRng(seed) {
    let state = Number(seed) >>> 0;

    return function nextRandom() {
      state = (state + 0x6d2b79f5) >>> 0;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sampleIndex(probabilities, randomValue) {
    if (!Array.isArray(probabilities) || probabilities.length === 0) {
      throw new TypeError('probabilities must be a non-empty array');
    }
    if (probabilities.some((probability) => !Number.isFinite(probability) || probability < 0)) {
      throw new TypeError('probabilities must be finite non-negative numbers');
    }
    if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
      throw new RangeError('randomValue must be in the interval [0, 1)');
    }

    let cumulative = 0;
    for (let index = 0; index < probabilities.length; index += 1) {
      cumulative += probabilities[index];
      if (randomValue < cumulative) {
        return index;
      }
    }

    return probabilities.length - 1;
  }

  return Object.freeze({
    CANDIDATES,
    softmax,
    candidateDistribution,
    createRng,
    sampleIndex,
  });
});
