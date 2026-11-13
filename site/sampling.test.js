const test = require('node:test');
const assert = require('node:assert/strict');

const {
  CANDIDATES,
  softmax,
  candidateDistribution,
  createRng,
  sampleIndex,
} = require('./sampling.js');

test('softmax returns a normalized, stable distribution', () => {
  const probabilities = softmax([1000, 999, 998], 1);

  assert.equal(probabilities.length, 3);
  assert.ok(probabilities.every((probability) => probability > 0 && probability < 1));
  assert.ok(Math.abs(probabilities.reduce((sum, probability) => sum + probability, 0) - 1) < 1e-12);
  assert.ok(probabilities[0] > probabilities[1]);
  assert.ok(probabilities[1] > probabilities[2]);
});

test('lower temperature sharpens the candidate distribution', () => {
  const cold = candidateDistribution(0.1);
  const warm = candidateDistribution(1.5);

  assert.ok(cold[0] > 0.99);
  assert.ok(cold[0] > warm[0]);
  assert.ok(cold[0] > warm[warm.length - 1]);
  assert.deepEqual(
    candidateDistribution(0.8).map((probability) => Number(probability.toFixed(8))),
    softmax(CANDIDATES.map((candidate) => candidate.logit), 0.8).map((probability) => Number(probability.toFixed(8))),
  );
});

test('softmax rejects invalid inputs', () => {
  assert.throws(() => softmax([], 1), /non-empty array/);
  assert.throws(() => softmax([1, Number.NaN], 1), /finite number/);
  assert.throws(() => softmax([1, 2], 0), /greater than 0/);
  assert.throws(() => softmax([1, 2], Number.POSITIVE_INFINITY), /greater than 0/);
});

test('sampleIndex maps a uniform draw to its probability interval', () => {
  const probabilities = [0.1, 0.2, 0.7];

  assert.equal(sampleIndex(probabilities, 0), 0);
  assert.equal(sampleIndex(probabilities, 0.099999), 0);
  assert.equal(sampleIndex(probabilities, 0.1), 1);
  assert.equal(sampleIndex(probabilities, 0.299999), 1);
  assert.equal(sampleIndex(probabilities, 0.300001), 2);
  assert.equal(sampleIndex(probabilities, 0.999999), 2);
  assert.throws(() => sampleIndex(probabilities, 1), /interval/);
});

test('seeded RNG is repeatable and stays in the unit interval', () => {
  const first = createRng(42);
  const second = createRng(42);
  const different = createRng(43);
  const firstSequence = Array.from({ length: 8 }, first);
  const secondSequence = Array.from({ length: 8 }, second);

  assert.deepEqual(firstSequence, secondSequence);
  assert.ok(firstSequence.every((value) => value >= 0 && value < 1));
  assert.notDeepEqual(firstSequence, Array.from({ length: 8 }, different));
});
