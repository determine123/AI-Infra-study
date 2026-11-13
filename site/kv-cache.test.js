const test = require('node:test');
const assert = require('node:assert/strict');

const { DTYPE_BYTES, estimateKvCache, formatGib } = require('./kv-cache.js');

test('KV cache estimate accounts for K and V tensors', () => {
  const result = estimateKvCache({
    layers: 32,
    kvHeads: 8,
    contextTokens: 4096,
    batchSize: 8,
    headDim: 128,
    bytesPerElement: DTYPE_BYTES.bf16,
  });

  assert.equal(result.bytes, 4 * 1024 ** 3);
  assert.equal(result.gib, 4);
  assert.equal(formatGib(result.gib), '4.00 GiB');
});

test('fewer KV heads and narrower elements reduce the cache proportionally', () => {
  const baseline = estimateKvCache({
    layers: 32,
    kvHeads: 8,
    contextTokens: 2048,
    batchSize: 4,
    headDim: 128,
    bytesPerElement: DTYPE_BYTES.bf16,
  });
  const grouped = estimateKvCache({
    layers: 32,
    kvHeads: 2,
    contextTokens: 2048,
    batchSize: 4,
    headDim: 128,
    bytesPerElement: DTYPE_BYTES.bf16,
  });
  const fp8 = estimateKvCache({
    layers: 32,
    kvHeads: 8,
    contextTokens: 2048,
    batchSize: 4,
    headDim: 128,
    bytesPerElement: DTYPE_BYTES.fp8,
  });

  assert.equal(grouped.bytes * 4, baseline.bytes);
  assert.equal(fp8.bytes * 2, baseline.bytes);
});

test('KV cache estimate rejects incomplete or unsafe dimensions', () => {
  const valid = {
    layers: 32,
    kvHeads: 8,
    contextTokens: 4096,
    batchSize: 8,
    headDim: 128,
    bytesPerElement: 2,
  };

  for (const field of ['layers', 'kvHeads', 'contextTokens', 'batchSize', 'headDim']) {
    for (const value of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      assert.throws(
        () => estimateKvCache({ ...valid, [field]: value }),
        /positive safe integer/,
      );
    }
  }
  assert.throws(
    () => estimateKvCache({ ...valid, bytesPerElement: 0 }),
    /finite positive/,
  );
  assert.throws(
    () => estimateKvCache({ ...valid, contextTokens: Number.MAX_SAFE_INTEGER }),
    /too large/,
  );
});

test('KV cache formatting avoids implying false precision for tiny values', () => {
  assert.equal(formatGib(0), '0.00 GiB');
  assert.equal(formatGib(0.001), '<0.01 GiB');
  assert.equal(formatGib(1.234), '1.23 GiB');
  assert.throws(() => formatGib(Number.NaN), /finite non-negative/);
  assert.throws(() => formatGib(-1), /finite non-negative/);
});
