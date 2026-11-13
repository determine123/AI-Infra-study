(function attachKvCacheLab(root, factory) {
  const api = factory();

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }

  if (root) {
    root.KvCacheLab = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function createKvCacheLab() {
  'use strict';

  // Decimal byte sizes are used for teaching, not filesystem allocation.
  const DTYPE_BYTES = Object.freeze({
    fp32: 4,
    fp16: 2,
    bf16: 2,
    fp8: 1,
  });

  function requirePositiveInteger(name, value) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`${name} must be a positive safe integer`);
    }
  }

  function estimateKvCache({
    layers,
    kvHeads,
    contextTokens,
    batchSize,
    headDim,
    bytesPerElement,
  }) {
    requirePositiveInteger('layers', layers);
    requirePositiveInteger('kvHeads', kvHeads);
    requirePositiveInteger('contextTokens', contextTokens);
    requirePositiveInteger('batchSize', batchSize);
    requirePositiveInteger('headDim', headDim);
    if (!Number.isFinite(bytesPerElement) || bytesPerElement <= 0) {
      throw new RangeError('bytesPerElement must be a finite positive number');
    }

    // Two tensors (K and V) for every token, request, layer and KV head.
    const bytes =
      2 * layers * kvHeads * contextTokens * batchSize * headDim * bytesPerElement;
    if (!Number.isSafeInteger(bytes)) {
      throw new RangeError('KV cache estimate is too large for exact display');
    }

    return Object.freeze({
      bytes,
      gib: bytes / 1024 ** 3,
      inputs: Object.freeze({
        layers,
        kvHeads,
        contextTokens,
        batchSize,
        headDim,
        bytesPerElement,
      }),
    });
  }

  function formatGib(gib) {
    if (!Number.isFinite(gib) || gib < 0) {
      throw new RangeError('gib must be a finite non-negative number');
    }
    if (gib > 0 && gib < 0.01) {
      return '<0.01 GiB';
    }
    return `${gib.toFixed(2)} GiB`;
  }

  return Object.freeze({
    DTYPE_BYTES,
    estimateKvCache,
    formatGib,
  });
});
