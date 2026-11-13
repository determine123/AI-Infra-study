const $ = (id) => document.getElementById(id);
const lab = window.SamplingLab;
const candidates = lab.CANDIDATES;

let currentSeed = 42;
let rng = lab.createRng(currentSeed);
let sampleCount = 0;
let probabilities = [];

function formatPercent(probability) {
  const percent = probability * 100;
  if (probability > 0 && percent < 0.1) return '<0.1%';
  return `${percent.toFixed(1)}%`;
}

function renderDistribution(temperature) {
  probabilities = lab.candidateDistribution(temperature);
  $('probabilityList').innerHTML = candidates.map((candidate, index) => {
    const probability = probabilities[index];
    const percent = formatPercent(probability);
    return `<div class="prob-row">
      <div class="prob-label"><span>${candidate.token}</span><span>logit ${candidate.logit.toFixed(1)} · ${percent}</span></div>
      <div class="prob-track" role="meter" aria-label="${candidate.token} 概率" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${(probability * 100).toFixed(1)}"><span class="prob-fill" style="width:${(probability * 100).toFixed(1)}%"></span></div>
    </div>`;
  }).join('');

  let topIndex = 0;
  probabilities.forEach((probability, index) => {
    if (probability > probabilities[topIndex]) topIndex = index;
  });
  $('distributionNote').textContent = `当前最高：${candidates[topIndex].token} ${(probabilities[topIndex] * 100).toFixed(1)}%。temperature 越低，分布越尖锐。`;
}

function resetSampler(message) {
  const parsedSeed = Math.trunc(Number($('sampleSeed').value));
  currentSeed = Number.isSafeInteger(parsedSeed) && parsedSeed >= 0 && parsedSeed <= 0xffffffff ? parsedSeed : 42;
  $('sampleSeed').value = String(currentSeed);
  rng = lab.createRng(currentSeed);
  sampleCount = 0;
  renderDistribution(Number($('temperature').value));
  $('tokenOutput').innerHTML = '<span style="color:#888">next_token = </span><span style="color:#777">等待采样</span>';
  $('sampleMeta').textContent = `seed ${currentSeed} · ${message}`;
}

$('temperature').addEventListener('input', (e) => {
  $('tempValue').textContent = Number(e.target.value).toFixed(2);
  renderDistribution(Number(e.target.value));
});
$('sampleBtn').addEventListener('click', () => {
  renderDistribution(Number($('temperature').value));
  const randomValue = rng();
  const selectedIndex = lab.sampleIndex(probabilities, randomValue);
  const selected = candidates[selectedIndex];
  sampleCount += 1;
  $('tokenOutput').innerHTML = `<span style="color:#888">next_token = </span>“${selected.token}”`;
  $('sampleMeta').textContent = `seed ${currentSeed} · 采样 #${sampleCount} · u=${randomValue.toFixed(6)} · p=${(probabilities[selectedIndex] * 100).toFixed(1)}%`;
});
$('sampleSeed').addEventListener('change', () => resetSampler('随机序列已重置'));
$('resetSampling').addEventListener('click', () => resetSampler('随机序列已重置'));

function calculateCache() {
  const layers = Number($('layers').value) || 1;
  const heads = Number($('heads').value) || 1;
  const context = Number($('context').value) || 1;
  const batch = Number($('batch').value) || 1;
  const headDim = 128;
  const bytes = 2;
  const gb = 2 * layers * heads * context * batch * headDim * bytes / 1024 ** 3;
  $('cacheResult').innerHTML = `<strong>${gb.toFixed(2)} GB</strong> <span style="color:#999">KV Cache (FP16)</span>`;
}
['layers','heads','context','batch'].forEach(id => $(id).addEventListener('input', calculateCache));
$('calcCache').addEventListener('click', calculateCache);
calculateCache();

let requestId = 1;
const requests = [];
function renderQueue() {
  $('queue').innerHTML = requests.length ? requests.map(r => `<span class="queue-item ${r.done ? 'done' : ''}">R${r.id}</span>`).join('') : '<span style="color:#777;font-size:12px">waiting queue is empty</span>';
}
$('addRequest').addEventListener('click', () => {
  requests.push({ id: requestId++, tokens: 3 + Math.floor(Math.random() * 6), done: false });
  $('queueNote').textContent = `当前 running batch：${requests.filter(r => !r.done).length} 个请求。新请求可以随时加入。`;
  renderQueue();
});
$('tick').addEventListener('click', () => {
  const active = requests.filter(r => !r.done);
  if (!active.length) { $('queueNote').textContent = '没有可执行的请求，请先加入请求。'; return; }
  active.forEach(r => { r.tokens -= 1; if (r.tokens <= 0) r.done = true; });
  const finished = requests.filter(r => r.done).length;
  $('queueNote').textContent = `完成 ${finished} 个请求；本轮 Decode 后仍有 ${requests.filter(r => !r.done).length} 个请求运行中。`;
  renderQueue();
});
renderQueue();

resetSampler('尚未采样');

$('themeToggle').addEventListener('click', () => document.body.classList.toggle('dark'));
