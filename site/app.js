const $ = (id) => document.getElementById(id);

const words = ['token', 'KV', 'cache', 'attention', 'runtime', 'batch', 'decode', 'GPU', 'throughput', 'latency'];
let tokenIndex = 0;
$('temperature').addEventListener('input', (e) => $('tempValue').textContent = Number(e.target.value).toFixed(2));
$('sampleBtn').addEventListener('click', () => {
  const temp = Number($('temperature').value);
  const offset = Math.max(1, Math.round(temp * 3));
  const word = words[(tokenIndex++ * offset + Math.floor(Math.random() * words.length)) % words.length];
  $('tokenOutput').innerHTML = `<span style="color:#888">next_token = </span>“${word}”`;
});

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

$('themeToggle').addEventListener('click', () => document.body.classList.toggle('dark'));
