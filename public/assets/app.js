const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.tool-panel');
const results = document.querySelector('#results');
const loading = document.querySelector('#loading');
const errorBox = document.querySelector('#error');

tabs.forEach(tab => tab.addEventListener('click', () => {
  tabs.forEach(item => { item.classList.remove('active'); item.setAttribute('aria-selected', 'false'); });
  panels.forEach(panel => panel.classList.remove('active'));
  tab.classList.add('active');
  tab.setAttribute('aria-selected', 'true');
  document.querySelector(`#${tab.dataset.panel}`).classList.add('active');
  clearResult();
}));

document.querySelectorAll('.examples button').forEach(button => button.addEventListener('click', () => {
  document.querySelector('#nearest-word').value = button.textContent;
  document.querySelector('#nearest').requestSubmit();
}));

document.querySelector('#nearest').addEventListener('submit', async event => {
  event.preventDefault();
  const word = document.querySelector('#nearest-word').value;
  await run(`/api/nearest?word=${encodeURIComponent(word)}`, {}, data => renderList(data.results));
});

document.querySelector('#similarity').addEventListener('submit', async event => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  await run('/api/similarity', jsonPost({first: form.get('first'), second: form.get('second')}), data => renderScore(data.score, `${data.first} ↔ ${data.second}`));
});

document.querySelector('#analogy').addEventListener('submit', async event => {
  event.preventDefault();
  const form = Object.fromEntries(new FormData(event.currentTarget));
  await run('/api/analogy', jsonPost(form), data => renderList(data.results, data.formula));
});

document.querySelector('#sentences').addEventListener('submit', async event => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  await run('/api/sentence-similarity', jsonPost({first: form.get('first'), second: form.get('second')}), data => renderScore(data.score, 'Cümlelerin kosinüs benzerliği', [...data.tokens.first, ...data.tokens.second]));
});

function jsonPost(body) { return {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}; }
async function run(url, options, render) {
  loading.classList.remove('hidden'); errorBox.classList.add('hidden'); results.classList.add('hidden');
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'İstek tamamlanamadı.');
    render(data);
  } catch (error) {
    errorBox.textContent = error.message; errorBox.classList.remove('hidden');
  } finally {
    loading.classList.add('hidden'); results.classList.remove('hidden');
  }
}
function clearResult() { errorBox.classList.add('hidden'); results.innerHTML = '<div class="empty-orbit"><span>Bir deney<br>başlat</span></div>'; }
function renderList(items, heading = '') {
  results.innerHTML = `${heading ? `<p class="score-caption">${escapeHtml(heading)}</p>` : ''}${items.map((item, index) => `<div class="result-item"><span class="rank">0${index + 1}</span><div class="bar-wrap"><span class="bar-label">${escapeHtml(item.word)}</span><span class="bar"><i style="width:${Math.max(2, item.score * 100)}%"></i></span></div><span class="score">${item.score.toFixed(3)}</span></div>`).join('')}`;
}
function renderScore(score, caption, tokens = []) {
  const percent = Math.round(Math.max(0, score) * 100);
  results.innerHTML = `<div class="big-score">${percent}<small>%</small></div><p class="score-caption">${escapeHtml(caption)}</p>${tokens.length ? `<div class="token-list">${tokens.map(token => `<span>${escapeHtml(token)}</span>`).join('')}</div>` : ''}`;
}
function escapeHtml(value) { const node = document.createElement('span'); node.textContent = value; return node.innerHTML; }

fetch('/api/health').then(response => response.json()).then(data => { document.querySelector('#vocab-count').textContent = data.vocabulary_size; }).catch(() => { document.querySelector('.status').innerHTML = '<span style="background:#ff6b36"></span> Model çevrimdışı'; });

