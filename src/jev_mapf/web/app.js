const $ = id => document.getElementById(id);
const NEON = ['#00f0ff', '#ff2bd6', '#a6ff00', '#ffb400', '#9d6bff', '#ff4d6d', '#2bffc6', '#4d8bff',
  '#ff8a3d', '#e6ff3d', '#ff5ea8', '#3dd6ff', '#c77dff', '#7dff9b', '#ffd23d', '#ff3d3d'];
const MOVE = ['·', '↑', '↓', '←', '→'];
const MOVE_NAME = ['wait', 'up', 'down', 'left', 'right'];
const STATUS = {ready: 'READY', deciding: 'THINKING', paused: 'PAUSED', success: 'SOLVED', truncated: 'TIME UP',
  error: 'ERROR', stopped: 'STOPPED', running: 'RUNNING'};
const PROVIDER = {astar: 'A*', random: '随机', deepseek: 'DeepSeek', chat: 'OpenAI 兼容', qwen_rlcd: 'Qwen·RLCD', jev: 'Jev'};
let configuration, episode, archived = null, savedTerminal = null, following = true;
let commandPending = false, viewRevision = 0, initialized = false, activePreset = null;
const liveRecords = new Map();
const isTerminal = ep => !!ep && ['success', 'error', 'truncated', 'stopped'].includes(ep.status);

async function api(path, method = 'GET', body, timeoutMs = 15000) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetch(path, {method, signal: controller.signal, headers: {'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body)});
    const data = await r.json();
    if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    return data;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('服务响应超时');
    throw e;
  } finally { clearTimeout(timer); }
}
function notice(text) { $('notice').textContent = text || ''; }
async function guard(fn) { try { await fn(); } catch (e) { notice(e.message); } }
function el(tag, attrs = {}, text) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === 'style' ? n.style.cssText = v : n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}

// ---- models & connection ----
function selectProvider(p) {
  $('provider').value = p;
  for (const b of document.querySelectorAll('#models button')) b.classList.toggle('active', b.dataset.provider === p);
  connectionForm();
}
function connectionForm() {
  const p = $('provider').value, c = configuration.connections[p];
  $('open-connection').disabled = !c;
  if (!c) return;
  $('url').value = c.url; $('model').value = c.model; $('key').value = '';
  $('thinking').value = String(c.thinking); $('thinking').disabled = p !== 'deepseek'; $('max-tokens').value = c.max_tokens;
  $('url').readOnly = ['deepseek', 'jev'].includes(p);
  $('verified').textContent = c.verified ? 'VERIFIED' : c.has_key ? 'KEY SET' : '';
  $('key').placeholder = c.has_key ? '已配置，留空保留' : '';
  $('import-deepseek').hidden = p !== 'deepseek';
}
async function loadConfig() { configuration = await api('/api/config'); connectionForm(); }
async function saveConnection() {
  const p = $('provider').value;
  await api(`/api/connections/${p}`, 'PUT', {url: $('url').value.trim(), model: $('model').value.trim(),
    key: $('key').value.trim(), thinking: $('thinking').value === 'true', max_tokens: Number($('max-tokens').value)});
  await loadConfig(); notice('连接已保存');
}

// ---- run config ----
function setRunUrl(name) {
  const url = new URL(location.href);
  if (name) url.searchParams.set('run', name); else url.searchParams.delete('run');
  history.replaceState(null, '', url);
}
function applyRunConfig(config) {
  if (configuration.cases[config.case]) { $('case').value = config.case; $('case').onchange(); }
  selectProvider(config.provider);
  $('seed').value = config.seed; $('max-steps').value = config.max_steps; $('obs-radius').value = config.obs_radius ?? 3;
  $('coop-planner').checked = !!config.coop_planner;
  $('num-agents').value = config.num_agents ?? configuration.cases[config.case]?.default_agents ?? 4;
}
function runConfig() {
  return {case: $('case').value, provider: $('provider').value, seed: Number($('seed').value),
    max_steps: Number($('max-steps').value), obs_radius: Number($('obs-radius').value),
    num_agents: Number($('num-agents').value), coop_planner: $('coop-planner').checked};
}
async function createRun() {
  stopPlayback();
  episode = await api('/api/reset', 'POST', runConfig());
  archived = null; activePreset = null; following = true; savedTerminal = null; liveRecords.clear(); viewRevision++;
  setRunUrl(null); markPreset();
}
async function command(fn) {
  if (commandPending) return;
  commandPending = true; viewRevision++;
  for (const id of ['run', 'step', 'reset', 'open-saved', 'live']) $(id).disabled = true;
  try { await fn(); } catch (e) { notice(e.message); } finally { commandPending = false; await guard(refresh); }
}

// ---- playback ----
const SPEEDS = [1, 2, 4];
let playTimer = null, speedIndex = 0;
function stopPlayback() {
  clearInterval(playTimer); playTimer = null; $('play').textContent = '▶ 播放'; $('play').classList.remove('primary');
}
function startPlayback() {
  stopPlayback();
  if (!episode) return;
  if (Number($('timeline').value) >= Number($('timeline').max)) $('timeline').value = 0;
  following = false; $('play').textContent = '⏸ 暂停'; $('play').classList.add('primary');
  playTimer = setInterval(() => {
    const next = Number($('timeline').value) + 1, max = Number($('timeline').max);
    if (next > max) { stopPlayback(); following = !archived; return; }
    $('timeline').value = next; guard(refresh);
  }, 600 / SPEEDS[speedIndex]);
  guard(refresh);
}

// ---- replays ----
async function openArchive(data, name, preset = null) {
  stopPlayback();
  archived = data; activePreset = preset; following = false; viewRevision++;
  applyRunConfig(data.config); setRunUrl(preset ? `preset:${preset}` : name); markPreset(); notice('');
  $('timeline').max = data.frames.length - 1; $('timeline').value = 0;
  await refresh();
  startPlayback();
}
async function openSaved(name) { await openArchive(await api(`/api/runs/${encodeURIComponent(name)}`), name); }
async function openPreset(name) { await openArchive(await api(`/api/presets/${encodeURIComponent(name)}`), null, name); }
async function loadRuns() {
  const runs = await api('/api/runs');
  $('saved-runs').replaceChildren(el('option', {value: ''}, '全部记录'));
  for (const r of runs) $('saved-runs').appendChild(el('option', {value: r.name},
    `${r.name} · ${PROVIDER[r.config.provider] || r.config.provider} · ${STATUS[r.status] || r.status}`));
}
async function loadPresets() {
  const presets = await api('/api/presets');
  $('preset-list').replaceChildren();
  for (const p of presets) {
    const b = el('button', {class: 'preset', 'data-name': p.name});
    b.appendChild(el('b', {}, p.title));
    const tags = el('div', {class: 'tags'});
    tags.appendChild(el('span', {}, PROVIDER[p.config.provider] || p.config.provider));
    tags.appendChild(el('span', {}, `${p.config.num_agents} agents`));
    if (p.config.coop_planner) tags.appendChild(el('span', {class: 'coop'}, 'COOP'));
    tags.appendChild(el('span', {class: p.metrics.CSR === 1 ? 'ok' : 'fail'},
      p.metrics.CSR === 1 ? `✓ ${p.metrics.steps} 步` : `✗ ISR ${Math.round(p.metrics.ISR * 100)}%`));
    b.appendChild(tags);
    b.onclick = () => guard(() => openPreset(p.name));
    $('preset-list').appendChild(b);
  }
  markPreset();
}
function markPreset() {
  for (const b of document.querySelectorAll('.preset')) b.classList.toggle('active', b.dataset.name === activePreset);
}

// ---- rendering ----
let backfilledFor = null, backfillAt = 0;
async function backfill(final) {
  // Polling can miss fast ticks; fetch the full record once at the end, or (throttled) when scrubbing into a gap.
  if (archived || !episode || backfilledFor === episode.id) return;
  if (!final && Date.now() - backfillAt < 1500) return;
  backfillAt = Date.now();
  const full = await api('/api/export', 'GET', undefined, 30000);
  if (full.id !== episode?.id) return;
  for (const r of full.records || []) { r.__episode = full.id; liveRecords.set(r.tick, r); }
  if (isTerminal(full)) backfilledFor = full.id;
}
function records() { return archived ? archived.records || [] : [...liveRecords.values()]; }
function recordAt(tick) {
  if (archived) return (archived.records || []).find(r => r.tick === tick) || null;
  return liveRecords.get(tick) || null;
}
function svgNode(parent, tag, attrs, text) {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  parent.appendChild(n); return n;
}
let lastDrawn = '', lastFrame = null;
function draw(frameIndex) {
  if (!episode) return;
  const key = `${episode.id}:${frameIndex}:${episode.frames.length}`;
  if (key === lastDrawn) return; lastDrawn = key;
  const frame = episode.frames[frameIndex], rows = frame.map.length, cols = frame.map[0].length;
  const stepped = lastFrame && lastFrame.id === episode.id && lastFrame.index === frameIndex - 1;
  lastFrame = {id: episode.id, index: frameIndex};
  const anims = [];
  const cell = Math.min(64, 560 / Math.max(rows, cols)), pad = 14, w = cols * cell + 2 * pad, h = rows * cell + 2 * pad;
  const svg = $('grid'); svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.replaceChildren();
  const defs = svgNode(svg, 'defs', {});
  const f = svgNode(defs, 'filter', {id: 'glow', x: '-50%', y: '-50%', width: '200%', height: '200%'});
  svgNode(f, 'feGaussianBlur', {stdDeviation: cell * .08, result: 'b'});
  const m = svgNode(f, 'feMerge', {}); svgNode(m, 'feMergeNode', {in: 'b'}); svgNode(m, 'feMergeNode', {in: 'SourceGraphic'});
  const hatch = svgNode(defs, 'pattern', {id: 'hatch', width: 8, height: 8, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)'});
  svgNode(hatch, 'rect', {width: 8, height: 8, fill: '#020308'});
  svgNode(hatch, 'line', {x1: 0, y1: 0, x2: 0, y2: 8, stroke: '#2a1d5c', 'stroke-width': 3});
  const xy = ([r, c]) => [pad + (c + .5) * cell, pad + (r + .5) * cell];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const wall = frame.map[r][c];
    svgNode(svg, 'rect', {x: pad + c * cell + 1.5, y: pad + r * cell + 1.5, width: cell - 3, height: cell - 3, rx: cell * .1,
      fill: wall ? 'url(#hatch)' : '#0b1433', stroke: wall ? '#1a1238' : 'rgba(0,240,255,.22)', 'stroke-width': 1});
  }
  frame.goals.forEach((g, i) => {
    const [x, y] = xy(g), s = cell * .62, c = NEON[i % NEON.length];
    svgNode(svg, 'rect', {x: x - s / 2, y: y - s / 2, width: s, height: s, rx: cell * .1, fill: 'none', stroke: c,
      'stroke-width': 1.6, 'stroke-dasharray': '4 3', opacity: .85, filter: 'url(#glow)'});
  });
  frame.positions.forEach((p, i) => {
    const c = NEON[i % NEON.length];
    const pts = episode.frames.slice(0, frameIndex + 1).map(fr => xy(fr.positions[i]).join(',')).join(' ');
    svgNode(svg, 'polyline', {points: pts, fill: 'none', stroke: c, 'stroke-width': cell * .07, opacity: .45,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round', filter: 'url(#glow)'});
    const [x, y] = xy(p), atGoal = p[0] === frame.goals[i][0] && p[1] === frame.goals[i][1];
    const g = svgNode(svg, 'g', {transform: `translate(${x} ${y})`});
    svgNode(g, 'circle', {cx: 0, cy: 0, r: cell * .27, fill: atGoal ? c : '#04050d', stroke: c, 'stroke-width': 2.5, filter: 'url(#glow)'});
    svgNode(g, 'text', {x: 0, y: cell * .1, 'text-anchor': 'middle', fill: atGoal ? '#04050d' : c,
      'font-size': cell * .28, 'font-family': 'ui-monospace, monospace', 'font-weight': 700}, i);
    // Slide from the previous frame when stepping forward by one tick.
    const prev = stepped ? episode.frames[frameIndex - 1].positions[i] : null;
    if (prev && (prev[0] !== p[0] || prev[1] !== p[1])) {
      const [px, py] = xy(prev);
      const anim = svgNode(g, 'animateTransform', {attributeName: 'transform', type: 'translate', from: `${px} ${py}`,
        to: `${x} ${y}`, dur: `${Math.min(.35, .5 / SPEEDS[speedIndex])}s`, begin: 'indefinite', fill: 'freeze'});
      anims.push(anim);
    }
  });
  for (const anim of anims) anim.beginElement();
  $('legend').replaceChildren(...frame.positions.map((p, i) =>
    el('span', {style: `--c:${NEON[i % NEON.length]}`}, `A${i} [${p}] → [${frame.goals[i]}]`)));
  $('frame-label').textContent = `T ${frame.step}`;
}

function qwenProbabilities(call, options) {
  const a = call?.answer;
  if (!a?.probabilities) return null;
  const actions = options.map(o => o.actions[0]).sort((x, y) => x - y);
  const out = {};
  for (const [label, p] of Object.entries(a.probabilities)) {
    const action = actions['ABCDE'.indexOf(label)];
    const opt = options.find(o => o.actions[0] === action);
    if (opt) out[opt.id] = p;
  }
  return out;
}
function renderDecisions(record, frame) {
  const box = $('agents'); box.replaceChildren();
  if (!record || !record.decision) {
    $('tick-label').textContent = record?.error ? 'ERROR' : '—'; $('wall-ms').textContent = '—';
    box.appendChild(el('div', {class: 'empty'}, record?.error || record?.discarded || '等待决策'));
    return;
  }
  $('tick-label').textContent = `T${record.tick}`;
  $('wall-ms').textContent = Math.round(record.decision.latency_ms ?? 0);
  const agents = record.decision.agents || [];
  const calls = record.calls || [];
  const perAgentMs = agents.map((d, i) => d?.latency_ms ?? 0);
  const maxMs = Math.max(1, ...perAgentMs);
  agents.forEach((d, i) => {
    const color = NEON[i % NEON.length];
    const options = record.candidates?.[i]?.options || [];
    const facts = record.input?.agents?.[i]?.candidates || {};
    const mine = calls.filter(c => c.agent_id === i);
    const tokens = mine.reduce((s, c) => s + (c.usage?.total_tokens || 0), 0);
    const retries = mine.filter(c => c.retry_after_seconds).length;
    const probs = d.probabilities || (episode.config.provider === 'qwen_rlcd' ? qwenProbabilities(mine.at(-1), options) : null);
    const pos = frame?.positions?.[i], goal = frame?.goals?.[i];
    const card = el('div', {class: 'agent', style: `--c:${color}`});
    const top = el('div', {class: 'agent-top'});
    top.appendChild(el('b', {}, `A${i}`));
    if (pos && goal && pos[0] === goal[0] && pos[1] === goal[1]) top.appendChild(el('span', {class: 'badge'}, 'GOAL'));
    if (retries) top.appendChild(el('span', {class: 'tok'}, `↻${retries}`));
    if (tokens) top.appendChild(el('span', {class: 'tok'}, `${tokens} tok`));
    top.appendChild(el('span', {class: 'ms'}, `${Math.round(d.latency_ms ?? 0)} ms`));
    card.appendChild(top);
    const bar = el('div', {class: 'latency'}); bar.appendChild(el('i', {style: `width:${100 * (d.latency_ms ?? 0) / maxMs}%`}));
    card.appendChild(bar);
    const opts = el('div', {class: 'opts'});
    for (let a = 0; a < 5; a++) {
      const o = options.find(x => x.actions[0] === a);
      const fact = o ? facts[o.id] || {} : {};
      const cls = ['opt'];
      if (!o) cls.push('none');
      if (o && o.id === d.choice) cls.push('chosen');
      if (fact.suggested_by_group_plan) cls.push('plan');
      const chip = el('div', {class: cls.join(' '), title: MOVE_NAME[a]}, MOVE[a]);
      if (o && probs && probs[o.id] !== undefined) {
        chip.appendChild(el('small', {}, `${Math.round(probs[o.id] * 100)}%`));
        const p = el('div', {class: 'p'}); p.appendChild(el('i', {style: `width:${probs[o.id] * 100}%`})); chip.appendChild(p);
      } else if (o && fact.own_map_distance_change !== undefined && fact.own_map_distance_change !== null) {
        const dc = fact.own_map_distance_change;
        chip.appendChild(el('small', {}, dc < 0 ? '−1' : dc > 0 ? '+1' : '0'));
      }
      opts.appendChild(chip);
    }
    card.appendChild(opts);
    if (d.intent) card.appendChild(el('div', {class: 'intent'}, d.intent));
    box.appendChild(card);
  });
  const i = Number($('agent-view').value);
  $('input').textContent = JSON.stringify(record.input?.agents?.[i] ?? record.input ?? {}, null, 2);
  $('trace').textContent = JSON.stringify({decision: agents[i], calls: calls.filter(c => c.agent_id === i)}, null, 2);
}
function renderSpark(currentTick) {
  const svg = $('spark'); svg.replaceChildren();
  const rs = records().filter(r => r.decision).sort((a, b) => a.tick - b.tick);
  if (rs.length < 2) return;
  const vals = rs.map(r => r.decision.latency_ms || 0), max = Math.max(...vals, 1), W = 300, H = 44;
  const pts = vals.map((v, k) => `${(k / (vals.length - 1)) * W},${H - 4 - (v / max) * (H - 10)}`).join(' ');
  svgNode(svg, 'polyline', {points: `0,${H} ${pts} ${W},${H}`, fill: 'rgba(0,240,255,.08)', stroke: 'none'});
  svgNode(svg, 'polyline', {points: pts, fill: 'none', stroke: '#00f0ff', 'stroke-width': 1.5});
  const k = rs.findIndex(r => r.tick === currentTick);
  if (k >= 0) svgNode(svg, 'circle', {cx: (k / (vals.length - 1)) * W, cy: H - 4 - (vals[k] / max) * (H - 10), r: 3, fill: '#ff2bd6'});
}
function emptyView() {
  lastDrawn = ''; $('run-title').textContent = '—'; $('adapter-version').textContent = '';
  $('status').textContent = STATUS.ready; $('status').className = 'status ready';
  for (const [id, v] of [['csr', '—'], ['isr', '—'], ['steps', '0'], ['calls', '0'], ['cost', '—'], ['tokens', '0']]) $(id).textContent = v;
  $('grid').replaceChildren(); $('legend').replaceChildren(); $('timeline').value = 0; $('timeline').max = 0;
  $('frame-label').textContent = 'T 0'; renderDecisions(null); renderSpark(-1);
  $('run').textContent = '▶ 运行'; $('run').disabled = commandPending; $('reset').disabled = commandPending;
  for (const id of ['step', 'pause', 'stop', 'export']) $(id).disabled = true;
}
async function refresh() {
  const revision = viewRevision;
  const data = archived ? {running: false, episode: archived} : await api('/api/state');
  if (revision !== viewRevision) return;
  episode = data.episode;
  $('open-saved').disabled = commandPending; $('live').disabled = commandPending || !archived;
  if (!episode) { emptyView(); return; }
  if (!archived && episode.last_record) {
    const r = episode.last_record; r.__episode = episode.id;
    for (const [t, old] of liveRecords) if (old.__episode !== episode.id) liveRecords.delete(t);
    liveRecords.set(r.tick, r);
  }
  const cfg = episode.config, m = episode.metrics;
  $('run-title').textContent = `${configuration.cases[cfg.case]?.name || cfg.case} · ${PROVIDER[cfg.provider] || cfg.provider}`
    + `${cfg.coop_planner ? ' · COOP' : ''}${archived ? ' · REPLAY' : ''}`;
  $('adapter-version').textContent = cfg.prompt_version || '';
  const running = data.running && !episode.busy;
  $('status').textContent = running ? STATUS.running : STATUS[episode.status] || episode.status;
  $('status').className = `status ${running ? 'running' : episode.status}`;
  $('csr').textContent = m.CSR === 1 ? '✓' : '✗'; $('csr').className = m.CSR === 1 ? 'ok' : '';
  $('isr').textContent = `${Math.round(m.ISR * 100)}%`; $('steps').textContent = m.steps;
  $('calls').textContent = m.model_calls; $('cost').textContent = `${m.SoC ?? '—'} / ${m.makespan ?? '—'}`;
  $('tokens').textContent = (m.total_tokens || 0).toLocaleString();
  const n = episode.frames[0].positions.length;
  if ($('agent-view').options.length !== n) $('agent-view').replaceChildren(...Array.from({length: n}, (_, i) => el('option', {value: i}, `A${i}`)));
  $('timeline').max = episode.frames.length - 1;
  if (following) $('timeline').value = episode.frames.length - 1;
  const frameIndex = Math.min(Number($('timeline').value), episode.frames.length - 1);
  draw(frameIndex);
  // Decision that produced this frame; at T0 show the first decision, live shows the newest record.
  const shown = recordAt(Math.max(1, frameIndex)) || (following ? episode.last_record : null);
  renderDecisions(shown, episode.frames[frameIndex]); renderSpark(shown?.tick ?? -1);
  if (!shown && episode.error) notice(episode.error);
  const terminal = isTerminal(episode);
  $('run').textContent = archived || terminal ? '▶ 重新运行' : '▶ 运行';
  $('run').disabled = commandPending || (!archived && (data.running || episode.busy));
  $('step').disabled = commandPending || !!archived || data.running || episode.busy || terminal;
  $('reset').disabled = commandPending || (!archived && (data.running || episode.busy));
  $('pause').disabled = !!archived || terminal; $('stop').disabled = !!archived || terminal; $('export').disabled = false;
  if (terminal && !archived && episode.error) notice(episode.error);
  if (terminal && savedTerminal !== episode.id) { savedTerminal = episode.id; await loadRuns(); }
  if (!archived && (terminal ? backfilledFor !== episode.id : !shown && frameIndex > 0)) {
    await backfill(terminal); lastDrawn = '';
    const again = recordAt(Math.max(1, frameIndex));
    if (again) { renderDecisions(again, episode.frames[frameIndex]); renderSpark(again.tick); }
  }
}

// ---- wiring ----
for (const b of document.querySelectorAll('#models button')) b.onclick = () => selectProvider(b.dataset.provider);
$('provider').onchange = () => selectProvider($('provider').value);
$('case').onchange = () => {
  const c = configuration.cases[$('case').value];
  $('num-agents').value = c.default_agents; $('num-agents').disabled = !!c.fixed_agents;
};
$('open-connection').onclick = () => $('connection').showModal();
$('agent-view').onchange = () => { lastDrawn = ''; guard(refresh); };
$('open-saved').onclick = () => guard(async () => { if ($('saved-runs').value) await openSaved($('saved-runs').value); });
$('save-connection').onclick = () => guard(saveConnection);
$('test-connection').onclick = () => guard(async () => {
  const p = $('provider').value; $('test-connection').disabled = true;
  try {
    await saveConnection(); notice('测试中…');
    const r = await api(`/api/connections/${p}/test`, 'POST', undefined, 400000);
    await loadConfig(); notice(r.ok ? `连接正常 · ${r.model_calls} 次调用` : `失败：${r.error}`);
  } finally { $('test-connection').disabled = false; }
});
$('import-deepseek').onclick = () => guard(async () => { await api('/api/connections/import-deepseek', 'POST'); await loadConfig(); notice('已读取'); });
$('reset').onclick = () => command(async () => { await createRun(); notice(''); });
for (const action of ['run', 'step']) $(action).onclick = () => command(async () => {
  if (action === 'run' && (archived || !episode || isTerminal(episode))) await createRun();
  await api(`/api/control/${action}`, 'POST'); notice('');
});
for (const action of ['pause', 'stop']) $(action).onclick = () => guard(async () => { await api(`/api/control/${action}`, 'POST'); await refresh(); });
$('timeline').oninput = () => { stopPlayback(); following = Number($('timeline').value) === Number($('timeline').max); guard(refresh); };
$('play').onclick = () => playTimer ? stopPlayback() : startPlayback();
$('speed').onclick = () => {
  speedIndex = (speedIndex + 1) % SPEEDS.length; $('speed').textContent = `${SPEEDS[speedIndex]}×`;
  if (playTimer) startPlayback();
};
$('live').onclick = () => guard(async () => {
  stopPlayback(); archived = null; activePreset = null; following = true; viewRevision++; setRunUrl(null); markPreset();
  await refresh(); if (episode) applyRunConfig(episode.config);
});
$('export').onclick = () => guard(async () => {
  const r = archived || await api('/api/export');
  const url = URL.createObjectURL(new Blob([JSON.stringify(r, null, 2)], {type: 'application/json'}));
  const a = el('a', {href: url, download: `jev-mapf-${r.id}.json`}); a.click(); URL.revokeObjectURL(url);
});

await guard(async () => {
  await loadConfig();
  for (const [id, c] of Object.entries(configuration.cases).sort((a, b) => Number(b[1].official) - Number(a[1].official)))
    $('case').appendChild(el('option', {value: id}, c.name));
  $('case').value = configuration.cases['puzzle-03'] ? 'puzzle-03' : $('case').value;
  $('case').onchange(); selectProvider('deepseek');
  await Promise.all([loadRuns(), loadPresets()]);
  const selected = new URLSearchParams(location.search).get('run');
  if (selected?.startsWith('preset:')) await openPreset(selected.slice(7));
  else if (selected) await openSaved(selected);
  else { await refresh(); if (episode) applyRunConfig(episode.config); }
  initialized = true;
});
document.querySelector('main').inert = false;
document.documentElement.dataset.ready = String(initialized);
async function poll() { if (initialized && !archived && !commandPending) await guard(refresh); setTimeout(poll, 800); }
setTimeout(poll, 800);
