// ── State ─────────────────────────────────────────────────────────────────────

let currentSet       = null;
let currentTrackId   = null;
let playbackProgress = 0;    // seconds
let playbackDuration = 0;    // seconds
let lastPollAt       = 0;    // Date.now() when we last polled
let isPlaying        = false;
let playerPollInterval = null;
let rafId            = null;

// ── Canvas refs (set after DOM ready) ────────────────────────────────────────

let zoomCanvas, overviewCanvas;

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  zoomCanvas     = document.getElementById('wf-zoom');
  overviewCanvas = document.getElementById('wf-overview');
  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);

  const params = new URLSearchParams(window.location.search);
  if (params.get('error')) {
    showToast('Could not connect to Spotify. Try again.', true);
    window.history.replaceState({}, '', '/');
  }

  try {
    const me = await api('/api/me');
    showApp(me);
  } catch {
    document.getElementById('login-screen').style.display = 'flex';
  }
}

function showApp(me) {
  document.getElementById('login-screen').style.display = 'none';
  document.getElementById('app').style.display = 'flex';
  requestAnimationFrame(resizeCanvases); // size after browser lays out the panels

  const badge = document.getElementById('user-badge');
  badge.innerHTML = `
    ${me.image ? `<img src="${me.image}" alt="">` : ''}
    <span>${me.name || me.email || 'Connected'}</span>
    <a href="/logout">Disconnect</a>
  `;
  startPlayerPoll();
}

// ── Canvas sizing ─────────────────────────────────────────────────────────────

function resizeCanvases() {
  if (!zoomCanvas || !overviewCanvas) return;

  // Read the actual rendered size of each wrapper and match the canvas buffer.
  // This avoids HiDPI/scale conflicts — 1 canvas pixel = 1 CSS pixel.
  const sizeCanvas = (cv) => {
    const rect = cv.parentElement.getBoundingClientRect();
    const w = Math.round(rect.width)  || 256;
    const h = Math.round(rect.height) || 80;
    if (cv.width !== w || cv.height !== h) {
      cv.width  = w;
      cv.height = h;
    }
    // Paint background so canvas is visibly present
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#0d0d0d';
    ctx.fillRect(0, 0, w, h);
  };

  sizeCanvas(zoomCanvas);
  sizeCanvas(overviewCanvas);
}

// ── Waveform rendering ────────────────────────────────────────────────────────

function energyColor(amp) {
  if (amp < 0.25) return '#1d4ed8';
  if (amp < 0.45) return '#6d28d9';
  if (amp < 0.65) return '#a855f7';
  if (amp < 0.80) return '#f59e0b';
  return '#ef4444';
}

function drawZoom(waveform, beats, sections, currentTime, duration) {
  const cv  = zoomCanvas;
  const ctx = cv.getContext('2d');
  const W   = cv.width;
  const H   = cv.height;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, W, H);

  if (!waveform || waveform.length === 0) return;

  const windowSec = 8;
  const tStart    = Math.max(0, currentTime - windowSec / 2);
  const tEnd      = currentTime + windowSec / 2;
  const startRatio = tStart / Math.max(duration, 1);
  const endRatio   = Math.min(1, tEnd / Math.max(duration, 1));
  const si = Math.floor(startRatio * waveform.length);
  const ei = Math.ceil(endRatio   * waveform.length);
  const slice = waveform.slice(si, ei);
  if (slice.length === 0) return;

  // Section backgrounds
  (sections || []).forEach(s => {
    const sx = ((s.s - startRatio) / (endRatio - startRatio)) * W;
    const sw = (s.d / (endRatio - startRatio)) * W;
    ctx.fillStyle = `rgba(168,85,247,${(s.l * 0.12).toFixed(3)})`;
    ctx.fillRect(sx, 0, sw, H);
  });

  // Waveform bars
  const barW = W / slice.length;
  slice.forEach((amp, i) => {
    const x    = i * barW;
    const barH = amp * H * 0.88;
    const y    = (H - barH) / 2;
    ctx.fillStyle = energyColor(amp);
    ctx.fillRect(x, y, Math.max(0.8, barW - 0.5), barH);
  });

  // Beat grid
  const secPerPixel = windowSec / W;
  (beats || []).filter(b => b >= tStart && b <= tEnd).forEach(b => {
    const x = ((b - tStart) / windowSec) * W;
    ctx.fillStyle = 'rgba(255,255,255,0.12)';
    ctx.fillRect(x, 0, 1, H);
  });

  // Played-over overlay (left half tinted)
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  ctx.fillRect(0, 0, W / 2, H);
}

function drawOverview(waveform, currentTime, duration) {
  const cv  = overviewCanvas;
  const ctx = cv.getContext('2d');
  const W   = cv.width;
  const H   = cv.height;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, W, H);

  if (!waveform || waveform.length === 0) return;

  const barW = W / waveform.length;
  waveform.forEach((amp, i) => {
    const x    = i * barW;
    const barH = amp * H * 0.88;
    const y    = (H - barH) / 2;
    ctx.fillStyle = energyColor(amp);
    ctx.fillRect(x, y, Math.max(0.5, barW - 0.3), barH);
  });

  // Played overlay
  if (duration > 0) {
    const px = (currentTime / duration) * W;
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, px, H);

    // Playhead
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.fillRect(px - 1, 0, 2, H);
  }
}

function miniWaveformSVG(waveform, w, h) {
  if (!waveform || waveform.length === 0) return '';
  const step = w / waveform.length;
  const bars = waveform.map((v, i) => {
    const x    = (i * step).toFixed(1);
    const bh   = (v * h * 0.9).toFixed(1);
    const y    = ((h - bh) / 2).toFixed(1);
    const bw   = Math.max(0.4, step - 0.4).toFixed(1);
    return `<rect x="${x}" y="${y}" width="${bw}" height="${bh}" fill="${energyColor(v)}"/>`;
  }).join('');
  return `<svg width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg">${bars}</svg>`;
}

// ── Animation loop ────────────────────────────────────────────────────────────

function startAnimation() {
  if (rafId) cancelAnimationFrame(rafId);

  function frame() {
    if (!isPlaying || !currentTrackId) { rafId = null; return; }

    const elapsed = (Date.now() - lastPollAt) / 1000;
    const t = Math.min(playbackProgress + elapsed, playbackDuration);

    const track = currentSet?.tracks.find(tr => tr.id === currentTrackId);
    if (track?.waveform) {
      drawZoom(track.waveform, track.beats, track.sections, t, playbackDuration);
      drawOverview(track.waveform, t, playbackDuration);
    }

    // Smooth progress bar
    if (playbackDuration > 0) {
      const pct = (t / playbackDuration) * 100;
      document.getElementById('progress-fill').style.width = pct + '%';
      document.getElementById('time-cur').textContent = fmtSec(t);
    }

    rafId = requestAnimationFrame(frame);
  }
  rafId = requestAnimationFrame(frame);
}

// ── Chat ──────────────────────────────────────────────────────────────────────

function handleKey(e) { if (e.key === 'Enter') sendVibe(); }

async function sendVibe() {
  const input = document.getElementById('vibe-input');
  const vibe  = input.value.trim();
  if (!vibe) return;

  input.value = '';
  setInputEnabled(false);
  addUserMessage(vibe);
  const loadingEl = addLoadingMessage();

  try {
    const result = await api('/api/generate-set', 'POST', { vibe });
    loadingEl.remove();
    currentSet = result;
    renderSet(result);
    addDJMessage(`"${result.set_name}" — ${result.dj_intro}`);
  } catch (err) {
    loadingEl.remove();
    addDJMessage(`Couldn't build that set. ${err.message || 'Try again.'}`);
  } finally {
    setInputEnabled(true);
    input.focus();
  }
}

function addUserMessage(text) {
  const chat = document.getElementById('chat-messages');
  const el   = document.createElement('div');
  el.className = 'user-message';
  el.innerHTML = `<div class="user-bubble">${esc(text)}</div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function addDJMessage(text) {
  const chat = document.getElementById('chat-messages');
  const el   = document.createElement('div');
  el.className = 'dj-message';
  el.innerHTML = `<div class="dj-avatar">⚡</div><div class="dj-bubble">${esc(text)}</div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function addLoadingMessage() {
  const chat = document.getElementById('chat-messages');
  const el   = document.createElement('div');
  el.className = 'dj-message';
  el.innerHTML = `
    <div class="dj-avatar">⚡</div>
    <div class="dj-bubble">
      <div class="loading-dots"><span></span><span></span><span></span></div>
      Analyzing your taste and building the set…
    </div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function setInputEnabled(on) {
  document.getElementById('vibe-input').disabled = !on;
  document.getElementById('send-btn').disabled   = !on;
}

// ── Render set ────────────────────────────────────────────────────────────────

function renderSet(set) {
  const list    = document.getElementById('set-list');
  const nameEl  = document.getElementById('set-name-display');
  const metaEl  = document.getElementById('set-meta');
  const actions = document.getElementById('set-actions');

  nameEl.textContent = set.set_name;

  const totalMs  = set.tracks.reduce((s, t) => s + (t.duration_ms || 0), 0);
  const discCount = set.tracks.filter(t => !t.familiar).length;
  const bpmVals   = set.tracks.map(t => t.bpm).filter(Boolean);
  const avgBpm    = bpmVals.length ? Math.round(bpmVals.reduce((a, b) => a + b, 0) / bpmVals.length) : null;

  metaEl.textContent = [
    `${set.tracks.length} tracks`,
    `${Math.round(totalMs / 60000)} min`,
    discCount ? `${discCount} new` : null,
    avgBpm    ? `avg ${avgBpm} BPM` : null,
  ].filter(Boolean).join(' · ');

  list.innerHTML = '';
  set.tracks.forEach((track, i) => {
    list.appendChild(buildTrackItem(track, i));

    // Transition connector between tracks
    if (i < set.tracks.length - 1) {
      const next = set.tracks[i + 1];
      list.appendChild(buildConnector(track, next));
    }
  });

  actions.style.display = 'block';

  // Preview first track's waveform immediately
  const first = set.tracks[0];
  if (first?.waveform?.length) {
    loadTrackWaveform(first);
    const dur = (first.duration_ms || 0) / 1000;
    drawZoom(first.waveform, first.beats, first.sections, 0, dur);
    drawOverview(first.waveform, 0, dur);
  }
}

function buildTrackItem(track, index) {
  const el = document.createElement('div');
  el.className = 'track-item';
  el.dataset.trackId = track.id;

  const bpm     = track.bpm ? Math.round(track.bpm) : null;
  const camelot = track.camelot || '';
  const keyClass = camelot.endsWith('A') ? 'key-A' : camelot.endsWith('B') ? 'key-B' : '';
  const miniSvg = miniWaveformSVG(track.waveform, 80, 20);

  el.innerHTML = `
    <div class="track-num">${index + 1}</div>
    ${track.album_art
      ? `<img class="track-thumb" src="${track.album_art}" alt="" loading="lazy">`
      : `<div class="track-thumb"></div>`}
    <div class="track-body">
      <div class="track-title">${esc(track.name)}</div>
      <div class="track-artist-small">${esc(track.artist)}</div>
      <div class="track-wf-row">
        <div class="mini-wf">${miniSvg}</div>
        ${bpm     ? `<span class="beat-chip chip-bpm">${bpm}</span>` : ''}
        ${camelot ? `<span class="beat-chip chip-key ${keyClass}">${camelot}</span>` : ''}
        ${!track.familiar ? '<span class="discovery-badge">★ NEW</span>' : ''}
      </div>
      ${track.note ? `<div class="track-note">"${esc(track.note)}"</div>` : ''}
    </div>`;
  return el;
}

function buildConnector(a, b) {
  const el = document.createElement('div');
  const compat = camelotCompat(a.camelot, b.camelot);
  el.className = `track-connector conn-${compat}`;

  const bpmDiff = a.bpm && b.bpm ? Math.abs(Math.round(a.bpm) - Math.round(b.bpm)) : null;
  const bpmStr  = bpmDiff !== null ? `±${bpmDiff} BPM` : '';
  const keyStr  = a.camelot && b.camelot ? `${a.camelot}→${b.camelot}` : '';
  const label   = [keyStr, bpmStr].filter(Boolean).join(' · ') || '↓';

  el.innerHTML = `<div class="conn-dot"></div><span>${label}</span>`;
  return el;
}

function camelotCompat(c1, c2) {
  if (!c1 || !c2 || c1 === '?' || c2 === '?') return 'unknown';
  try {
    const n1 = parseInt(c1), l1 = c1.slice(-1);
    const n2 = parseInt(c2), l2 = c2.slice(-1);
    if (n1 === n2) return 'perfect';
    if (l1 === l2) {
      const d = Math.abs(n1 - n2);
      if (d === 1 || d === 11) return 'perfect';
      if (d === 2 || d === 10) return 'good';
    }
    return 'clash';
  } catch { return 'unknown'; }
}

// ── Play set ──────────────────────────────────────────────────────────────────

async function dropSet() {
  if (!currentSet) return;
  const btn = document.getElementById('drop-btn');
  btn.disabled = true;
  btn.innerHTML = '<span>Dropping…</span>';

  const uris = currentSet.tracks.map(t => t.uri).filter(Boolean);
  try {
    await api('/api/play-set', 'POST', { track_uris: uris });
    showToast('Set is playing on Spotify!');
    startPlayerPoll();
  } catch (err) {
    showToast(err.message || 'Could not start playback. Is Spotify open?', true);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>▶ DROP THE SET</span>';
  }
}

// ── Player controls ───────────────────────────────────────────────────────────

async function togglePlay() {
  try { await api('/api/player/control', 'POST', { action: isPlaying ? 'pause' : 'play' }); }
  catch (err) { showToast(err.message || 'Playback control failed', true); }
}

async function control(action) {
  try { await api('/api/player/control', 'POST', { action }); }
  catch (err) { showToast(err.message || 'Playback control failed', true); }
}

// ── Player state polling ──────────────────────────────────────────────────────

function startPlayerPoll() {
  if (playerPollInterval) return;
  updatePlayerState();
  playerPollInterval = setInterval(updatePlayerState, 5000);
}

async function updatePlayerState() {
  try {
    const state = await api('/api/player/state');
    renderPlayerState(state);
  } catch { /* silent */ }
}

function renderPlayerState(state) {
  const wasPlaying = isPlaying;
  isPlaying = state.playing;

  document.getElementById('play-btn').textContent = isPlaying ? '⏸' : '▶';

  if (state.track_name) {
    document.getElementById('now-track').textContent  = state.track_name;
    document.getElementById('now-artist').textContent = state.artist;
    document.getElementById('now-device').textContent = state.device ? `On: ${state.device}` : '';
    if (state.album_art) {
      document.getElementById('np-art').style.backgroundImage = `url(${state.album_art})`;
    }
  }

  playbackDuration = (state.duration_ms || 0) / 1000;
  playbackProgress = (state.progress_ms || 0) / 1000;
  lastPollAt       = Date.now();

  document.getElementById('time-dur').textContent = fmtSec(playbackDuration);

  // Track changed — update waveform panel
  if (state.track_id && state.track_id !== currentTrackId) {
    currentTrackId = state.track_id;
    const track = currentSet?.tracks.find(t => t.id === state.track_id);
    if (track) loadTrackWaveform(track);

    // Highlight active track in set list
    document.querySelectorAll('.track-item').forEach(el => {
      el.classList.toggle('active', el.dataset.trackId === state.track_id);
    });
  }

  if (isPlaying && !wasPlaying) startAnimation();
  if (!isPlaying && rafId) { cancelAnimationFrame(rafId); rafId = null; }

  // Static draw when paused
  if (!isPlaying) {
    const track = currentSet?.tracks.find(t => t.id === currentTrackId);
    if (track?.waveform) {
      drawZoom(track.waveform, track.beats, track.sections, playbackProgress, playbackDuration);
      drawOverview(track.waveform, playbackProgress, playbackDuration);
    }
    document.getElementById('progress-fill').style.width =
      playbackDuration > 0 ? (playbackProgress / playbackDuration * 100) + '%' : '0%';
    document.getElementById('time-cur').textContent = fmtSec(playbackProgress);
  }
}

function loadTrackWaveform(track) {
  // Update beat readout
  document.getElementById('bpm-val').textContent    = track.bpm ? Math.round(track.bpm) : '--';
  const keyEl = document.getElementById('key-val');
  keyEl.textContent  = track.camelot || '--';
  keyEl.className    = 'beat-val camelot-val ' + (track.camelot?.endsWith('A') ? 'key-A' : track.camelot?.endsWith('B') ? 'key-B' : '');
  document.getElementById('source-val').textContent = track.wf_source || '--';
}

// ── Utilities ─────────────────────────────────────────────────────────────────

async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error(data.detail || `Request failed (${r.status})`);
  }
  return r.json().catch(() => ({}));
}

function fmtSec(s) {
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

let toastTimer = null;
function showToast(msg, error = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = 'toast show' + (error ? ' error' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3500);
}

// ── Boot ──────────────────────────────────────────────────────────────────────

init();
