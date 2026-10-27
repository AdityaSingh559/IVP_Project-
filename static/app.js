(() => {
  'use strict';

  // ------------------------------------------------------------------ helpers
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  function h(tag, attrs = {}, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  async function api(url, opts) {
    let res;
    try { res = await fetch(url, opts); }
    catch { throw new ApiError('Could not reach the server. Make sure it is still running.', 0); }
    if (!res.ok) {
      let msg = `Request failed (${res.status}).`;
      try { msg = (await res.json()).detail || msg; } catch { /* non-JSON error */ }
      throw new ApiError(msg, res.status);
    }
    return res.json();
  }

  let toastTimer;
  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 4200);
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
  const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

  // ------------------------------------------------------------------ state
  const S = {
    session: null,
    steps: [],
    cards: {},
    runToken: 0,
    unlocked: new Set(['home', 'catalogue', 'upload', 'leak']),
  };

  // ------------------------------------------------------------------ views
  const VIEWS = ['home', 'catalogue', 'upload', 'leak', 'pipeline', 'analysis', 'results'];

  function show(view, { scroll = true } = {}) {
    if (!VIEWS.includes(view) || !S.unlocked.has(view)) view = 'home';
    for (const v of VIEWS) document.getElementById(v).hidden = v !== view;
    $$('.subnav a').forEach((a) => {
      a.classList.toggle('active', a.dataset.view === view);
      const open = S.unlocked.has(a.dataset.view);
      a.classList.toggle('locked', !open);
      if (open) a.removeAttribute('aria-disabled'); else a.setAttribute('aria-disabled', 'true');
    });
    if (location.hash !== `#${view}`) history.replaceState(null, '', `#${view}`);
    if (scroll) window.scrollTo({ top: 0, behavior: 'auto' });
  }

  const unlock = (...v) => v.forEach((x) => S.unlocked.add(x));
  const lock = (...v) => v.forEach((x) => S.unlocked.delete(x));

  function initNav() {
    $$('.subnav a, #getStarted').forEach((a) => a.addEventListener('click', (e) => {
      e.preventDefault();
      const v = a.dataset.view || a.getAttribute('href').slice(1);
      if (S.unlocked.has(v)) show(v);
    }));
    addEventListener('hashchange', () => show(location.hash.slice(1) || 'home'));
  }

  // ------------------------------------------------------------------ home
  async function initHome() {
    try {
      const d = await api('/api/steps');
      S.steps = d.steps;
    } catch (e) { toast(e.message); }
    const stats = $('#stats');
    const add = (n, label) => stats.append(h('div', { class: 'stat' }, h('b', {}, n), h('span', {}, label)));
    add(String(S.steps.length || 12), 'pipeline steps');
    add('4', 'battery types');
    add('0', 'AI models');
  }

  // ------------------------------------------------------------------ catalogue
  const SHAPE = { cylinder: 'cylinder (lying on its side)', button: 'round cell (face up)' };

  function renderCatalogue(d) {
    $('#catSource').textContent = `${d.rows.length} batteries, from ${d.source}`;
    const warn = $('#catWarn');
    warn.hidden = !d.errors.length;
    warn.textContent = d.errors.length ? `Some rows were skipped: ${d.errors.join('; ')}` : '';
    const fmt = (v) => (v == null || v === '' ? '–' : `${Number(v).toFixed(1)} mm`);
    $('#catBody').replaceChildren(...d.rows.map((r) => h('tr', {},
      h('td', { class: 'name' }, r.name), h('td', {}, SHAPE[r.type] || r.type),
      h('td', { class: 'num' }, r.type === 'cylinder' ? fmt(r.size_mm) : '–'),
      h('td', { class: 'num' }, r.type === 'cylinder' ? fmt(r.second_mm) : fmt(r.size_mm)),
      h('td', { class: 'num' }, r.price))));
  }

  function catError(msg) { $('#catError').textContent = msg; $('#catError').hidden = !msg; }

  async function initCatalogue() {
    try { renderCatalogue(await api('/api/catalogue')); } catch (e) { catError(e.message); }
    $('#catFile').addEventListener('change', async (e) => {
      const f = e.target.files[0];
      e.target.value = '';
      if (!f) return;
      const fd = new FormData();
      fd.append('file', f);
      catError('');
      try { renderCatalogue(await api('/api/catalogue', { method: 'POST', body: fd })); toast('Catalogue loaded.'); }
      catch (err) { catError(err.message); }
    });
    const loadSet = async (name, msg) => {
      catError('');
      try { renderCatalogue(await api(`/api/catalogue/sample?name=${name}`, { method: 'POST' })); toast(msg); }
      catch (err) { catError(err.message); }
    };
    $('#catHouse').addEventListener('click', () => loadSet('battery', 'Catalogue reset to the four batteries.'));
  }

  // ------------------------------------------------------------------ upload
  const drop = $('#drop');
  const fileInput = $('#fileInput');

  function showUploadError(msg) { $('#uploadError').textContent = msg; $('#uploadError').hidden = !msg; }

  function validate(file) {
    if (!file) return 'No file selected.';
    const okExt = /\.(jpe?g|png|tiff?|webp|bmp)$/i.test(file.name);
    if (!/^image\/(jpeg|png|tiff|webp|bmp)$/i.test(file.type) && !okExt) return `"${file.name}" is not a JPG, PNG, TIFF, WebP or BMP image.`;
    if (file.size > 20 * 1024 * 1024) return 'That file is larger than 20 MB.';
    if (file.size === 0) return 'That file is empty.';
    return null;
  }

  async function handleFile(file) {
    const err = validate(file);
    showUploadError(err);
    if (err) return;
    stopCamera();
    const fd = new FormData();
    fd.append('file', file);
    drop.classList.add('busy');
    try {
      toast('Reading your photo...');
      const d = await api('/api/upload', { method: 'POST', body: fd });
      resetPipeline();
      S.session = d.session;
      S.steps = d.steps;
      showPreview(d);
    } catch (e) {
      showUploadError(e.message);
    } finally {
      drop.classList.remove('busy');
    }
  }

  function showPreview(d) {
    const i = d.info;
    $('#previewImg').src = d.original_url;
    const facts = [
      ['Format', i.format],
      ['File size', `${i.file_kb} KB`],
      ['Size', `${i.size[0]} x ${i.size[1]} px` + (i.resized ? ` (scaled from ${i.orig_size[0]} x ${i.orig_size[1]})` : '')],
      ['As a matrix', `${i.size[1]} x ${i.size[0]} x 3`],
      ['Catalogue', `${d.catalogue_rows} rows`],
      ['Scale from', 'credit card, 85.60 x 53.98 mm'],
    ];
    $('#facts').replaceChildren(...facts.map(([k, v]) => h('div', {}, h('dt', {}, k), h('dd', {}, v))));
    $('#preview').hidden = false;
    $('#preview').scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  async function loadSamples() {
    try {
      const list = await api('/api/samples');
      $('#sampleRow').replaceChildren(...list.map((s) => h('button', {
        type: 'button', onclick: async () => {
          try {
            const blob = await (await fetch(s.url)).blob();
            handleFile(new File([blob], s.name, { type: 'image/jpeg' }));
          } catch { showUploadError('Could not load that sample.'); }
        },
      }, h('img', { src: s.url, alt: s.name, loading: 'lazy' }), h('span', {}, s.name))));
      if (!list.length) $('.samples').hidden = true;
    } catch { $('.samples').hidden = true; }
  }

  function initUpload() {
    drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
    ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
    fileInput.addEventListener('change', () => { handleFile(fileInput.files[0]); fileInput.value = ''; });
    $('#phoneInput').addEventListener('change', (e) => { handleFile(e.target.files[0]); e.target.value = ''; });
    $('#changeBtn').addEventListener('click', () => { $('#preview').hidden = true; window.scrollTo({ top: 0 }); });
    $('#startBtn').addEventListener('click', startPipeline);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      $('#camOpen').disabled = true;
      $('#camOpen').title = 'Live camera needs a secure (https or localhost) page';
    }
    $('#camOpen').addEventListener('click', openCamera);
    $('#camClose').addEventListener('click', stopCamera);
    $('#camShot').addEventListener('click', capture);
    loadSamples();
  }

  let stream = null;
  async function openCamera() {
    showUploadError('');
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 } }, audio: false });
    } catch (e) {
      const why = e && e.name === 'NotAllowedError' ? 'permission was denied' : 'no camera was found or it is in use';
      showUploadError(`Could not open the camera: ${why}. You can upload a file or use the phone camera button instead.`);
      return;
    }
    $('#camVideo').srcObject = stream;
    $('#camIdle').hidden = true;
    $('#camLive').hidden = false;
  }
  function stopCamera() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    $('#camVideo').srcObject = null;
    $('#camLive').hidden = true;
    $('#camIdle').hidden = false;
  }
  function capture() {
    const v = $('#camVideo');
    if (!v.videoWidth) { toast('The camera is still starting, try again in a moment.'); return; }
    const c = $('#camCanvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    c.toBlob((blob) => {
      if (!blob) { showUploadError('Could not capture the photo.'); return; }
      handleFile(new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' }));
    }, 'image/jpeg', 0.95);
  }

  // ------------------------------------------------------------------ pipeline
  function resetPipeline() {
    S.runToken += 1;
    S.session = null; S.cards = {};
    $('#pipeCards').replaceChildren();
    $('#progressList').replaceChildren();
    $('#analysisBody').replaceChildren();
    $('#resultsBody').replaceChildren();
    $('#pipeEnd').hidden = true;
    lock('pipeline', 'analysis', 'results');
  }

  const SYMBOL = { done: '✓', skipped: '–', current: '', pending: '' };

  function buildProgress() {
    $('#progressList').replaceChildren(...S.steps.map((s) => h('li', { id: `pg-${s.id}`, class: 'locked' },
      h('a', { href: `#step-${s.id}`, onclick: (e) => { e.preventDefault(); S.cards[s.id]?.el.scrollIntoView({ behavior: 'smooth', block: 'start' }); } },
        h('span', { class: 'dot' }), h('span', { class: 'lbl' }, s.title)))));
  }

  function setProgress(id, state) {
    const li = $(`#pg-${id}`);
    if (!li) return;
    li.className = state === 'pending' ? 'locked' : state;
    $('.dot', li).textContent = SYMBOL[state] ?? '';
  }

  function setBar(done, total, text) {
    $('#runText').textContent = text;
    $('#runCount').textContent = `${done} of ${total} steps`;
    $('#barFill').style.width = `${(done / total) * 100}%`;
  }

  async function startPipeline() {
    if (!S.session) return;
    // a session runs once; uploading again starts a new one
    const token = ++S.runToken;
    S.cards = {};
    $('#pipeCards').replaceChildren();
    $('#pipeEnd').hidden = true;
    $('#startBtn').disabled = true;
    lock('analysis', 'results');
    unlock('pipeline');
    buildProgress();
    setBar(0, S.steps.length, 'Getting ready...');
    show('pipeline');
    let done = 0;
    for (const step of S.steps) {
      if (token !== S.runToken) return;
      setProgress(step.id, 'current');
      setBar(done, S.steps.length, `${step.title}...`);
      const card = addCard(step);
      let r;
      try {
        await sleep(250);
        r = await api(`/api/s/${S.session}/step/${step.id}`, { method: 'POST' });
      } catch (e) {
        if (token !== S.runToken) return;
        if (e.status === 404) return sessionExpired();
        return runFailed(e.message);
      }
      if (token !== S.runToken) return;
      fillCard(card, step, r);
      setProgress(step.id, r.status === 'applied' ? 'done' : 'skipped');
      done += 1;
      setBar(done, S.steps.length, done === S.steps.length ? 'All steps finished' : `${step.title} done`);
    }
    await finish(token);
  }

  function runFailed(msg) {
    $('#runText').textContent = 'Something went wrong';
    $('#pipeCards').append(h('div', { class: 'err', role: 'alert' }, msg,
      h('div', { class: 'row' }, h('button', { class: 'btn btn-primary', type: 'button', onclick: () => { show('upload'); } }, 'Upload the photo again'))));
  }

  function sessionExpired() {
    toast('This session has expired. Please upload the photo again.');
    resetPipeline();
    $('#preview').hidden = true;
    show('upload');
  }

  function addCard(step) {
    const i = S.steps.indexOf(step);
    const pill = h('span', { class: 'pill working' }, 'Working');
    const body = h('div', { class: 'body' }, h('div', { class: 'working' }, h('span', { class: 'spin' }), 'Working on your photo...'));
    const el = h('article', { class: 'card step is-working', id: `step-${step.id}` },
      h('div', { class: 'step-head' }, h('span', { class: 'step-num' }, i + 1), h('h3', {}, step.title), pill),
      h('p', { class: 'step-desc' }, step.desc),
      body);
    $('#pipeCards').append(el);
    const c = { el, pill, body };
    S.cards[step.id] = c;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return c;
  }

  function figure(url, label, cls = '') {
    return h('figure', {},
      h('div', { class: 'img-box' }, h('img', { src: url, alt: `${label} image`, loading: 'lazy' })),
      h('figcaption', { class: cls }, label));
  }

  function chipsFor(step) {
    if (!step.syllabus.length && !step.extra.length) return null;
    return h('div', { class: 'chips' },
      step.syllabus.map((x) => h('span', { class: 'chip', title: 'Syllabus concept' }, x)),
      step.extra.map((x) => h('span', { class: 'chip ext', title: 'Beyond the syllabus' }, x)));
  }

  function factsBox(facts) {
    return h('div', { class: 'diag' }, facts.map((f) => h('div', { class: `diag-item ${f.flag ? 'flag' : ''}` },
      h('span', { class: 'diag-label' }, f.label), h('b', {}, f.value), h('small', {}, f.note),
      f.flag ? h('span', { class: 'diag-tag' }, 'check') : null)));
  }

  function fillCard(c, step, r) {
    c.el.classList.remove('is-working');
    const applied = r.status === 'applied';
    c.pill.className = `pill ${applied ? 'applied' : 'skipped'}`;
    c.pill.textContent = applied ? 'Done' : 'Skipped';
    const nodes = [];
    if (!applied) {
      c.el.classList.add('compact');
      nodes.push(h('p', { class: 'reason' }, r.summary));
    } else {
      nodes.push(chipsFor(step));
      if (r.after_url) nodes.push(h('div', { class: 'ba' }, figure(r.before_url, 'Before'), figure(r.after_url, 'After', 'after')));
      if (r.facts && r.facts.length) nodes.push(factsBox(r.facts));
      nodes.push(h('div', { class: 'summary' }, r.summary));
    }
    if (r.details && r.details.length) {
      nodes.push(h('details', { class: 'more' }, h('summary', {}, 'Object by object'), h('ul', {}, r.details.map((d) => h('li', {}, d)))));
    }
    if (r.aux && r.aux.length) {
      nodes.push(h('details', { class: 'more' }, h('summary', {}, 'Behind the scenes'),
        h('div', { class: 'aux' }, r.aux.map((a) => h('figure', {}, h('img', { src: a.url, alt: a.label, loading: 'lazy' }), h('figcaption', {}, a.label))))));
    }
    c.body.replaceChildren(...nodes.filter(Boolean));
  }

  // ------------------------------------------------------------------ analysis + results
  async function finish(token) {
    let a, res;
    try {
      [a, res] = await Promise.all([api(`/api/s/${S.session}/analysis`), api(`/api/s/${S.session}/results`)]);
    } catch (e) {
      if (token !== S.runToken) return;
      if (e.status === 404) return sessionExpired();
      return runFailed(e.message);
    }
    if (token !== S.runToken) return;
    renderAnalysis(a);
    renderResults(res);
    unlock('analysis', 'results');
    $('#startBtn').disabled = false;
    show('pipeline', { scroll: false });
    $('#pipeEnd').hidden = false;
    $('#pipeEnd').scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  const STATUS_TEXT = { identified: 'Named', ambiguous: 'Ambiguous', unknown: 'Unknown' };
  const sizesText = (s) => Object.entries(s).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ') || '–';

  // battery counter: one row per catalogue battery, with the grand total
  function counterTable(counter, total) {
    const anyLeak = counter.some((l) => l.leaking);
    return h('table', { class: 'metrics invoice-table counter-table' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Battery'), h('th', { class: 'num' }, 'Count'),
        anyLeak ? h('th', { class: 'num' }, 'Leaking') : null,
        h('th', { class: 'num' }, 'Price each'), h('th', { class: 'num' }, 'Total'))),
      h('tbody', {}, counter.map((l) => h('tr', { class: l.quantity ? '' : 'zero' },
        h('td', {}, l.name), h('td', { class: 'num' }, h('b', {}, l.quantity)),
        anyLeak ? h('td', { class: 'num leaknum' }, l.leaking ? h('b', {}, `⚠ ${l.leaking}`) : '–') : null,
        h('td', { class: 'num' }, l.unit_price), h('td', { class: 'num' }, l.line_total)))),
      h('tfoot', {}, h('tr', {}, h('td', {}, 'Grand total'), h('td', { class: 'num' }, counter.reduce((s, l) => s + l.quantity, 0)),
        anyLeak ? h('td', { class: 'num' }, counter.reduce((s, l) => s + l.leaking, 0)) : null,
        h('td', {}, ''), h('td', { class: 'num' }, total))));
  }

  // ------------------------------------------------------------------ leakage warnings
  const COND_TEXT = { ok: 'OK', check: 'Check', damaged: 'Leaking' };

  function healthBanner(hs) {
    if (!hs) return null;
    const n = hs.counts;
    const title = hs.level === 'damaged'
      ? `Warning: ${n.damaged} ${n.damaged === 1 ? 'battery looks' : 'batteries look'} leaking`
      : hs.level === 'check'
        ? `${n.check} ${n.check === 1 ? 'battery has' : 'batteries have'} a small deposit: check by eye`
        : 'No leakage seen on any battery';
    const icon = hs.level === 'ok' ? '✓' : '!';
    return h('div', { class: `hb ${hs.level}`, role: hs.level === 'ok' ? null : 'alert' },
      h('span', { class: 'hb-icon', 'aria-hidden': 'true' }, icon),
      h('div', {}, h('b', {}, title), h('p', {}, hs.advice),
        hs.level !== 'ok' ? h('p', { class: 'small', style: 'margin-top:4px' }, 'Leaking batteries are circled in magenta on the photo, small deposits in amber.') : null));
  }

  function renderAnalysis(a) {
    const n = a.counts;
    S.counter = a.counter;
    S.grandTotal = a.grand_total;
    const cards = h('div', { class: 'an-cards' },
      h('div', { class: 'card an-card' }, h('b', {}, a.batteries), h('span', {}, 'batteries counted')),
      h('div', { class: 'card an-card' }, h('b', {}, `₹${a.grand_total}`), h('span', {}, 'grand total')),
      h('div', { class: `card an-card ${n.ambiguous ? 'flag' : ''}` }, h('b', {}, n.ambiguous), h('span', {}, 'ambiguous')),
      h('div', { class: `card an-card ${n.unknown ? 'bad' : ''}` }, h('b', {}, n.unknown), h('span', {}, 'unknown')),
      h('div', { class: `card an-card ${a.health && a.health.counts.damaged ? 'leak' : ''}` }, h('b', {}, a.health ? a.health.counts.damaged : 0), h('span', {}, 'leaking')),
      h('div', { class: 'card an-card' }, h('b', {}, a.mean_confidence != null ? pct(a.mean_confidence, 0) : '–'), h('span', {}, 'average confidence')));

    const counterCard = h('div', { class: 'card panel table-wrap' },
      h('h3', {}, 'Battery counter'),
      h('p', { class: 'sub' }, 'How many of each battery were found, times its price. Objects that need a check are not counted.'),
      counterTable(a.counter, a.grand_total));

    // what was found
    const scaleLine = a.scale
      ? `Credit card found: ${a.scale.size_px[0]} x ${a.scale.size_px[1]} px, so 1 px is about ${a.scale.mm_per_px} mm. Its four corners give a homography, so sizes are in true millimetres.`
      : 'No credit card found, so no sizes in millimetres and no battery could be named.';
    const cropLine = a.crop
      ? (a.crop.status === 'applied'
        ? `White paper found and cut to a straight ${a.crop.size[0]} x ${a.crop.size[1]} px rectangle${a.crop.angle ? `, after turning it ${a.crop.angle.toFixed(1)} deg` : ''}.`
        : `Not needed: ${a.crop.reason}.`)
      : '–';
    const found = h('div', { class: 'card panel' },
      h('h3', {}, 'What was found'),
      h('p', { class: 'sub' }, 'From the measurements on this photo.'),
      h('ul', { class: 'flag-list' },
        h('li', {}, h('b', {}, 'Auto-crop'), h('small', {}, cropLine)),
        h('li', {}, h('b', {}, 'Scale'), h('small', {}, scaleLine)),
        h('li', {}, h('b', {}, 'Sharpness'), h('small', {}, `${a.blur_var} (Laplacian variance; below 15 counts as blurred)`))));

    // needs a check by hand
    const flagged = a.parts.filter((p) => p.status !== 'identified');
    const check = h('div', { class: 'card panel' },
      h('h3', {}, 'Needs a check by hand'),
      h('p', { class: 'sub' }, 'These objects are not counted in the total. The number matches the numbered photo.'),
      flagged.length
        ? h('ul', { class: 'flag-list' }, flagged.map((p) => h('li', {},
          h('b', {}, `Object ${p.no} `), h('span', { class: `st ${p.status}` }, STATUS_TEXT[p.status]),
          h('small', {}, p.status === 'ambiguous' ? `Could be ${p.between[0]} or ${p.between[1]}: ${p.reason}.` : `${cap(p.reason)}.`),
          p.candidates.length ? h('small', {}, `Nearest: ${p.candidates.map(([d, nm]) => `${nm} (${d.toFixed(2)})`).join(', ')}`) : null)))
        : h('p', {}, '✓ Every object was named. Nothing needs a check.'),
      h('p', { class: 'sub', style: 'margin-top:12px' }, 'A cell standing on its end, or a battery of a size not in the catalogue, shows up here.'));

    const annot = h('div', { class: 'card panel annot' },
      h('h3', {}, 'Numbered photo'),
      h('img', { src: a.annotated_url, alt: 'Photo with every object outlined and numbered' }),
      h('div', { class: 'key' },
        h('span', {}, h('i', { style: 'background:#1d5fe0' }), 'named'),
        h('span', {}, h('i', { style: 'background:#d97706' }), 'ambiguous'),
        h('span', {}, h('i', { style: 'background:#c82828' }), 'unknown'),
        h('span', {}, h('i', { style: 'background:#787878' }), 'credit card')));

    const table = h('div', { class: 'card panel table-wrap' },
      h('h3', {}, 'Every object'),
      h('p', { class: 'sub' }, 'Sizes in millimetres: a cylinder (AA, AAA) has a length and a diameter, a round cell (CR2016, LR44) a diameter. Confidence is high when the size fits one battery closely and no other battery nearly as well.'),
      h('table', { class: 'metrics parts-table' },
        h('thead', {}, h('tr', {}, ['No', 'Shape', 'Sizes (mm)', 'Result', 'Confidence', 'Condition'].map((t) => h('th', {}, t)))),
        h('tbody', {}, a.parts.map((p) => h('tr', {},
          h('td', { class: 'num' }, p.no),
          h('td', {}, p.type),
          h('td', { class: 'num' }, sizesText(p.sizes)),
          h('td', {}, h('span', { class: `st ${p.status}` }, p.status === 'identified' ? (p.count > 1 ? `${p.count} x ${p.name}` : p.name) : STATUS_TEXT[p.status]),
            p.group ? h('small', { class: 'muted', style: 'display:block' }, `${p.group}, counted by size`) : null),
          h('td', { class: 'num' }, p.confidence != null ? pct(p.confidence, 0) : '–'),
          h('td', {}, h('span', { class: `cond ${p.health}` }, COND_TEXT[p.health] || 'OK'),
            p.health !== 'ok' ? h('small', { class: 'muted', style: 'display:block' }, p.health_reason) : null))))));

    const stepList = h('ul', { class: 'step-list' }, a.steps.map((s) => {
      const on = s.status === 'applied';
      return h('li', {}, h('span', { class: `dot ${on ? 'done' : 'skip'}` }, on ? '✓' : '–'),
        h('div', {}, h('b', {}, s.title), h('small', {}, s.summary)));
    }));
    const concepts = h('div', { class: 'concepts' },
      h('h4', {}, `Syllabus concepts used (${a.concepts.syllabus.length})`),
      h('div', { class: 'chips' }, a.concepts.syllabus.map((c) => h('span', { class: 'chip' }, c))),
      h('h4', {}, `Beyond the syllabus (${a.concepts.extra.length})`),
      h('div', { class: 'chips' }, a.concepts.extra.map((c) => h('span', { class: 'chip ext' }, c))),
      h('h4', {}, 'Not used'),
      h('p', { class: 'muted small' }, a.not_used.join(', ') + '.'));
    const did = h('div', { class: 'card panel an-wide' },
      h('h3', {}, 'What was done'),
      h('p', { class: 'sub' }, 'Each step decided on its own whether it was needed.'),
      stepList, concepts);

    $('#analysisBody').replaceChildren(healthBanner(a.health) || '', cards, counterCard,
      h('div', { class: 'an-grid' }, annot, h('div', { style: 'display:grid;gap:22px' }, found, check)),
      table, did);
    counterCard.style.margin = '22px 0';
    table.style.marginTop = '22px';
  }

  function renderResults(r) {
    const inv = r.invoice;
    const base = `/api/s/${S.session}/download`;
    const lines = S.counter
      ? counterTable(S.counter, S.grandTotal)
      : h('p', { class: 'muted' }, 'Nothing could be named, so there is nothing to count.');
    const notCharged = inv.not_charged.length
      ? h('div', { class: 'not-charged' }, h('b', {}, `${inv.not_charged.length} not counted, check by hand:`),
        h('ul', {}, inv.not_charged.map((x) => {
          const p = r.parts.find((q) => q.status !== 'identified' && x.label != null && q.reason === x.reason && q.status === x.status);
          return h('li', {}, `${p ? `Object ${p.no}` : 'Item'}: ${x.status}, ${x.reason}`);
        })))
      : null;
    $('#resultsBody').replaceChildren(healthBanner(r.health) || '', h('div', { class: 'res-grid' },
      h('div', { class: 'card panel annot' }, h('h3', {}, 'Numbered photo'), h('img', { src: r.annotated_url, alt: 'Numbered photo' })),
      h('div', { class: 'card panel' }, h('h3', {}, 'Battery count and total'), h('p', { class: 'sub' }, 'Count times catalogue price. Flagged objects are not counted.'), lines, notCharged)),
    h('div', { class: 'dl-row', style: 'margin-top:26px' },
      h('a', { class: 'btn btn-primary btn-lg', href: `${base}/pdf`, download: '' }, '↓ Download PDF'),
      h('a', { class: 'btn btn-primary btn-lg', href: `${base}/csv`, download: '' }, '↓ Download CSV'),
      h('a', { class: 'btn btn-primary btn-lg', href: `${base}/jpg`, download: '' }, '↓ Download JPG')),
    h('div', { class: 'end-actions' }, h('button', { class: 'btn btn-ghost', type: 'button', onclick: () => { location.href = '/#upload'; location.reload(); } }, 'Check another photo')));
  }

  // ------------------------------------------------------------------ leakage check page
  function leakError(msg) { $('#leakError').textContent = msg; $('#leakError').hidden = !msg; }

  async function runLeak(file) {
    const err = validate(file);
    leakError(err);
    if (err) return;
    const fd = new FormData();
    fd.append('file', file);
    $('#leakBody').replaceChildren();
    $('#leakBusy').hidden = false;
    $('#leakBusy').scrollIntoView({ behavior: 'smooth', block: 'center' });
    try {
      renderLeak(await api('/api/leakcheck', { method: 'POST', body: fd }));
    } catch (e) {
      leakError(e.message);
    } finally {
      $('#leakBusy').hidden = true;
    }
  }

  function renderLeak(d) {
    if (!d.found) {
      $('#leakBody').replaceChildren(h('p', { class: 'notice warn' }, d.message));
      return;
    }
    const unit = d.has_scale ? 'Deposit (mm²)' : 'Deposit (% of battery)';
    const table = h('div', { class: 'card panel table-wrap' },
      h('h3', {}, 'Battery by battery'),
      h('p', { class: 'sub' }, d.has_scale
        ? 'The credit card gave the scale, so each deposit is measured in mm². 1.5 mm² or more asks for a check, 4 mm² or more counts as leaking.'
        : 'No credit card in the photo, so each deposit is judged as a share of the battery\'s own area (0.6% asks for a check, 2% counts as leaking).'),
      h('table', { class: 'metrics leak-table' },
        h('thead', {}, h('tr', {}, ['No', 'Battery', 'Condition', unit, 'What was seen'].map((t) => h('th', {}, t)))),
        h('tbody', {}, d.rows.map((r) => h('tr', {},
          h('td', { class: 'num' }, r.no),
          h('td', {}, r.name || r.shape),
          h('td', {}, h('span', { class: `cond ${r.status}` }, COND_TEXT[r.status])),
          h('td', { class: 'num' }, d.has_scale ? (r.area_mm2 != null ? r.area_mm2.toFixed(1) : '–') : `${(100 * r.frac).toFixed(1)}%`),
          h('td', { class: 'why' }, r.status === 'ok' ? 'no crust or corrosion' : r.reason))))));
    const imgs = h('div', { class: 'card panel' },
      h('h3', {}, 'What was found'),
      h('div', { class: 'leak-imgs' },
        h('figure', {}, h('img', { src: d.annotated_url, alt: 'Numbered photo with leaking batteries circled' }), h('figcaption', {}, 'Numbered photo: magenta circle = leaking, amber = check')),
        h('figure', {}, h('img', { src: d.leak_url, alt: 'Leak pixels highlighted' }), h('figcaption', {}, 'Deposit pixels in magenta'))));
    const aux = d.aux.length
      ? h('details', { class: 'more card panel', style: 'margin-top:22px' }, h('summary', {}, 'Behind the scenes: zones, texture and glare'),
        h('div', { class: 'aux' }, d.aux.map((a) => h('figure', {}, h('img', { src: a.url, alt: a.label, loading: 'lazy' }), h('figcaption', {}, a.label)))))
      : null;
    const facts = d.facts && d.facts.length ? factsBox(d.facts) : null;
    $('#leakBody').replaceChildren(healthBanner({ counts: d.counts, level: d.level, advice: d.advice }),
      h('div', { class: 'leak-hero' }, imgs, h('div', {}, table)),
      facts ? h('div', { class: 'card panel', style: 'margin-top:22px' }, h('h3', {}, 'Numbers'), facts, h('p', { class: 'sub', style: 'margin-top:10px' }, d.summary)) : null,
      aux);
    $('#leakBody').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function loadLeakAccuracy() {
    let r;
    try { r = await api('/api/leak_report'); } catch { return; }
    const p = (x) => `${Math.round(100 * x)}%`;
    const bars = r.by_area.map((b) => h('div', { class: 'bar-row' },
      h('span', {}, `${b.area_mm2} mm²`), h('span', { class: 'track' }, h('i', { style: `width:${100 * b.flagged}%` })), h('span', { class: 'v' }, p(b.flagged))));
    $('#leakAcc').replaceChildren(h('div', { class: 'card panel', style: 'margin-top:26px' },
      h('h3', {}, 'How accurate is it?'),
      h('p', { class: 'sub' }, `Measured by tools/test_leakage.py on ${r.photos} drawn photos with leaks of known size and ${r.clean_batteries} clean batteries. These photos were never used to choose the thresholds (that was done on separate tuning photos).`),
      h('div', { class: 'acc-grid' },
        h('div', {}, h('b', {}, p(r.detection_rate)), h('span', {}, `of ${r.leaked_batteries} leaked batteries flagged`)),
        h('div', {}, h('b', {}, p(r.false_alarm_rate)), h('span', {}, 'false alarms on clean batteries')),
        h('div', {}, h('b', {}, p(r.precision)), h('span', {}, 'of flags were real leaks (precision)')),
        h('div', {}, h('b', {}, p(r.by_kind.green)), h('span', {}, 'green corrosion found')),
        h('div', {}, h('b', {}, p(r.by_kind.white)), h('span', {}, 'white crust found'))),
      h('h4', {}, 'Found, by size of the deposit'), ...bars,
      h('p', { class: 'muted small', style: 'margin-top:10px' }, 'Very small deposits (about 1-2 mm², a few grains) are hard to see at phone-photo resolution. Larger ones are found far more often. Not yet measured on real leaked batteries.')));
  }

  function initLeak() {
    const d = $('#leakDrop');
    d.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#leakFile').click(); } });
    ['dragenter', 'dragover'].forEach((ev) => d.addEventListener(ev, (e) => { e.preventDefault(); d.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((ev) => d.addEventListener(ev, (e) => { e.preventDefault(); d.classList.remove('over'); }));
    d.addEventListener('drop', (e) => runLeak(e.dataTransfer.files[0]));
    $('#leakFile').addEventListener('change', (e) => { runLeak(e.target.files[0]); e.target.value = ''; });
    $('#leakPhone').addEventListener('change', (e) => { runLeak(e.target.files[0]); e.target.value = ''; });
    $('#leakSample').addEventListener('click', async () => {
      try {
        const blob = await (await fetch('/samples/battery_4_leaking.jpg')).blob();
        runLeak(new File([blob], 'battery_4_leaking.jpg', { type: 'image/jpeg' }));
      } catch { leakError('Could not load the test photo.'); }
    });
    loadLeakAccuracy();
  }

  // ------------------------------------------------------------------ init
  initNav();
  initLeak();
  initCatalogue();
  initUpload();
  initHome();
  $('#toAnalysis').addEventListener('click', () => show('analysis'));
  $('#toResults').addEventListener('click', () => show('results'));
  const start = location.hash.slice(1);
  show(['home', 'catalogue', 'upload', 'leak'].includes(start) ? start : 'home', { scroll: false });
})();
