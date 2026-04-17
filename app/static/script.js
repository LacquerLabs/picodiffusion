// ── helpers ────────────────────────────────────────────────────────────────
function trimExt(filename) {
  return filename.replace(/\.[^.]+$/, '');
}

// ── footer year ───────────────────────────────────────────────────────────
document.getElementById('footer-year').textContent = new Date().getFullYear();

// ── help panel ────────────────────────────────────────────────────────────
const helpPanel = document.getElementById('help-panel');
const helpTitle = document.getElementById('help-panel-title');
const helpBody = document.getElementById('help-panel-body');
let helpContent = {};
let activeHelp = null;

fetch('/static/help.json')
  .then(r => r.json())
  .then(data => { helpContent = data; })
  .catch(() => { console.warn('could not load help content'); });

document.querySelectorAll('.help-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const key = btn.dataset.help;
    if (activeHelp === key) {
      helpPanel.classList.remove('open');
      activeHelp = null;
      return;
    }
    const content = helpContent[key];
    if (!content) return;
    helpTitle.textContent = content.title;
    helpBody.innerHTML = content.body;
    helpPanel.classList.add('open');
    activeHelp = key;
  });
});

document.getElementById('help-panel-close').addEventListener('click', () => {
  helpPanel.classList.remove('open');
  activeHelp = null;
});

// ── model list ──────────────────────────────────────────────────────────────
const checkpointSel = document.getElementById('checkpoint');
const vaeSel = document.getElementById('vae');
const loraSel = document.getElementById('lora');
const btn = document.getElementById('generate-btn');
const dot = document.getElementById('status-dot');
const stxt = document.getElementById('status-text');

function loadModelList() {
  fetch('/models')
    .then(r => r.json())
    .then(d => {
      checkpointSel.innerHTML = '';
      if (d.checkpoints.length === 0) {
        checkpointSel.innerHTML = '<option value="">— no checkpoints found —</option>';
      } else {
        d.checkpoints.forEach(name => {
          const o = document.createElement('option');
          o.value = name;
          o.textContent = trimExt(name);
          checkpointSel.appendChild(o);
        });
        btn.disabled = false;
      }

      vaeSel.innerHTML = '<option value="">none (use checkpoint embedded)</option>';
      d.vaes.forEach(name => {
        const o = document.createElement('option');
        o.value = name;
        o.textContent = trimExt(name);
        vaeSel.appendChild(o);
      });

      loraSel.innerHTML = '<option value="">none</option>';
      d.loras.forEach(name => {
        const o = document.createElement('option');
        o.value = name;
        o.textContent = trimExt(name);
        loraSel.appendChild(o);
      });
    })
    .catch(() => {
      checkpointSel.innerHTML = '<option value="">— could not load list —</option>';
    });
}

loadModelList();

// ── status polling ──────────────────────────────────────────────────────────
function pollHealth() {
  fetch('/health')
    .then(r => r.json())
    .then(d => {
      if (d.status === 'ready') {
        dot.className = 'ready';
        const vaeLabel = d.vae ? trimExt(d.vae) : 'embedded vae';
        stxt.textContent = `${trimExt(d.checkpoint)} · ${vaeLabel} · ${d.device}`;
      } else if (d.status === 'loading') {
        dot.className = 'loading';
        stxt.textContent = 'loading model…';
      } else {
        dot.className = 'idle';
        stxt.textContent = 'select a model and generate';
      }
    })
    .catch(() => {
      dot.className = 'error';
      stxt.textContent = 'service unreachable';
    })
    .finally(() => {
      setTimeout(pollHealth, 3000);
    });
}

pollHealth();

// ── sliders ─────────────────────────────────────────────────────────────────
const stepsInput = document.getElementById('steps');
const cfgInput = document.getElementById('cfg');
const loraWeightInput = document.getElementById('lora-weight');
const widthInput = document.getElementById('width');
const heightInput = document.getElementById('height');
const loraWeightGroup = document.getElementById('lora-weight-group');

loraSel.addEventListener('change', () => {
  loraWeightGroup.style.display = loraSel.value ? '' : 'none';
});

stepsInput.addEventListener('input', () => {
  document.getElementById('steps-val').textContent = stepsInput.value;
});
cfgInput.addEventListener('input', () => {
  document.getElementById('cfg-val').textContent = cfgInput.value;
});
loraWeightInput.addEventListener('input', () => {
  document.getElementById('lora-weight-val').textContent = loraWeightInput.value;
});
widthInput.addEventListener('input', () => {
  document.getElementById('width-val').textContent = widthInput.value;
});
heightInput.addEventListener('input', () => {
  document.getElementById('height-val').textContent = heightInput.value;
});

// ── dice button ─────────────────────────────────────────────────────────────
document.getElementById('dice-btn').addEventListener('click', () => {
  document.getElementById('seed').value = Math.floor(Math.random() * 4294967295);
});

// ── generate ────────────────────────────────────────────────────────────────
let timerInterval = null;
let progressInterval = null;

function setGenerating(on) {
  btn.disabled = on;
  document.getElementById('placeholder').style.display = on ? 'none' : '';
  document.getElementById('progress-overlay').style.display = on ? 'flex' : 'none';
  document.getElementById('output-image').style.display = 'none';
  document.getElementById('meta').style.display = 'none';
  document.getElementById('download-link').style.display = 'none';
  document.getElementById('error-msg').style.display = 'none';

  if (on) {
    let secs = 0;
    document.getElementById('timer').textContent = '0s';
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = 'loading model…';
    timerInterval = setInterval(() => {
      secs++;
      document.getElementById('timer').textContent = secs + 's';
    }, 1000);
    progressInterval = setInterval(() => {
      fetch('/progress').then(r => r.json()).then(d => {
        if (d.total > 0) {
          const pct = Math.round((d.step / d.total) * 100);
          document.getElementById('progress-bar').style.width = pct + '%';
          document.getElementById('progress-text').textContent =
            'step ' + d.step + ' / ' + d.total;
        }
      }).catch(() => { });
    }, 500);
  } else {
    clearInterval(timerInterval);
    clearInterval(progressInterval);
    timerInterval = null;
    progressInterval = null;
    btn.disabled = false;
  }
}

document.getElementById('generate-btn').addEventListener('click', async () => {
  const checkpoint = checkpointSel.value;
  if (!checkpoint) return;

  const seedInput = document.getElementById('seed');
  const seedLocked = document.getElementById('seed-lock').checked;

  const body = {
    checkpoint: checkpoint,
    vae: vaeSel.value || null,
    lora: loraSel.value || null,
    lora_weight: parseFloat(loraWeightInput.value),
    sampler: document.getElementById('sampler').value,
    prompt: document.getElementById('prompt').value,
    negative_prompt: document.getElementById('negative_prompt').value,
    steps: parseInt(stepsInput.value, 10),
    cfg: parseFloat(cfgInput.value),
    width: parseInt(widthInput.value, 10),
    height: parseInt(heightInput.value, 10),
  };

  // If seed is locked, send whatever is in the field.
  // Otherwise let the server pick a random seed.
  if (seedLocked) {
    const seedRaw = seedInput.value.trim();
    if (seedRaw !== '') body.seed = parseInt(seedRaw, 10);
  }

  setGenerating(true);

  let data;
  try {
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    data = await resp.json();
  } catch (err) {
    setGenerating(false);
    const errEl = document.getElementById('error-msg');
    errEl.style.display = 'block';
    errEl.textContent = 'Network error: ' + err.message;
    document.getElementById('placeholder').style.display = 'none';
    return;
  }

  setGenerating(false);

  if (data.error) {
    const errEl = document.getElementById('error-msg');
    errEl.style.display = 'block';
    errEl.textContent = data.error;
    document.getElementById('placeholder').style.display = 'none';
    return;
  }

  // show image
  const img = document.getElementById('output-image');
  img.src = 'data:image/png;base64,' + data.image;
  img.style.display = 'block';

  // fill seed back in so it is easy to reproduce
  document.getElementById('seed').value = data.seed;

  // show meta
  const vaeLabel = data.vae ? trimExt(data.vae) : 'embedded vae';
  const loraLabel = data.lora ? `${trimExt(data.lora)} @ ${data.lora_weight}` : 'no lora';
  const meta = document.getElementById('meta');
  meta.style.display = 'block';
  meta.innerHTML =
    `<strong>${trimExt(data.checkpoint)}</strong> &nbsp;·&nbsp; ${vaeLabel} &nbsp;·&nbsp; ${loraLabel}<br>` +
    `${data.sampler} &nbsp;·&nbsp; ` +
    `seed <strong>${data.seed}</strong> &nbsp;·&nbsp; ` +
    `${data.steps} steps &nbsp;·&nbsp; ` +
    `cfg ${data.cfg} &nbsp;·&nbsp; ` +
    `<strong>${data.elapsed_seconds}s</strong>`;

  // show download link
  const dl = document.getElementById('download-link');
  dl.href = 'data:image/png;base64,' + data.image;
  dl.download = `picod_${data.seed}_${String(Math.floor(Math.random() * 1000)).padStart(3, '0')}.png`;
  dl.style.display = '';

  document.getElementById('placeholder').style.display = 'none';
});
