const dataUrl = "./data/leaderboard.json";

const fmtPct = (x) => `${(x * 100).toFixed(2)}%`;

function rankModels(models) {
  return [...models].sort((a, b) => {
    if (b.accuracy !== a.accuracy) return b.accuracy - a.accuracy;
    if (b.sampleCount !== a.sampleCount) return b.sampleCount - a.sampleCount;
    return a.name.localeCompare(b.name);
  });
}

function renderMeta(meta) {
  document.getElementById("updated-at").textContent = meta.updatedAt;
  document.getElementById("protocol-name").textContent = `${meta.dataset} · ${meta.protocol}`;
  const pills = [
    `${meta.dataset} benchmark`,
    `${meta.totalSamples} total samples`,
    meta.promptMode,
    meta.toolMode,
  ];
  document.getElementById("meta-pills").innerHTML = pills
    .map((label) => `<span class="meta-pill">${label}</span>`)
    .join("");
}

function renderPodium(models) {
  const podium = document.getElementById("podium");
  podium.innerHTML = rankModels(models)
    .slice(0, 3)
    .map(
      (model, idx) => `
        <article class="podium-card">
          <div class="podium-rank">#${idx + 1}</div>
          <div class="model-name">${model.name}</div>
          <div class="provider">${model.provider}</div>
          <div class="score">${(model.accuracy * 100).toFixed(2)}<small> acc</small></div>
          <div class="model-meta">
            <span class="status-pill ${model.status}">${model.statusLabel}</span>
            <span class="meta-pill">${model.sampleCount}/${model.totalSamples} samples</span>
            <span class="meta-pill">${fmtPct(model.toolRate)} tool rate</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderTable(models) {
  const tbody = document.getElementById("leaderboard-body");
  tbody.innerHTML = rankModels(models)
    .map(
      (model, idx) => `
        <tr>
          <td class="rank-cell">#${idx + 1}</td>
          <td>
            <div>${model.name}</div>
            <div class="provider">${model.provider}</div>
          </td>
          <td class="score-cell">${fmtPct(model.accuracy)}</td>
          <td>${model.sampleCount}/${model.totalSamples}</td>
          <td>${fmtPct(model.toolRate)}</td>
          <td>${model.directAccuracy == null ? "—" : fmtPct(model.directAccuracy)}</td>
          <td>${model.z3Accuracy == null ? "—" : fmtPct(model.z3Accuracy)}</td>
          <td><span class="status-pill ${model.status}">${model.statusLabel}</span></td>
        </tr>
      `
    )
    .join("");
}

function renderBenchmarks(benchmarks) {
  const grid = document.getElementById("benchmark-grid");
  grid.innerHTML = benchmarks
    .map((bench) => {
      const progress = (bench.currentModels / bench.targetModels) * 100;
      return `
        <div class="progress-item">
          <div class="progress-top">
            <strong>${bench.name}</strong>
            <span>${bench.currentModels}/${bench.targetModels} models</span>
          </div>
          <div class="progress-bar"><div class="progress-fill" style="width:${progress.toFixed(2)}%"></div></div>
          <div class="progress-caption">
            ${bench.note}
          </div>
        </div>
      `;
    })
    .join("");
}

function renderProtocol(meta) {
  const list = document.getElementById("protocol-list");
  list.innerHTML = meta.rules.map((rule) => `<li>${rule}</li>`).join("");
}

async function boot() {
  const res = await fetch(dataUrl);
  const payload = await res.json();
  renderMeta(payload.meta);
  renderPodium(payload.boards.current);
  renderTable(payload.boards.current);
  renderBenchmarks(payload.benchmarks);
  renderProtocol(payload.meta);
}

boot();
