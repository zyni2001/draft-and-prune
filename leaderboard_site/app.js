const dataUrl = "./data/leaderboard.json";

const fmtPct = (x) => `${(x * 100).toFixed(2)}%`;

function rankModels(models) {
  const isFinal = (m) => m.status === "final";
  const cmp = (a, b) => {
    if (b.accuracy !== a.accuracy) return b.accuracy - a.accuracy;
    if (b.sampleCount !== a.sampleCount) return b.sampleCount - a.sampleCount;
    return a.name.localeCompare(b.name);
  };
  const finals = models.filter(isFinal).sort(cmp);
  const inFlight = models.filter((m) => !isFinal(m)).sort(cmp);
  return [...finals, ...inFlight];
}

function renderMeta(meta) {
  document.getElementById("updated-at").textContent = meta.updatedAt;
  const pills = meta.pills ?? [
    meta.promptMode,
    meta.toolMode,
  ];
  document.getElementById("meta-pills").innerHTML = pills
    .map((label) => `<span class="meta-pill">${label}</span>`)
    .join("");
}

function renderTrackMeta(meta, track) {
  document.getElementById("protocol-name").textContent = `${track.name} · ${meta.protocol}`;
  document.getElementById("track-stats").textContent =
    track.currentTrackStats ?? `${track.totalSamples} samples`;
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
          <td><span class="status-pill ${model.status}">${model.statusLabel}</span></td>
        </tr>
      `
    )
    .join("");
}

function renderBenchmarkSwitcher(tracks, activeId, onSelect) {
  const container = document.getElementById("benchmark-switcher");
  container.innerHTML = tracks
    .map(
      (track) => `
        <button class="track-pill ${track.id === activeId ? "active" : ""}" data-track-id="${track.id}">
          ${track.name}
        </button>
      `
    )
    .join("");
  container.querySelectorAll(".track-pill").forEach((btn) => {
    btn.addEventListener("click", () => onSelect(btn.dataset.trackId));
  });
}

async function boot() {
  // Avoid stale leaderboard.json after deploys (GitHub Pages / browser HTTP cache).
  const res = await fetch(dataUrl, { cache: "no-store" });
  const payload = await res.json();
  renderMeta(payload.meta);
  const tracks = payload.tracks;
  const trackMap = Object.fromEntries(tracks.map((track) => [track.id, track]));
  const renderTrack = (trackId) => {
    const track = trackMap[trackId];
    renderTrackMeta(payload.meta, track);
    renderBenchmarkSwitcher(tracks, trackId, renderTrack);
    renderTable(payload.boards[trackId] ?? []);
  };
  renderTrack(payload.meta.defaultTrack);
}

boot();
