const $ = (sel) => document.querySelector(sel);
const pct = (x) => `${(x * 100).toFixed(1)}%`;

function nearest(grid, value) {
  return grid.reduce((best, row) =>
    Math.abs(row.threshold - value) < Math.abs(best.threshold - value) ? row : best
  );
}

function renderOpen(data) {
  const d = data.dataset;
  $("#lede").innerHTML =
    `<strong>${d.strokes}</strong> strokes in <strong>${d.rows.toLocaleString()}</strong> rows. ` +
    `That is ${pct(d.prevalence)}. Age does almost all of the work.`;
  const max = Math.max(...data.age_bands.map((b) => b.rate));
  $("#age").innerHTML = data.age_bands
    .map((b) => {
      const hot = b.rate >= 0.15;
      return `<div class="band ${hot ? "hot" : ""}">
        <span>${b.slice}</span>
        <div class="track"><i style="width:${(b.rate / max) * 100}%"></i></div>
        <span>${pct(b.rate)}</span>
      </div>`;
    })
    .join("");
}

function renderModels(data) {
  const show = data.models.filter((m) => m.name !== "Logistic, class-weighted");
  const weighted = data.models.find((m) => m.name === "Logistic, class-weighted");
  $("#modelsTable").innerHTML = `
    <thead><tr><th>Model</th><th>ROC</th><th>PR-AUC</th><th>Brier</th><th>Sens @ 0.5</th></tr></thead>
    <tbody>
      ${show
        .map(
          (m) => `<tr class="${m.name === "Logistic" ? "desk" : ""}">
            <td>${m.name}</td>
            <td>${m.roc_auc.toFixed(3)}</td>
            <td>${m.pr_auc.toFixed(3)}</td>
            <td>${m.brier.toFixed(3)}</td>
            <td>${pct(m.at_half.sensitivity)}</td>
          </tr>`
        )
        .join("")}
    </tbody>`;
  const note = document.createElement("p");
  note.className = "prose";
  note.textContent = weighted
    ? `Class-weighted logistic ROC ${weighted.roc_auc.toFixed(3)}, but Brier ${weighted.brier.toFixed(3)} — those scores are not risks. Unweighted logistic Brier is ${data.brier_desk.toFixed(3)}, next to the majority Brier of about 0.047.`
    : "";
  $("#modelsTable").after(note);

  const colors = {
    "Age only": "#6d6458",
    Logistic: "#b4332a",
    "Random forest": "#1c1915",
    "Decision tree": "#8d8478",
  };
  const pad = 36;
  const w = 420;
  const h = 320;
  const X = (v) => pad + v * (w - pad - 12);
  const Y = (v) => h - pad - v * (h - pad - 12);
  let svg = `<line x1="${pad}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#c4b8a8" stroke-dasharray="4 4"/>`;
  svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${Y(0)}" stroke="#1c1915"/>`;
  svg += `<line x1="${pad}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(0)}" stroke="#1c1915"/>`;
  Object.entries(colors).forEach(([name, color]) => {
    const pts = (data.roc[name] || [])
      .map((p) => `${X(p.fpr).toFixed(1)},${Y(p.tpr).toFixed(1)}`)
      .join(" ");
    svg += `<polyline fill="none" stroke="${color}" stroke-width="${name === "Logistic" ? 2.6 : 1.6}" points="${pts}"/>`;
  });
  svg += `<text x="${pad}" y="16" font-family="IBM Plex Mono" font-size="11" fill="#6d6458">TPR</text>`;
  svg += `<text x="${w - 70}" y="${h - 8}" font-family="IBM Plex Mono" font-size="11" fill="#6d6458">FPR</text>`;
  let y = 28;
  Object.entries(colors).forEach(([name, color]) => {
    svg += `<text x="250" y="${y}" font-family="IBM Plex Mono" font-size="11" fill="${color}">${name}</text>`;
    y += 16;
  });
  $("#roc").innerHTML = svg;
}

function renderPoint(data, row) {
  const g = data.thresholds;
  const input = $("#cut");
  input.max = String(g.length - 1);
  const idx = g.findIndex((r) => r.threshold === row.threshold);
  input.value = String(idx < 0 ? 0 : idx);
  $("#cutLabel").textContent = row.threshold.toFixed(2);
  const per = {
    flagged: (1000 * row.flagged) / data.dataset.test_rows,
    caught: (1000 * row.tp) / data.dataset.test_rows,
    missed: (1000 * row.fn) / data.dataset.test_rows,
    false_flags: (1000 * row.fp) / data.dataset.test_rows,
  };
  $("#flags").innerHTML = [
    [`${pct(row.sensitivity)}`, "Sensitivity"],
    [`${pct(row.ppv)}`, "Precision"],
    [`${row.false_flags_per_true ?? "—"}`, "False flags / true"],
    [`${per.flagged.toFixed(0)}`, "Flagged / 1,000"],
  ]
    .map(([b, s]) => `<div class="flag"><b>${b}</b><span>${s}</span></div>`)
    .join("");
}

function renderBurden(data) {
  const pad = 36;
  const w = 420;
  const h = 320;
  const X = (v) => pad + v * (w - pad - 12);
  const Y = (v) => h - pad - v * (h - pad - 12);
  let svg = `<line x1="${pad}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#c4b8a8"/>`;
  svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${Y(0)}" stroke="#1c1915"/>`;
  svg += `<line x1="${pad}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(0)}" stroke="#1c1915"/>`;
  data.calibration.forEach((b) => {
    const r = Math.max(3, Math.sqrt(b.n) * 0.7);
    svg += `<circle cx="${X(b.mean_p)}" cy="${Y(b.observed)}" r="${r}" fill="none" stroke="#b4332a" stroke-width="1.5"/>`;
  });
  svg += `<text x="${pad}" y="16" font-family="IBM Plex Mono" font-size="11" fill="#6d6458">Observed</text>`;
  svg += `<text x="${w - 120}" y="${h - 8}" font-family="IBM Plex Mono" font-size="11" fill="#6d6458">Mean predicted</text>`;
  $("#cal").innerHTML = svg;
  $("#odds").innerHTML = `
    <thead><tr><th>Association, training fold</th><th>OR</th><th>95% CI</th></tr></thead>
    <tbody>
      ${data.odds_ratios
        .map(
          (r) => `<tr>
            <td>${r.term}</td>
            <td>${r.odds_ratio.toFixed(2)}</td>
            <td>${r.ci_low.toFixed(2)}–${r.ci_high.toFixed(2)}</td>
          </tr>`
        )
        .join("")}
    </tbody>`;
  const op = data.operating;
  $("#close").textContent =
    `At a cut of ${op.threshold.toFixed(2)}, about ${op.per_1000.flagged} people per 1,000 are flagged, ` +
    `${op.per_1000.caught} of the strokes in this file are caught, and ${op.per_1000.missed} are missed. ` +
    `Age, per 10 years, roughly doubles the odds. Glucose and BMI do not earn a louder sentence than that. ` +
    `${data.note}`;
}

async function boot() {
  const data = await fetch("casebook/metrics.json", { cache: "no-store" }).then((r) => r.json());
  const params = new URLSearchParams(location.search);
  if (params.get("shot")) document.body.dataset.shot = params.get("shot");
  renderOpen(data);
  renderModels(data);
  const requested = Number(params.get("t"));
  let row = data.operating;
  if (!Number.isNaN(requested)) row = nearest(data.thresholds, requested);
  renderPoint(data, row);
  renderBurden(data);
  $("#cut").addEventListener("input", (event) => {
    renderPoint(data, data.thresholds[Number(event.target.value)]);
  });
}

boot();
