(function () {
  var HERO = ["Majority rate", "Decision tree", "Random forest", "Age only", "Logistic"];
  var STYLE = {
    "Majority rate": { color: "#8d8376", width: 1.6, dash: "4 3" },
    "Age only": { color: "#2457a6", width: 1.8, dash: "" },
    "Logistic": { color: "#c7372f", width: 2.6, dash: "" },
    "Decision tree": { color: "#2f6b45", width: 1.6, dash: "" },
    "Random forest": { color: "#241c16", width: 1.6, dash: "1.2 2.4" }
  };

  var DATA = null;

  function svgEl(name, attrs) {
    var el = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attrs).forEach(function (key) {
      el.setAttribute(key, attrs[key]);
    });
    return el;
  }

  function pct(value, digits) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    return (Number(value) * 100).toFixed(digits == null ? 1 : digits) + "%";
  }

  function fixed(value, digits) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    return Number(value).toFixed(digits);
  }

  function orNum(value) {
    var text = Number(value).toFixed(3);
    return text.endsWith("0") ? Number(value).toFixed(2) : text;
  }

  function formatP(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    if (Number(value) < 0.001) return "<0.001";
    return Number(value).toFixed(3);
  }

  function ciPct(interval) {
    if (!interval) return "";
    return "95% CI " + Math.round(interval[0] * 100) + "–" + Math.round(interval[1] * 100) + "%";
  }

  function range3(interval) {
    return fixed(interval[0], 3) + " to " + fixed(interval[1], 3);
  }

  function signed3(value) {
    return (value > 0 ? "+" : "") + fixed(value, 3);
  }

  function oneDecimal(value) {
    return (Math.round(Number(value) * 10) / 10).toFixed(1);
  }

  function byName(models) {
    var map = {};
    models.forEach(function (model) { map[model.name] = model; });
    return map;
  }

  function rowForThreshold(data, t) {
    var best = data.thresholds[0];
    var bestD = Infinity;
    data.thresholds.forEach(function (row) {
      var d = Math.abs(row.threshold - t);
      if (d < bestD - 1e-9) {
        best = row;
        bestD = d;
      }
    });
    return best;
  }

  function perThousand(row, n) {
    return {
      flagged: (1000 * row.flagged) / n,
      caught: (1000 * row.tp) / n,
      missed: (1000 * row.fn) / n,
      false_flags: (1000 * row.fp) / n
    };
  }

  function renderAge(bands) {
    var root = document.getElementById("age-strip");
    var max = 0;
    bands.forEach(function (band) { if (band.rate > max) max = band.rate; });
    if (max <= 0) max = 1;
    root.replaceChildren();
    bands.forEach(function (band) {
      var row = document.createElement("div");
      row.className = "age-row";
      var label = document.createElement("span");
      label.className = "age-label";
      label.textContent = band.slice;
      var track = document.createElement("span");
      track.className = "age-track";
      var bar = document.createElement("span");
      bar.className = "age-bar";
      bar.style.width = (100 * band.rate / max) + "%";
      track.appendChild(bar);
      var rate = document.createElement("span");
      rate.className = "age-rate";
      rate.textContent = pct(band.rate);
      var n = document.createElement("span");
      n.className = "age-n";
      n.textContent = band.strokes + "/" + band.n;
      row.append(label, track, rate, n);
      root.appendChild(row);
    });
  }

  function renderModels(data) {
    var models = byName(data.models);
    var order = ["Majority rate", "Age only", "Logistic", "Decision tree", "Random forest"];
    var body = document.getElementById("model-body");
    body.replaceChildren();
    order.forEach(function (name) {
      var model = models[name];
      var tr = document.createElement("tr");
      if (name === data.desk_model) tr.className = "desk";
      var label = name;
      if (name === data.desk_model) label += "  desk";
      else if (!model.scores_are_risks) label += "  †";
      var cells = [
        label,
        fixed(model.roc_auc, 3),
        fixed(model.pr_auc, 3),
        fixed(model.brier, 3),
        pct(model.at_half.sensitivity, 1)
      ];
      cells.forEach(function (text, index) {
        var cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        if (index === 0 && name === data.desk_model) {
          cell.textContent = "Logistic";
          var tag = document.createElement("span");
          tag.className = "tag";
          tag.textContent = "desk";
          cell.appendChild(tag);
        } else if (index === 0 && !model.scores_are_risks) {
          cell.textContent = name + " †";
        } else {
          cell.textContent = text;
        }
        tr.appendChild(cell);
      });
      body.appendChild(tr);
    });

    var weighted = models["Logistic, class-weighted"];
    var age = models["Age only"];
    var desk = models["Logistic"];
    var lift = data.vs_age;
    var holdoutRate = data.dataset.test_strokes / data.dataset.test_rows;
    var rocCoversZero = lift.roc_auc_ci[0] <= 0 && lift.roc_auc_ci[1] >= 0;
    var prAboveZero = lift.pr_auc_ci[0] > 0;
    document.getElementById("model-lede").textContent =
      "Unweighted age reaches ROC " + fixed(age.roc_auc, 3) +
      " (95% CI " + range3(age.roc_auc_ci) + ") and Brier " + fixed(age.brier, 3) +
      ". The desk logistic reaches ROC " + fixed(desk.roc_auc, 3) +
      " (" + range3(desk.roc_auc_ci) + ") and Brier " + fixed(desk.brier, 3) +
      ". Paired ROC difference " + signed3(lift.roc_auc) +
      " (" + range3(lift.roc_auc_ci) + "); " +
      (rocCoversZero ? "that interval includes zero" : "that interval stays off zero") +
      ". PR-AUC moves from " + fixed(age.pr_auc, 3) + " to " + fixed(desk.pr_auc, 3) +
      ", difference " + signed3(lift.pr_auc) + " (" + range3(lift.pr_auc_ci) + "), " +
      (prAboveZero ? "and stays above zero" : "and includes zero") +
      ". Holdout prevalence is " + pct(holdoutRate) +
      ", and the majority PR-AUC of " + fixed(models["Majority rate"].pr_auc, 3) +
      " sits on that base rate. Intervals resample this holdout; the models are not refit.";
    document.getElementById("weighted-note").textContent =
      "† Tree, forest, and the class-weighted logistic are ranking scores. The class-weighted logistic, left off the chart, has ROC " +
      fixed(weighted.roc_auc, 3) + ", PR-AUC " + fixed(weighted.pr_auc, 3) +
      ", and Brier " + fixed(weighted.brier, 3) +
      ". Its 0.50 is not a 50% risk, and its Brier is not comparable to the desk model. " +
      "Age is unweighted, so its Brier is. The desk logistic is L2-penalized (C = 1) with no class weights. " +
      "Recall at 0.50 on that model is " + pct(desk.at_half.sensitivity, 1) +
      " because almost nobody is assigned a risk that high.";
  }

  function drawRoc(roc) {
    var svg = document.getElementById("roc");
    svg.replaceChildren();
    var W = 420;
    var H = 360;
    var L = 44;
    var T = 12;
    var R = 12;
    var B = 36;
    var plotW = W - L - R;
    var plotH = H - T - B;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    function x(f) { return L + f * plotW; }
    function y(t) { return T + (1 - t) * plotH; }

    [0, 0.5, 1].forEach(function (g) {
      svg.appendChild(svgEl("line", {
        x1: x(0), x2: x(1), y1: y(g), y2: y(g),
        stroke: "rgba(55,96,158,0.25)", "stroke-width": "1"
      }));
      svg.appendChild(svgEl("line", {
        x1: x(g), x2: x(g), y1: y(0), y2: y(1),
        stroke: "rgba(55,96,158,0.25)", "stroke-width": "1"
      }));
    });

    HERO.forEach(function (name) {
      var pts = roc[name];
      if (!pts) return;
      var style = STYLE[name];
      var points = pts.map(function (p) {
        return x(p.fpr).toFixed(1) + "," + y(p.tpr).toFixed(1);
      }).join(" ");
      var attrs = {
        points: points,
        fill: "none",
        stroke: style.color,
        "stroke-width": String(style.width),
        "stroke-linejoin": "round",
        "stroke-linecap": "round"
      };
      if (style.dash) attrs["stroke-dasharray"] = style.dash;
      svg.appendChild(svgEl("polyline", attrs));
    });

    [["0", x(0), y(0) + 14, "start"], ["1", x(1), y(0) + 14, "end"]].forEach(function (tick) {
      var text = svgEl("text", { x: tick[1], y: tick[2], "text-anchor": tick[3] });
      text.textContent = tick[0];
      svg.appendChild(text);
    });
    var xlab = svgEl("text", { x: x(0.5), y: H - 8, "text-anchor": "middle" });
    xlab.textContent = "False positive rate";
    var ylab = svgEl("text", {
      x: 14,
      y: y(0.5),
      "text-anchor": "middle",
      transform: "rotate(-90 14 " + y(0.5).toFixed(1) + ")"
    });
    ylab.textContent = "True positive rate";
    svg.append(xlab, ylab);

    var legend = document.getElementById("roc-legend");
    legend.replaceChildren();
    ["Majority rate", "Age only", "Logistic", "Decision tree", "Random forest"].forEach(function (name) {
      var item = document.createElement("span");
      var swatch = document.createElement("i");
      swatch.className = "swatch";
      swatch.style.borderTopColor = STYLE[name].color;
      if (STYLE[name].dash) swatch.style.borderTopStyle = "dashed";
      item.append(swatch, document.createTextNode(name === "Logistic" ? "Logistic (desk)" : name));
      legend.appendChild(item);
    });
  }

  function drawCalibration(bins, threshold) {
    var svg = document.getElementById("calibration");
    svg.replaceChildren();
    var W = 420;
    var H = 320;
    var L = 54;
    var T = 22;
    var R = 16;
    var B = 40;
    var plotW = W - L - R;
    var plotH = H - T - B;
    var xmax = 0.2;
    bins.forEach(function (bin) {
      xmax = Math.max(xmax, bin.mean_p, bin.observed);
      if (bin.observed_ci) xmax = Math.max(xmax, bin.observed_ci[1]);
    });
    xmax = Math.min(1, xmax * 1.25);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    function x(v) { return L + (v / xmax) * plotW; }
    function y(v) { return T + (1 - v / xmax) * plotH; }

    [0, xmax / 2, xmax].forEach(function (g) {
      svg.appendChild(svgEl("line", {
        x1: x(0), x2: x(xmax), y1: y(g), y2: y(g),
        stroke: "rgba(55,96,158,0.22)", "stroke-width": "1"
      }));
      var tick = svgEl("text", { x: L - 6, y: y(g) + 3, "text-anchor": "end" });
      tick.textContent = g === 0 ? "0" : g.toFixed(2);
      svg.appendChild(tick);
    });
    svg.appendChild(svgEl("line", {
      x1: x(0), y1: y(0), x2: x(xmax), y2: y(xmax),
      stroke: "#8d8376", "stroke-width": "1.2", "stroke-dasharray": "4 3"
    }));
    if (threshold != null && threshold > 0 && threshold < xmax) {
      svg.appendChild(svgEl("line", {
        x1: x(threshold), x2: x(threshold), y1: y(0), y2: y(xmax),
        stroke: "#241c16", "stroke-width": "1", "stroke-dasharray": "2 2"
      }));
    }

    bins.forEach(function (bin, index) {
      var cx = x(bin.mean_p);
      var cy = y(bin.observed);
      if (bin.observed_ci) {
        svg.appendChild(svgEl("line", {
          x1: cx.toFixed(1),
          x2: cx.toFixed(1),
          y1: y(bin.observed_ci[0]).toFixed(1),
          y2: y(bin.observed_ci[1]).toFixed(1),
          stroke: "#c7372f",
          "stroke-width": "1.2"
        }));
      }
      var radius = Math.max(3.5, Math.min(9, Math.sqrt(bin.n) * 0.32));
      svg.appendChild(svgEl("circle", {
        cx: cx.toFixed(1),
        cy: cy.toFixed(1),
        r: radius.toFixed(1),
        fill: "#c7372f",
        "fill-opacity": bin.n < 30 ? "0.55" : "0.9"
      }));
      var above = index % 2 === 0;
      var label = svgEl("text", {
        x: (cx + radius + 4).toFixed(1),
        y: (above ? cy - radius - 8 : cy + radius + 14).toFixed(1)
      });
      label.textContent = "n=" + bin.n;
      if (bin.mean_p > xmax * 0.72) {
        label.setAttribute("text-anchor", "end");
        label.setAttribute("x", (cx - radius - 4).toFixed(1));
        label.setAttribute("y", (cy - radius - 6).toFixed(1));
      }
      svg.appendChild(label);
    });

    var xlab = svgEl("text", { x: x(xmax / 2), y: H - 8, "text-anchor": "middle" });
    xlab.textContent = "Mean predicted risk";
    var ylab = svgEl("text", {
      x: 14,
      y: y(xmax / 2),
      "text-anchor": "middle",
      transform: "rotate(-90 14 " + y(xmax / 2).toFixed(1) + ")"
    });
    ylab.textContent = "Observed rate";
    svg.append(xlab, ylab);
    [0, xmax].forEach(function (tick, index) {
      var text = svgEl("text", {
        x: x(tick),
        y: y(0) + 14,
        "text-anchor": index === 0 ? "start" : "end"
      });
      text.textContent = tick === 0 ? "0" : tick.toFixed(2);
      svg.appendChild(text);
    });
  }

  function renderOdds(data) {
    var body = document.getElementById("odds-body");
    body.replaceChildren();
    data.odds_ratios.forEach(function (row) {
      var tr = document.createElement("tr");
      if (row.term.indexOf("Age") === 0) tr.className = "age-term";
      var excludes = row.ci_low > 1 || row.ci_high < 1;
      var cells = [
        row.term + (excludes ? " *" : ""),
        orNum(row.odds_ratio),
        orNum(row.ci_low) + "–" + orNum(row.ci_high),
        formatP(row.p_value)
      ];
      cells.forEach(function (text, index) {
        var cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        cell.textContent = text;
        tr.appendChild(cell);
      });
      body.appendChild(tr);
    });

    var heartRows = data.heart_disease;
    var yes = heartRows.filter(function (row) { return Number(row.slice) === 1; })[0];
    var no = heartRows.filter(function (row) { return Number(row.slice) === 0; })[0];
    var heart = data.odds_ratios.filter(function (row) { return row.term === "Heart disease"; })[0];
    var clear = data.odds_ratios.filter(function (row) {
      return row.ci_low > 1 || row.ci_high < 1;
    }).map(function (row) { return row.term; });
    var association = data.association_model && data.association_model.note
      ? data.association_model.note
      : data.odds_design;
    document.getElementById("odds-note").textContent =
      association +
      " Heart disease is " + pct(yes.rate) + " versus " + pct(no.rate) +
      " in the raw file, then an odds ratio of " + orNum(heart.odds_ratio) +
      " once age is in the model. * marks intervals that exclude 1: " +
      clear.join("; ") + ".";
  }

  function fillStatGrid(root, pairs) {
    root.replaceChildren();
    pairs.forEach(function (pair) {
      var cell = document.createElement("div");
      var name = document.createElement("span");
      name.textContent = pair[0];
      var value = document.createElement("strong");
      value.textContent = pair[1];
      cell.append(name, value);
      if (pair[2]) {
        var note = document.createElement("em");
        note.textContent = pair[2];
        cell.appendChild(note);
      }
      root.appendChild(cell);
    });
  }

  function applyThreshold(row) {
    var data = DATA;
    var n = data.dataset.test_rows;
    var operating = data.operating;
    var same = Math.abs(row.threshold - operating.threshold) < 1e-9;
    document.getElementById("threshold-value").textContent = fixed(row.threshold, 2);
    document.getElementById("threshold-tag").textContent = same
      ? "Casebook operating point"
      : "Grid point on the desk model";
    document.getElementById("threshold").value = String(row.threshold);
    fillStatGrid(document.getElementById("quad"), [
      ["Sensitivity", pct(row.sensitivity), ciPct(row.sensitivity_ci)],
      ["Specificity", pct(row.specificity), ciPct(row.specificity_ci)],
      ["PPV", pct(row.ppv), ciPct(row.ppv_ci)],
      ["False flags / true", fixed(row.false_flags_per_true, 2)]
    ]);
    var sensCi = row.sensitivity_ci ? " (" + ciPct(row.sensitivity_ci) + ")" : "";
    document.getElementById("point-readout").textContent =
      "At " + fixed(row.threshold, 2) + " the holdout flags " + row.flagged +
      " of " + n + " people and catches " + row.tp + " of " + data.dataset.test_strokes +
      " strokes" + sensCi + ", missing " + row.fn + ". About " + fixed(row.false_flags_per_true, 2) +
      " false flags arrive for each stroke caught.";

    var burden = perThousand(row, n);
    document.getElementById("tally-kicker").textContent =
      "Per 1,000 people like the holdout · threshold " + fixed(row.threshold, 2) +
      (same ? " · operating point" : "");
    fillStatGrid(document.getElementById("tally"), [
      ["Flagged", oneDecimal(burden.flagged)],
      ["Caught", oneDecimal(burden.caught)],
      ["Missed", oneDecimal(burden.missed)],
      ["False flags", oneDecimal(burden.false_flags)]
    ]);
  }

  function render(data) {
    DATA = data;
    var ds = data.dataset;
    document.getElementById("stroke-count").textContent = String(ds.strokes);
    document.getElementById("stroke-denom").textContent =
      "of " + ds.rows.toLocaleString("en-US") + " rows · " + pct(ds.prevalence, 2);
    var correct = (1 - ds.prevalence) * 100;
    document.getElementById("open-lede").textContent =
      pct(ds.prevalence, 2) + " of the file has a stroke recorded. Calling every row low-risk is right " +
      correct.toFixed(1) + "% of the time and catches nobody. Majority accuracy is the trap.";
    var imp = ds.imputation;
    document.getElementById("open-method").textContent =
      "Stratified 80/20 before imputation, seed " + ds.seed + ". Train median BMI " +
      Number(imp.train_median).toFixed(1) + "; the full-file median is " +
      Number(imp.full_median).toFixed(1) + ". " + imp.missing_rows + " BMI values were missing.";
    renderAge(data.age_bands);
    var young = data.age_bands[0];
    var old = data.age_bands[data.age_bands.length - 1];
    document.getElementById("age-note").textContent =
      "Under 18 the rate is " + pct(young.rate) + " (" + young.strokes + " of " + young.n +
      "). At 80 and older it is " + pct(old.rate) + " (" + old.strokes + " of " + old.n +
      "). Age dominates the file.";

    renderModels(data);
    drawRoc(data.roc);
    var selectionNote = data.operating && data.operating.selection_note;
    document.getElementById("point-lede").textContent = selectionNote
      ? selectionNote
      : "Probabilities come from the unweighted logistic. Among thresholds with sensitivity of at least 70%, the casebook keeps the one that flags the fewest people: " +
        fixed(data.operating.threshold, 2) + ". The slider reads the precomputed grid, including ?t=0.15.";

    var input = document.getElementById("threshold");
    var grid = data.thresholds;
    input.min = String(grid[0].threshold);
    input.max = String(grid[grid.length - 1].threshold);
    input.step = "0.01";
    document.getElementById("scale-min").textContent = fixed(grid[0].threshold, 2);
    var span = Number(input.max) - Number(input.min);
    var mark = (data.operating.threshold - Number(input.min)) / span;
    document.getElementById("desk-mark").style.left = (mark * 100) + "%";

    drawCalibration(data.calibration, data.operating.threshold);
    var summary = data.calibration_summary;
    var models = byName(data.models);
    var coversOne = summary.slope_ci[0] < 1 && summary.slope_ci[1] > 1;
    var covering = null;
    data.calibration.forEach(function (bin) {
      if (bin.lo <= data.operating.threshold && data.operating.threshold <= bin.hi) covering = bin;
    });
    var calNote =
      "Equal-count bins, so scores below 0.12 are not one dot. The dashed line is the operating threshold. " +
      "Slope " + fixed(summary.slope, 2) + " (95% CI " + fixed(summary.slope_ci[0], 2) +
      " to " + fixed(summary.slope_ci[1], 2) + ") on the holdout logit; " +
      (coversOne ? "the interval includes 1" : "the interval excludes 1") +
      ", and these probabilities were not rescaled to that fit. Brier " +
      fixed(data.brier_desk, 3) + " versus " + fixed(models["Age only"].brier, 3) +
      " for unweighted age and " + fixed(models["Majority rate"].brier, 3) +
      " for the constant rate.";
    if (covering && covering.observed_ci) {
      calNote += " The bin that contains " + fixed(data.operating.threshold, 2) +
        " has mean predicted " + fixed(covering.mean_p, 3) + " and observed " +
        pct(covering.observed) + " (" + ciPct(covering.observed_ci) + ", n=" + covering.n + ").";
    }
    document.getElementById("cal-note").textContent = calNote;
    renderOdds(data);
    document.getElementById("footer-note").textContent = data.note;

    var params = new URLSearchParams(location.search);
    var requested = params.get("t");
    var start = requested == null || requested === "" || Number.isNaN(Number(requested))
      ? data.operating.threshold
      : Number(requested);
    applyThreshold(rowForThreshold(data, start));

    input.addEventListener("input", function () {
      var row = rowForThreshold(data, Number(input.value));
      applyThreshold(row);
      var next = new URLSearchParams(location.search);
      next.set("t", Number(row.threshold).toFixed(2));
      history.replaceState(null, "", "?" + next.toString());
    });

    document.getElementById("status").textContent = "";
    document.documentElement.dataset.ready = "1";
  }

  function fail(message) {
    var status = document.getElementById("status");
    status.classList.add("show");
    status.textContent = message;
  }

  fetch("metrics.json")
    .then(function (response) {
      if (!response.ok) throw new Error("metrics.json " + response.status);
      return response.json();
    })
    .then(render)
    .catch(function () {
      fail("metrics.json did not load. From the repo root, run python3 -m casebook.analyze and then python3 -m http.server.");
    });
})();
