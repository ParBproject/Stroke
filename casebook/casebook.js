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
    var holdoutRate = data.dataset.test_strokes / data.dataset.test_rows;
    document.getElementById("model-lede").textContent =
      "Age alone reaches ROC " + fixed(age.roc_auc, 3) +
      ". The unweighted logistic reaches " + fixed(desk.roc_auc, 3) +
      ", a modest lift. PR-AUC, the rare-event metric, moves from " +
      fixed(age.pr_auc, 3) + " to " + fixed(desk.pr_auc, 3) +
      ". Holdout prevalence is " + pct(holdoutRate) +
      ", and the majority PR-AUC of " + fixed(models["Majority rate"].pr_auc, 3) +
      " sits on that base rate.";
    document.getElementById("weighted-note").textContent =
      "† Class-weighted or balanced scores are for ranking. The class-weighted logistic, left off the chart, has ROC " +
      fixed(weighted.roc_auc, 3) + ", PR-AUC " + fixed(weighted.pr_auc, 3) +
      ", and Brier " + fixed(weighted.brier, 3) +
      ". Its 0.50 is not a 50% risk. The desk model is the unweighted logistic: Brier " +
      fixed(desk.brier, 3) + ", and recall at 0.50 is " + pct(desk.at_half.sensitivity, 1) +
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

  function drawCalibration(bins) {
    var svg = document.getElementById("calibration");
    svg.replaceChildren();
    var W = 420;
    var H = 320;
    var L = 48;
    var T = 22;
    var R = 18;
    var B = 40;
    var plotW = W - L - R;
    var plotH = H - T - B;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    function x(v) { return L + v * plotW; }
    function y(v) { return T + (1 - v) * plotH; }

    [0, 0.5, 1].forEach(function (g) {
      svg.appendChild(svgEl("line", {
        x1: x(0), x2: x(1), y1: y(g), y2: y(g),
        stroke: "rgba(55,96,158,0.22)", "stroke-width": "1"
      }));
    });
    svg.appendChild(svgEl("line", {
      x1: x(0), y1: y(0), x2: x(1), y2: y(1),
      stroke: "#8d8376", "stroke-width": "1.2", "stroke-dasharray": "4 3"
    }));

    bins.forEach(function (bin) {
      var cx = x(bin.mean_p);
      var cy = y(bin.observed);
      var radius = Math.max(3.5, Math.min(9, Math.sqrt(bin.n) * 0.32));
      svg.appendChild(svgEl("circle", {
        cx: cx.toFixed(1),
        cy: cy.toFixed(1),
        r: radius.toFixed(1),
        fill: "#c7372f",
        "fill-opacity": bin.n < 10 ? "0.55" : "0.9"
      }));
      var label = svgEl("text", { x: (cx + radius + 5).toFixed(1), y: (cy - 6).toFixed(1) });
      label.textContent = "n=" + bin.n;
      if (bin.observed > 0.8) {
        label.setAttribute("text-anchor", "end");
        label.setAttribute("x", (cx - radius - 4).toFixed(1));
        label.setAttribute("y", (cy + 4).toFixed(1));
      } else if (bin.n > 200) {
        label.setAttribute("y", (cy - radius - 4).toFixed(1));
      }
      svg.appendChild(label);
    });

    var xlab = svgEl("text", { x: x(0.5), y: H - 8, "text-anchor": "middle" });
    xlab.textContent = "Mean predicted risk";
    var ylab = svgEl("text", {
      x: 14,
      y: y(0.5),
      "text-anchor": "middle",
      transform: "rotate(-90 14 " + y(0.5).toFixed(1) + ")"
    });
    ylab.textContent = "Observed rate";
    svg.append(xlab, ylab);
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
    document.getElementById("odds-note").textContent =
      "Unweighted logit on the training fold. Smoking contrasts are against never smoked. " +
      "Work type and gender Other were omitted; those levels separated. " +
      "Heart disease is " + pct(yes.rate) + " versus " + pct(no.rate) +
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
      ["Sensitivity", pct(row.sensitivity)],
      ["Specificity", pct(row.specificity)],
      ["PPV", pct(row.ppv)],
      ["False flags / true", fixed(row.false_flags_per_true, 2)]
    ]);
    document.getElementById("point-readout").textContent =
      "At " + fixed(row.threshold, 2) + " the holdout flags " + row.flagged +
      " of " + n + " people and catches " + row.tp + " of " + data.dataset.test_strokes +
      " strokes, missing " + row.fn + ". About " + fixed(row.false_flags_per_true, 2) +
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
    document.getElementById("point-lede").textContent =
      "Probabilities come from the unweighted logistic. Among thresholds with sensitivity of at least 70%, the casebook keeps the one that flags the fewest people: " +
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

    drawCalibration(data.calibration);
    var biggest = data.calibration.slice().sort(function (a, b) { return b.n - a.n; })[0];
    var top = data.calibration[data.calibration.length - 1];
    document.getElementById("cal-note").textContent =
      "Brier " + fixed(data.brier_desk, 3) + " against " +
      fixed(byName(data.models)["Majority rate"].brier, 3) +
      " for the constant base rate. " + biggest.n + " of " + ds.test_rows +
      " holdout rows sit in one low bin (mean predicted " + fixed(biggest.mean_p, 3) +
      ", observed " + pct(biggest.observed) + "). The highest bin has n=" + top.n +
      " and is not a calibrated mass of 80% risks.";
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
