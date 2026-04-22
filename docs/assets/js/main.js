(function () {
  'use strict';

  var DATA_FILE = 'assets/data/site_data.json';
  var ROOT_PREFIX = '';
  var PAGE = 'home';
  var ARTIFACT_ID = '';
  var SITE_VERSION = '';
  var SITE_DATA = null;

  function initPageConfig() {
    var body = document.body || document.getElementsByTagName('body')[0];
    if (!body) return;
    ROOT_PREFIX = body.getAttribute('data-root-prefix') || '';
    PAGE = body.getAttribute('data-page') || 'home';
    ARTIFACT_ID = body.getAttribute('data-artifact-id') || '';
    SITE_VERSION = body.getAttribute('data-site-version') || '';
  }

  function resolvePath(path) {
    return ROOT_PREFIX + path;
  }

  function versionedPath(path) {
    if (!SITE_VERSION) return path;
    return path + (path.indexOf('?') === -1 ? '?v=' : '&v=') + encodeURIComponent(SITE_VERSION);
  }

  function endsWith(text, suffix) {
    return String(text).slice(-suffix.length) === suffix;
  }

  function isDownloadable(path) {
    var suffixes = ['.csv', '.json', '.svg', '.md'];
    for (var i = 0; i < suffixes.length; i++) {
      if (endsWith(path, suffixes[i])) return true;
    }
    return false;
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderDefinitionList(definitions) {
    if (!definitions || !definitions.length) return '';
    var rows = [];
    for (var i = 0; i < definitions.length; i++) {
      var item = definitions[i] || [];
      var symbolHtml = item.length > 2 ? String(item[2] || '') : '';
      var symbolMarkup = symbolHtml
        ? '<span class="equation-symbol">' + symbolHtml + '</span>'
        : '<span class="math-inline">\\(' + escapeHtml(item[0] || '') + '\\)</span>';
      rows.push(
        '<div class="definition-row"><dt>' +
        symbolMarkup +
        '</dt><dd>' +
        escapeHtml(item[1] || '') +
        '</dd></div>'
      );
    }
    return '<dl class="equation-definitions">' + rows.join('') + '</dl>';
  }

  function renderEquationMarkup(latex, htmlMarkup) {
    if (htmlMarkup) {
      return '<div class="math"><div class="equation-rendered">' + htmlMarkup + '</div></div>';
    }
    return '<div class="math">\\[' + escapeHtml(latex || '') + '\\]</div>';
  }

  function linkMarkup(link, className) {
    var cls = className || 'link-chip';
    var download = isDownloadable(link.href) ? ' download' : '';
    return '<a class="' + cls + '" href="' + escapeHtml(resolvePath(link.href)) + '"' + download + '>' + escapeHtml(link.label) + '</a>';
  }

  function getThemePalette() {
    var dark = (document.documentElement.getAttribute('data-theme') || 'light') === 'dark';
    if (dark) {
      return {
        paper: '#1c2128',
        plot: '#1c2128',
        text: '#e6edf3',
        grid: '#30363d',
        axis: '#4a525d',
        hoverBg: '#242a33',
        hoverFont: '#ffffff',
        colors: ['#58a6ff', '#d29922', '#f85149', '#3fb950', '#c27abf', '#e0a050']
      };
    }
    return {
      paper: '#ffffff',
      plot: '#ffffff',
      text: '#17171a',
      grid: '#e2e8f0',
      axis: '#9aa5b3',
      hoverBg: '#19385f',
      hoverFont: '#ffffff',
      colors: ['#2b6cb0', '#d69e2e', '#9b2c2c', '#276749', '#805ad5', '#dd6b20']
    };
  }

  function rgba(hex, alpha) {
    var value = String(hex).replace('#', '');
    var bigint = parseInt(value, 16);
    var r = (bigint >> 16) & 255;
    var g = (bigint >> 8) & 255;
    var b = bigint & 255;
    return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
  }

  function requestText(path, callback) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', resolvePath(path), true);
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) return;
      if (xhr.status >= 200 && xhr.status < 300) {
        callback(null, xhr.responseText);
      } else {
        callback(new Error('Failed to load ' + path + ': ' + xhr.status));
      }
    };
    xhr.send();
  }

  function requestJson(path, callback) {
    requestText(path, function (error, text) {
      if (error) {
        callback(error);
        return;
      }
      try {
        callback(null, JSON.parse(text));
      } catch (parseError) {
        callback(parseError);
      }
    });
  }

  function parseCsv(text) {
    var lines = String(text || '').trim().split(/\r?\n/);
    if (!lines.length || !lines[0]) return [];
    var headers = lines[0].split(',');
    var rows = [];
    for (var i = 1; i < lines.length; i++) {
      if (!lines[i]) continue;
      var values = lines[i].split(',');
      var row = {};
      for (var j = 0; j < headers.length; j++) {
        row[headers[j]] = values[j] !== undefined ? values[j] : '';
      }
      rows.push(row);
    }
    return rows;
  }

  function loadSiteData(callback) {
    requestJson(versionedPath(DATA_FILE), function (error, data) {
      if (error) {
        callback(error);
        return;
      }
      SITE_DATA = data;
      callback(null, data);
    });
  }

  function typesetMath(targets) {
    if (window.MathJax && window.MathJax.typesetPromise) {
      return window.MathJax.typesetPromise(targets || []);
    }
    return Promise.resolve();
  }

  function scheduleTypeset(targets, attempt) {
    var tries = attempt || 0;
    if (window.MathJax && window.MathJax.typesetPromise) {
      typesetMath(targets);
      return;
    }
    if (tries >= 40) {
      return;
    }
    window.setTimeout(function () {
      scheduleTypeset(targets, tries + 1);
    }, 250);
  }

  function revealAll() {
    if (!window.IntersectionObserver) return;
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    var nodes = document.querySelectorAll('.reveal');
    for (var i = 0; i < nodes.length; i++) {
      observer.observe(nodes[i]);
    }
  }

  function renderMetrics() {
    var target = document.getElementById('metric-grid');
    if (!target || !SITE_DATA) return;
    var html = '';
    var metrics = SITE_DATA.metrics || [];
    for (var i = 0; i < metrics.length; i++) {
      html += '<article class="metric-card reveal"><span>' + escapeHtml(metrics[i].label) + '</span><strong>' + escapeHtml(metrics[i].value) + '</strong><p>' + escapeHtml(metrics[i].note) + '</p></article>';
    }
    target.innerHTML = html;
  }

  function renderInsightCards(cards, targetId) {
    var target = document.getElementById(targetId || 'insight-grid');
    if (!target) return;
    var html = '';
    for (var i = 0; i < cards.length; i++) {
      html += '<article class="insight-card reveal"><div class="eyebrow">' + escapeHtml(cards[i].kicker) + '</div><h3>' + escapeHtml(cards[i].title) + '</h3><p>' + escapeHtml(cards[i].body) + '</p></article>';
    }
    target.innerHTML = html;
  }

  function buildLinkRow(links, className) {
    var html = '';
    for (var i = 0; i < links.length; i++) {
      html += linkMarkup(links[i], className);
    }
    return html;
  }

  function renderJobList(jobIds, targetId) {
    var target = document.getElementById(targetId || 'job-list');
    if (!target || !SITE_DATA) return;
    var html = '';
    for (var i = 0; i < jobIds.length; i++) {
      var job = SITE_DATA.jobs[jobIds[i]];
      if (!job) continue;
      var branchNote = job.branch_note ? '<p class="branch-note">' + escapeHtml(job.branch_note) + '</p>' : '';
      html += '<section class="section-block job-block reveal" id="' + escapeHtml(job.anchor) + '">';
      html += '<div class="job-top"><div><div class="eyebrow">' + escapeHtml(job.kicker) + '</div><h3>' + escapeHtml(job.title) + '</h3><p>' + escapeHtml(job.subtitle) + '</p>';
      html += '<div class="meta-row"><span>' + escapeHtml(job.estimator_label) + '</span><span>' + escapeHtml(job.observation_label) + '</span><span>' + escapeHtml(job.treatment_label) + '</span><span>' + escapeHtml(job.branch_label) + '</span></div>' + branchNote + '</div>';
      html += '<div><p>' + escapeHtml(job.summary) + '</p><div class="button-row" style="margin-top:14px;">' + buildLinkRow(job.links) + '</div></div></div>';
      html += '<div class="mini-grid">';
      for (var j = 0; j < job.outcomes.length; j++) {
        var outcome = job.outcomes[j];
        html += '<article class="mini-chart-card"><div><h4>' + escapeHtml(outcome.label) + '</h4><p class="mini-note">Quarter-by-quarter response under the selected public control branch.</p></div><div class="mini-chart" id="' + escapeHtml(outcome.chart_dom_id) + '"></div></article>';
      }
      html += '</div></section>';
    }
    target.innerHTML = html;
  }

  function renderDepositAccounting() {
    var target = document.getElementById('independent-evidence');
    if (!target || !SITE_DATA || !SITE_DATA.home || !SITE_DATA.home.independent_evidence) return;
    var block = SITE_DATA.home.independent_evidence;
    var html = '';
      html += '<section class="section-block job-block reveal" id="independent-evidence-block">';
      html += '<div class="job-top"><div><div class="eyebrow">Independent non-TDC evidence</div><h3>' + escapeHtml(block.title) + '</h3><p>' + escapeHtml(block.subtitle) + '</p></div>';
      html += '<div><p>' + escapeHtml(block.summary) + '</p><p class="branch-note">' + escapeHtml(block.impact_summary || '') + '</p><div class="button-row" style="margin-top:14px;">' + buildLinkRow(block.links || []) + '</div></div></div>';
      html += '<div class="mini-grid">';
      for (var i = 0; i < block.outcomes.length; i++) {
        var outcome = block.outcomes[i];
        html += '<article class="mini-chart-card"><div><h4>' + escapeHtml(outcome.label) + '</h4><p class="mini-note">Quarter-by-quarter response under the imported `tdcpass` strict-source comparison.</p></div><div class="mini-chart" id="' + escapeHtml(outcome.chart_dom_id) + '"></div></article>';
      }
      html += '</div>';
      html += '<div class="insight-grid" style="margin-top:18px;">';
      for (var j = 0; j < (block.note_lines || []).length; j++) {
        html += '<article class="insight-card"><div class="slot-label">Boundary</div><p>' + escapeHtml(block.note_lines[j]) + '</p></article>';
      }
      html += '</div>';
    html += '</section>';
    target.innerHTML = html;
  }

  function renderTreatmentComparisons() {
    var target = document.getElementById('treatment-comparisons');
    if (!target || !SITE_DATA || !SITE_DATA.sidecar || !SITE_DATA.sidecar.treatment_comparisons) return;
    var blocks = SITE_DATA.sidecar.treatment_comparisons;
    var html = '';
    for (var i = 0; i < blocks.length; i++) {
      var block = blocks[i];
      html += '<section class="section-block job-block reveal">';
      html += '<div class="job-top"><div><div class="eyebrow">Treatment variants</div><h3>' + escapeHtml(block.title) + '</h3><p>' + escapeHtml(block.subtitle) + '</p></div>';
      html += '<div><p>' + escapeHtml(block.summary) + '</p><div class="button-row" style="margin-top:14px;">' + buildLinkRow(block.links) + '</div></div></div>';
      html += '<div class="mini-grid">';
      for (var j = 0; j < block.outcomes.length; j++) {
        var outcome = block.outcomes[j];
        html += '<article class="mini-chart-card"><div><h4>' + escapeHtml(outcome.label) + '</h4><p class="mini-note">Each line shows the same quarterly response under a different TDC construction.</p></div><div class="mini-chart" id="' + escapeHtml(outcome.chart_dom_id) + '"></div></article>';
      }
      html += '</div></section>';
    }
    target.innerHTML = html;
  }

  function renderIvLabSummary() {
    var target = document.getElementById('iv-lab-summary');
    if (!target || !SITE_DATA || !SITE_DATA.sidecar || !SITE_DATA.sidecar.iv_lab) return;
    var block = SITE_DATA.sidecar.iv_lab;
    var html = '<section class="section-block reveal"><div class="job-top"><div><div class="eyebrow">IV search</div><h3>IV mining results.</h3><p>The current search scans available shock × state candidates and compares the configured instrument with the best alternative found in the current data.</p></div><div><div class="meta-row"><span>' + escapeHtml(String(block.jobs_scanned)) + ' IV jobs</span><span>' + escapeHtml(String(block.total_candidates)) + ' candidates</span></div><div class="button-row" style="margin-top:14px;">' + buildLinkRow(block.links) + '</div></div></div>';
    html += '<div class="gallery-grid">';
    for (var i = 0; i < block.jobs.length; i++) {
      var job = block.jobs[i];
      html += '<article class="artifact-card"><div class="slot-label">IV search</div><h3>' + escapeHtml(job.job_id) + '</h3><p>Current median F: ' + escapeHtml(job.current_median_f == null ? 'n/a' : Number(job.current_median_f).toFixed(2)) + ' • Best median F: ' + escapeHtml(job.best_median_f == null ? 'n/a' : Number(job.best_median_f).toFixed(2)) + '</p><p>Current: ' + escapeHtml(job.current_recommendation || 'n/a') + '<br>Best alternative: ' + escapeHtml(job.best_recommendation || 'n/a') + '</p></article>';
    }
    html += '</div></section>';
    target.innerHTML = html;
  }

  function renderRobustnessSummary() {
    var overviewTarget = document.getElementById('robustness-overview');
    var gridTarget = document.getElementById('robustness-grid');
    if (!overviewTarget || !gridTarget || !SITE_DATA || !SITE_DATA.robustness) return;
    var robustness = SITE_DATA.robustness;
    var overview = robustness.overview || {};
    overviewTarget.innerHTML =
      '<section class="section-block robustness-overview reveal">' +
      '<div><div class="eyebrow">Robustness</div><h3>' + escapeHtml(overview.title || 'Larger control set') + '</h3><p>' + escapeHtml(overview.summary || '') + '</p><div class="button-row" style="margin-top:14px;">' + buildLinkRow(overview.links || []) + '</div></div>' +
      '<div class="robustness-stats">' +
      '<article class="robustness-stat"><span>Raw series</span><strong>' + escapeHtml(String(overview.series_count || '0')) + '</strong></article>' +
      '<article class="robustness-stat"><span>Lagged indicators</span><strong>' + escapeHtml(String(overview.feature_count || '0')) + '</strong></article>' +
      '<article class="robustness-stat"><span>Daily lags</span><strong>' + escapeHtml(String(overview.daily_lags || '0')) + '</strong></article>' +
      '<article class="robustness-stat"><span>Common K</span><strong>' + escapeHtml(String(overview.recommended_k_mode || 'n/a')) + '</strong></article>' +
      '</div></section>';

    var html = '';
    for (var i = 0; i < robustness.jobs.length; i++) {
      var job = robustness.jobs[i];
      html += '<article class="robustness-card section-block reveal">';
      html += '<div class="slot-label">Robustness</div><h3>' + escapeHtml(job.title) + '</h3><p>' + escapeHtml(job.summary) + '</p>';
      html += '<div class="meta-row"><span>Cross-check: ' + escapeHtml(String(job.ml_public_branch_label || 'DML')) + '</span><span>K=' + escapeHtml(String(job.recommended_k || 'n/a')) + '</span><span>' + escapeHtml(String(job.screened_feature_count || '0')) + ' lagged indicators</span><span>' + escapeHtml(String(job.treatment_variant_count || '0')) + ' treatment variants</span></div>';
      html += '<div class="mini-grid">';
      html += '<article class="mini-chart-card"><div><h4>Control ladder</h4><p class="mini-note">Average absolute response magnitude under the baseline and expanded control sets.</p></div><div class="robustness-chart" id="' + escapeHtml(job.chart_dom_id) + '"></div></article>';
      html += '<article class="mini-chart-card"><div><h4>Interpretation</h4><p class="mini-note">' + escapeHtml(job.interpretation) + '</p><p class="mini-note">' + escapeHtml(job.ml_public_branch_reason || '') + '</p></div><div class="robustness-links">' + buildLinkRow(job.links || []) + '</div></article>';
      html += '</div></article>';
    }
    gridTarget.innerHTML = html;
  }

  function renderArtifacts(mainRows, appendixRows) {
    var mainTarget = document.getElementById('artifact-grid-main');
    var appendixTarget = document.getElementById('artifact-grid-appendix');
    if (!mainTarget || !appendixTarget || !SITE_DATA) return;

    function cardMarkup(rows) {
      var html = '';
      for (var i = 0; i < rows.length; i++) {
        html += '<article class="artifact-card reveal"><div class="slot-label">' + escapeHtml(rows[i].slot_label) + '</div><h3>' + escapeHtml(rows[i].title) + '</h3><p>' + escapeHtml(rows[i].subtitle) + '</p><div class="button-row">' + buildLinkRow(rows[i].links) + '</div></article>';
      }
      return html;
    }

    mainTarget.innerHTML = cardMarkup(mainRows);
    appendixTarget.innerHTML = cardMarkup(appendixRows);
  }

  function renderDeferred(rows) {
    var target = document.getElementById('deferred-grid');
    if (!target) return;
    var html = '';
    for (var i = 0; i < rows.length; i++) {
      var meta = '';
      for (var j = 0; j < rows[i].meta.length; j++) {
        meta += '<span>' + escapeHtml(rows[i].meta[j]) + '</span>';
      }
      html += '<article class="deferred-card reveal"><div class="slot-label">' + escapeHtml(rows[i].tag) + '</div><h3>' + escapeHtml(rows[i].title) + '</h3><p>' + escapeHtml(rows[i].reason) + '</p><div class="meta-row">' + meta + '</div></article>';
    }
    target.innerHTML = html;
  }

  function renderGallery(rows) {
    var target = document.getElementById('artifact-gallery-grid');
    if (!target) return;
    var html = '';
    for (var i = 0; i < rows.length; i++) {
      html += '<article class="artifact-card reveal"><div class="slot-label">' + escapeHtml(rows[i].slot_label) + '</div><h3>' + escapeHtml(rows[i].title) + '</h3><p>' + escapeHtml(rows[i].subtitle) + '</p><div class="button-row">' + buildLinkRow(rows[i].links) + '</div></article>';
    }
    target.innerHTML = html;
  }

  function renderArtifactTableRows(rows) {
    var html = '';
    for (var i = 0; i < rows.length; i++) {
      html += '<tr><td>' + escapeHtml(rows[i].outcome_label) + '</td><td>' + escapeHtml(rows[i].horizon) + '</td><td>' + escapeHtml(rows[i].beta) + '</td><td>' + escapeHtml(rows[i].se) + '</td><td>' + escapeHtml(rows[i].lower95) + '</td><td>' + escapeHtml(rows[i].upper95) + '</td><td>' + escapeHtml(rows[i].p_value_normal) + '</td><td>' + escapeHtml(rows[i].n) + '</td></tr>';
    }
    return html;
  }

  function renderArtifactDetail(artifact, callback) {
    var summaryTarget = document.getElementById('artifact-summary');
    var bodyTarget = document.getElementById('artifact-body');
    if (!summaryTarget || !bodyTarget || !artifact || !SITE_DATA) {
      if (callback) callback();
      return;
    }
    summaryTarget.innerHTML = '<div class="eyebrow">' + escapeHtml(artifact.slot_label) + '</div><h1>' + escapeHtml(artifact.title) + '</h1><p>' + escapeHtml(artifact.subtitle) + '</p><div class="artifact-downloads">' + buildLinkRow(artifact.links, 'button') + '</div>';

    if (artifact.kind === 'figure') {
      var job = SITE_DATA.jobs[artifact.job_id];
      var outcomes = [];
      for (var i = 0; i < job.outcomes.length; i++) {
        if ((artifact.outcome_ids || []).indexOf(job.outcomes[i].key) !== -1) outcomes.push(job.outcomes[i]);
      }
      var figureHtml = '<section class="section-block reveal"><p>' + escapeHtml(artifact.caption) + '</p><div class="mini-grid" style="margin-top:20px;">';
      for (var j = 0; j < outcomes.length; j++) {
        figureHtml += '<article class="mini-chart-card"><div><h4>' + escapeHtml(outcomes[j].label) + '</h4><p class="mini-note">Quarter-by-quarter response of this outcome.</p></div><div class="mini-chart" id="' + escapeHtml(artifact.artifact_id + '-' + outcomes[j].key) + '"></div></article>';
      }
      figureHtml += '</div></section>';
      bodyTarget.innerHTML = figureHtml;
      if (callback) callback();
      return;
    }

    var jobRows = [];
    var jobData = SITE_DATA.jobs[artifact.job_id];
    var k;
    if (artifact.table_rows && artifact.table_rows.length) {
      jobRows = artifact.table_rows;
      bodyTarget.innerHTML = '<section class="table-shell reveal"><p style="margin-bottom:14px;">' + escapeHtml(artifact.caption) + '</p><table class="result-table"><thead><tr><th>Outcome</th><th>H</th><th>Beta</th><th>SE</th><th>95% lower</th><th>95% upper</th><th>p-value</th><th>N</th></tr></thead><tbody>' + renderArtifactTableRows(jobRows) + '</tbody></table></section>';
      if (callback) callback();
      return;
    }

    if (artifact.outcome_ids && artifact.outcome_ids.length && artifact.horizons && artifact.horizons.length) {
      for (k = 0; k < jobData.table_rows.length; k++) {
        if (artifact.outcome_ids.indexOf(jobData.table_rows[k].outcome) !== -1 && artifact.horizons.indexOf(jobData.table_rows[k].horizon) !== -1) {
          jobRows.push(jobData.table_rows[k]);
        }
      }
      bodyTarget.innerHTML = '<section class="table-shell reveal"><p style="margin-bottom:14px;">' + escapeHtml(artifact.caption) + '</p><table class="result-table"><thead><tr><th>Outcome</th><th>H</th><th>Beta</th><th>SE</th><th>95% lower</th><th>95% upper</th><th>p-value</th><th>N</th></tr></thead><tbody>' + renderArtifactTableRows(jobRows) + '</tbody></table></section>';
      if (callback) callback();
      return;
    }

    var csvLink = null;
    for (k = 0; k < artifact.links.length; k++) {
      if (artifact.links[k].label === 'CSV') csvLink = artifact.links[k];
    }
    if (!csvLink) {
      if (callback) callback();
      return;
    }

    requestText(csvLink.href, function (error, text) {
      var rows = [];
      if (!error) {
        var csvRows = parseCsv(text);
        for (var m = 0; m < csvRows.length; m++) {
          rows.push({
            outcome_label: String(csvRows[m].outcome || '').split('_').join(' '),
            horizon: csvRows[m].horizon,
            beta: csvRows[m].beta,
            se: csvRows[m].se,
            lower95: csvRows[m].lower95,
            upper95: csvRows[m].upper95,
            p_value_normal: csvRows[m].p_value_normal,
            n: csvRows[m].n
          });
        }
      }
      bodyTarget.innerHTML = '<section class="table-shell reveal"><p style="margin-bottom:14px;">' + escapeHtml(artifact.caption) + '</p><table class="result-table"><thead><tr><th>Outcome</th><th>H</th><th>Beta</th><th>SE</th><th>95% lower</th><th>95% upper</th><th>p-value</th><th>N</th></tr></thead><tbody>' + renderArtifactTableRows(rows) + '</tbody></table></section>';
      if (callback) callback();
    });
  }

  function renderChart(domId, outcome, colorIndex) {
    var container = document.getElementById(domId);
    if (!container || !outcome || typeof Plotly === 'undefined') return;
    var palette = getThemePalette();
    var color = palette.colors[colorIndex % palette.colors.length];
    var x = [];
    var beta = [];
    var lower = [];
    var upper = [];
    var customdata = [];
    for (var i = 0; i < outcome.points.length; i++) {
      x.push(outcome.points[i].horizon);
      beta.push(outcome.points[i].beta);
      lower.push(outcome.points[i].lower95);
      upper.push(outcome.points[i].upper95);
      customdata.push([outcome.points[i].p_value, outcome.points[i].lower95, outcome.points[i].upper95]);
    }
    var traces = [
      { x: x, y: lower, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
      { x: x, y: upper, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: rgba(color, 0.14), hoverinfo: 'skip', showlegend: false },
      {
        x: x,
        y: beta,
        mode: 'lines+markers',
        line: { color: color, width: 2.3 },
        marker: { color: color, size: 6, line: { color: color, width: 1.4 } },
        customdata: customdata,
        hovertemplate: 'h=%{x}<br>%{y:.4f}<br>p=%{customdata[0]:.3f}<br>95% CI [%{customdata[1]:.4f}, %{customdata[2]:.4f}]<extra></extra>'
      }
    ];
    var layout = {
      autosize: true,
      margin: { t: 8, r: 10, b: 42, l: 50 },
      paper_bgcolor: palette.paper,
      plot_bgcolor: palette.plot,
      font: { color: palette.text, family: 'Inter, sans-serif', size: 13 },
      hovermode: 'x unified',
      hoverlabel: { bgcolor: palette.hoverBg, font: { color: palette.hoverFont } },
      xaxis: {
        title: { text: 'Horizon', font: { size: 11 } },
        tickmode: 'linear',
        dtick: 1,
        gridcolor: palette.grid,
        linecolor: palette.axis,
        zeroline: false
      },
      yaxis: {
        title: { text: 'Response', font: { size: 11 } },
        gridcolor: palette.grid,
        zeroline: true,
        zerolinecolor: palette.axis,
        zerolinewidth: 1.1
      },
      showlegend: false
    };
    Plotly.react(container, traces, layout, { displayModeBar: false, responsive: true });
  }

  function renderComparisonChart(domId, outcome) {
    var container = document.getElementById(domId);
    if (!container || !outcome || typeof Plotly === 'undefined') return;
    var palette = getThemePalette();
    var traces = [];
    for (var i = 0; i < outcome.lines.length; i++) {
      var line = outcome.lines[i];
      var color = palette.colors[i % palette.colors.length];
      var x = [];
      var beta = [];
      var lower = [];
      var upper = [];
      var customdata = [];
      for (var j = 0; j < line.points.length; j++) {
        x.push(line.points[j].horizon);
        beta.push(line.points[j].beta);
        lower.push(line.points[j].lower95);
        upper.push(line.points[j].upper95);
        customdata.push([line.points[j].p_value, line.points[j].lower95, line.points[j].upper95]);
      }
      traces.push({
        x: x,
        y: lower,
        mode: 'lines',
        line: { width: 0 },
        hoverinfo: 'skip',
        showlegend: false
      });
      traces.push({
        x: x,
        y: upper,
        mode: 'lines',
        line: { width: 0 },
        fill: 'tonexty',
        fillcolor: rgba(color, i === 0 ? 0.16 : 0.08),
        hoverinfo: 'skip',
        showlegend: false
      });
      traces.push({
        x: x,
        y: beta,
        mode: 'lines+markers',
        name: line.label,
        line: { color: color, width: i === 0 ? 2.8 : 2 },
        marker: { color: color, size: i === 0 ? 7 : 5 },
        customdata: customdata,
        hovertemplate: '%{fullData.name}<br>h=%{x}<br>%{y:.4f}<br>p=%{customdata[0]:.3f}<br>95% CI [%{customdata[1]:.4f}, %{customdata[2]:.4f}]<extra></extra>'
      });
    }
    var layout = {
      autosize: true,
      margin: { t: 8, r: 10, b: 42, l: 50 },
      paper_bgcolor: palette.paper,
      plot_bgcolor: palette.plot,
      font: { color: palette.text, family: 'Inter, sans-serif', size: 13 },
      hovermode: 'x unified',
      hoverlabel: { bgcolor: palette.hoverBg, font: { color: palette.hoverFont } },
      legend: { orientation: 'h', y: -0.28, x: 0, font: { size: 11 } },
      xaxis: {
        title: { text: 'Horizon', font: { size: 11 } },
        tickmode: 'linear',
        dtick: 1,
        gridcolor: palette.grid,
        linecolor: palette.axis,
        zeroline: false
      },
      yaxis: {
        title: { text: 'Response', font: { size: 11 } },
        gridcolor: palette.grid,
        zeroline: true,
        zerolinecolor: palette.axis,
        zerolinewidth: 1.1
      },
      showlegend: true
    };
    Plotly.react(container, traces, layout, { displayModeBar: false, responsive: true });
  }

  function renderRobustnessChart(domId, job, colorIndex) {
    var container = document.getElementById(domId);
    if (!container || !job || typeof Plotly === 'undefined') return;
    var palette = getThemePalette();
    var color = palette.colors[colorIndex % palette.colors.length];
    var x = [];
    var y = [];
    var text = [];
    var markerColors = [];
    for (var i = 0; i < job.ladder.length; i++) {
      x.push(job.ladder[i].label);
      y.push(job.ladder[i].avg_abs_beta);
      text.push('rows=' + job.ladder[i].rows_written);
      markerColors.push(i === job.recommended_index ? color : rgba(color, 0.45));
    }
    var traces = [
      {
        x: x,
        y: y,
        type: 'bar',
        marker: { color: markerColors },
        text: text,
        hovertemplate: '%{x}<br>|beta|=%{y:.4f}<br>%{text}<extra></extra>'
      }
    ];
    var layout = {
      autosize: true,
      margin: { t: 8, r: 10, b: 52, l: 58 },
      paper_bgcolor: palette.paper,
      plot_bgcolor: palette.plot,
      font: { color: palette.text, family: 'Inter, sans-serif', size: 13 },
      hoverlabel: { bgcolor: palette.hoverBg, font: { color: palette.hoverFont } },
      xaxis: {
        tickangle: -20,
        gridcolor: palette.grid,
        linecolor: palette.axis
      },
      yaxis: {
        title: { text: 'Average |response|' },
        gridcolor: palette.grid,
        zeroline: true,
        zerolinecolor: palette.axis,
        zerolinewidth: 1.1
      },
      showlegend: false
    };
    Plotly.react(container, traces, layout, { displayModeBar: false, responsive: true });
  }

  function renderAllCharts() {
    var i;
    var j;
    if (!SITE_DATA) return;
    if (PAGE === 'home') {
      for (i = 0; i < SITE_DATA.home.main_job_ids.length; i++) {
        var homeJob = SITE_DATA.jobs[SITE_DATA.home.main_job_ids[i]];
        var homeOutcomes = (homeJob && homeJob.outcomes) || [];
        for (j = 0; j < homeOutcomes.length; j++) {
          renderChart(homeOutcomes[j].chart_dom_id, homeOutcomes[j], j);
        }
      }
      var independentEvidence = SITE_DATA.home.independent_evidence;
      if (independentEvidence && independentEvidence.outcomes) {
        for (i = 0; i < independentEvidence.outcomes.length; i++) {
          renderChart(independentEvidence.outcomes[i].chart_dom_id, independentEvidence.outcomes[i], i);
        }
      }
      for (i = 0; i < SITE_DATA.sidecar.job_ids.length; i++) {
        var evidenceJob = SITE_DATA.jobs[SITE_DATA.sidecar.job_ids[i]];
        var evidenceOutcomes = (evidenceJob && evidenceJob.outcomes) || [];
        for (j = 0; j < evidenceOutcomes.length; j++) {
          renderChart(evidenceOutcomes[j].chart_dom_id, evidenceOutcomes[j], j);
        }
      }
      var treatmentComparisonsHome = SITE_DATA.sidecar.treatment_comparisons || [];
      for (i = 0; i < treatmentComparisonsHome.length; i++) {
        for (j = 0; j < treatmentComparisonsHome[i].outcomes.length; j++) {
          renderComparisonChart(treatmentComparisonsHome[i].outcomes[j].chart_dom_id, treatmentComparisonsHome[i].outcomes[j]);
        }
      }
      if (SITE_DATA.robustness && SITE_DATA.robustness.jobs) {
        for (i = 0; i < SITE_DATA.robustness.jobs.length; i++) {
          renderRobustnessChart(SITE_DATA.robustness.jobs[i].chart_dom_id, SITE_DATA.robustness.jobs[i], i);
        }
      }
    } else if (PAGE === 'sidecar') {
      for (i = 0; i < SITE_DATA.sidecar.job_ids.length; i++) {
        var sideJob = SITE_DATA.jobs[SITE_DATA.sidecar.job_ids[i]];
        var sideOutcomes = (sideJob && sideJob.outcomes) || [];
        for (j = 0; j < sideOutcomes.length; j++) {
          renderChart(sideOutcomes[j].chart_dom_id, sideOutcomes[j], j);
        }
      }
      var treatmentComparisons = SITE_DATA.sidecar.treatment_comparisons || [];
      for (i = 0; i < treatmentComparisons.length; i++) {
        for (j = 0; j < treatmentComparisons[i].outcomes.length; j++) {
          renderComparisonChart(treatmentComparisons[i].outcomes[j].chart_dom_id, treatmentComparisons[i].outcomes[j]);
        }
      }
      if (SITE_DATA.robustness && SITE_DATA.robustness.jobs) {
        for (i = 0; i < SITE_DATA.robustness.jobs.length; i++) {
          renderRobustnessChart(SITE_DATA.robustness.jobs[i].chart_dom_id, SITE_DATA.robustness.jobs[i], i);
        }
      }
    } else if (PAGE === 'artifact') {
      var artifact = SITE_DATA.artifacts[ARTIFACT_ID];
      if (!artifact || artifact.kind !== 'figure') return;
      var artifactJob = SITE_DATA.jobs[artifact.job_id];
      for (i = 0; i < artifactJob.outcomes.length; i++) {
        if ((artifact.outcome_ids || []).indexOf(artifactJob.outcomes[i].key) !== -1) {
          renderChart(artifact.artifact_id + '-' + artifactJob.outcomes[i].key, artifactJob.outcomes[i], i);
        }
      }
    }
  }

  function renderPage(callback) {
    if (!SITE_DATA) {
      if (callback) callback();
      return;
    }
    if (PAGE === 'home') {
      renderMetrics();
      renderInsightCards(SITE_DATA.home.insights, 'questions-grid');
      renderJobList(SITE_DATA.home.main_job_ids, 'headline-job-list');
      renderDepositAccounting();
      renderInsightCards(SITE_DATA.sidecar.insights, 'additional-evidence-grid');
      renderJobList(SITE_DATA.sidecar.job_ids, 'job-list-sidecar');
      renderTreatmentComparisons();
      renderIvLabSummary();
      renderRobustnessSummary();
      renderArtifacts(SITE_DATA.home.main_artifacts, SITE_DATA.home.appendix_artifacts);
      renderDeferred(SITE_DATA.deferred_jobs);
      if (callback) callback();
    } else if (PAGE === 'sidecar') {
      renderMetrics();
      renderInsightCards(SITE_DATA.sidecar.insights, 'additional-evidence-grid');
      renderJobList(SITE_DATA.sidecar.job_ids, 'job-list-sidecar');
      renderTreatmentComparisons();
      renderIvLabSummary();
      renderRobustnessSummary();
      if (callback) callback();
    } else if (PAGE === 'gallery') {
      renderGallery(SITE_DATA.artifact_gallery);
      if (callback) callback();
    } else if (PAGE === 'artifact') {
      renderArtifactDetail(SITE_DATA.artifacts[ARTIFACT_ID], callback);
    } else if (callback) {
      callback();
    }
  }

  function afterRender() {
    revealAll();
    renderAllCharts();
    scheduleTypeset();
  }

  document.addEventListener('DOMContentLoaded', function () {
    initPageConfig();
    if (window.eaTdcTheme && window.eaTdcTheme.initToggle) {
      window.eaTdcTheme.initToggle();
    }
    loadSiteData(function (error) {
      if (error) {
        console.error(error);
        return;
      }
      renderPage(function () {
        afterRender();
      });
    });
    window.addEventListener('resize', function () { renderAllCharts(); });
    window.addEventListener('ea-tdc-themechange', function () { renderAllCharts(); });
    document.addEventListener('toggle', function (event) {
      var target = event.target;
      if (target && target.tagName === 'DETAILS' && target.open) {
        scheduleTypeset([target]);
      }
    });
  });
})();