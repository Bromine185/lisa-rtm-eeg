// Page logic: tabs, controls, every chart and table. Data only from dash.data(...).
(function () {
  const EX = window.EX;
  const f2 = d3.format('.2f'), f3 = d3.format('.3f'), f1 = d3.format('.1f'), pct = d3.format('.0%'), int = d3.format(',');
  const C = () => dash.colors;
  const enc = (v) => String(v).replace(/%/g, '%25').replace(/&/g, '%26').replace(/=/g, '%3D');
  const short = (s) => s.replace('Zürich (Prof. ', 'Zürich ').replace(')', '').replace('NorthDakota', 'North Dakota');
  const num = (v, f = f2) => (v == null || !isFinite(v) ? '—' : f(v));

  dash.calc('totals', {
    title: 'Headline counts',
    description: 'People recorded and analysed, labs, and trials of the analysed people (sum over labs and set sizes).',
    inputs: ['subjects', 'lab_setsize'],
    fn: (subjects, lab_setsize) => [
      { metric: 'participants', value: subjects.length },
      { metric: 'analysed', value: subjects.filter((s) => !s.excluded).length },
      { metric: 'labs', value: new Set(subjects.map((s) => s.lab)).size },
      { metric: 'trials', value: lab_setsize.reduce((a, r) => a + r.n, 0) },
    ],
  });
  dash.calc('lab_psd', {
    title: 'Lab median spectrum',
    description: 'For each lab and 1-Hz bin, the median over its people of their log10 power.',
    inputs: ['psd_subjects', 'subjects'],
    fn: (psd_subjects, subjects) => {
      const lab = Object.fromEntries(subjects.map((s) => [s.sub, s.lab]));
      const by = {};
      psd_subjects.forEach((r) => { (by[lab[r.sub]] = by[lab[r.sub]] || []).push(r); });
      const out = [];
      Object.entries(by).forEach(([l, rs]) => {
        const nb = rs[0].log10_power.length;
        for (let i = 0; i < nb; i++) {
          const v = rs.map((r) => r.log10_power[i]).filter((x) => x != null).sort((a, b) => a - b);
          if (v.length) out.push({ lab: l, f_hz: rs[0].f0_hz + i * rs[0].df_hz, median_log10_power: v[Math.floor(v.length / 2)] });
        }
      });
      return out;
    },
  });

  const state = { tab: 'overview', behLab: null, cdaLab: 'All labs', spLab: null, subLab: '', subQ: '', subInc: 'all', sel: null, sort: {} };
  const ok = (...ids) => ids.every((id) => dash.data(id).status === 'ok');
  const rows = (id) => dash.data(id).data || [];
  const failIf = (el, ...ids) => {
    const bad = ids.find((id) => ['error', 'connect', 'declined', 'missing'].includes(dash.data(id).status));
    if (bad) { EX.fail(el); el.classList.remove('dash-skeleton'); return true; }
    if (!ok(...ids)) { el.replaceChildren(); el.classList.add('dash-skeleton'); return true; }
    el.classList.remove('dash-skeleton'); return false;
  };

  // ---------- tabs
  const tabs = [...document.querySelectorAll('.ex-tabs [data-tab]')];
  const show = (id) => {
    state.tab = id;
    tabs.forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === id)));
    document.querySelectorAll('.ex-page').forEach((p) => { p.hidden = p.id !== id; });
    dash.setLink(id); draw();
  };
  tabs.forEach((b) => b.addEventListener('click', () => show(b.dataset.tab)));

  // ---------- controls
  const sel = (id, key, withAll) => {
    const el = document.getElementById(id);
    el.addEventListener('change', () => { state[key] = el.value || null; draw(); });
    return () => {
      if (!ok('labs')) return;
      const labs = rows('labs').map((r) => r.lab);
      const opts = (withAll != null ? [withAll] : []).concat(labs);
      if (el.options.length !== opts.length) el.replaceChildren(...opts.map((l) => new Option(l === '' ? 'All labs' : l, l)));
      if (state[key] == null) state[key] = withAll != null ? withAll : labs[0];
      el.value = state[key];
    };
  };
  const ctrls = [sel('beh-lab', 'behLab'), sel('cda-lab', 'cdaLab', 'All labs'), sel('sp-lab', 'spLab'), sel('sub-lab', 'subLab', '')];
  document.getElementById('sub-q').addEventListener('input', (e) => { state.subQ = e.target.value.trim(); draw(); });
  document.getElementById('sub-inc').addEventListener('change', (e) => { state.subInc = e.target.value; draw(); });

  // ---------- tables
  function fillTable(table, data, fmts, onRow, selKey) {
    const ths = [...table.querySelectorAll('th[data-field]')];
    const st = state.sort[table.id];
    let d = data.slice();
    if (st) d.sort((a, b) => { const x = a[st.f], y = b[st.f]; const r = (x == null) - (y == null) || (typeof x === 'number' ? x - y : String(x).localeCompare(String(y))); return st.dir * r; });
    ths.forEach((th) => th.setAttribute('aria-sort', st && st.f === th.dataset.field ? (st.dir > 0 ? 'ascending' : 'descending') : 'none'));
    const key = table.dataset.rowKey;
    const body = table.tBodies[0];
    body.replaceChildren(...d.map((r) => {
      const tr = document.createElement('tr'); tr.dataset.key = r[key];
      if (selKey != null && r[key] === selKey) tr.className = 'sel';
      ths.forEach((th) => {
        const td = document.createElement('td'); const f = th.dataset.field; const v = r[f];
        td.textContent = fmts[f] ? fmts[f](v) : v == null ? '—' : String(v);
        if (th.classList.contains('num')) td.className = 'num';
        if (f === 'note' || f === 'test' || f === 'reason') td.classList.add('wrap');
        tr.appendChild(td);
      });
      if (onRow) tr.addEventListener('click', () => onRow(r));
      return tr;
    }));
  }
  document.querySelectorAll('.ex-table').forEach((t) => t.querySelectorAll('th[data-field]').forEach((th) => th.addEventListener('click', () => {
    const cur = state.sort[t.id]; const f = th.dataset.field;
    state.sort[t.id] = { f, dir: cur && cur.f === f ? -cur.dir : 1 }; draw();
  })));

  // ---------- pages
  function drawHeader() {
    const tk = document.getElementById('takeaway');
    if (!ok('ideas', 'labs')) { tk.classList.add('dash-skeleton'); return; }
    tk.classList.remove('dash-skeleton');
    const I = rows('ideas');
    const g = (t) => I.find((r) => r.test === t);
    const span = (t, f) => { const s = document.createElement('b'); const r = g(t); s.dataset.source = 'ideas'; s.dataset.field = 'value'; s.dataset.where = 'test=' + enc(t); s.textContent = r ? f(r.value) : '—'; return s; };
    tk.replaceChildren(
      'Across the ten labs, how reliably a lab’s people get the easy 2-item trials right tracks the lab’s capacity (r = ',
      span('Lab-mean attention vs lab-mean K', f2), '). A single trial’s EEG adds ',
      span('Brier skill of trial EEG + eyes over the person’s own accuracy'.replace('’', "'"), f2), ' % forecast skill on whether it is answered correctly.');
  }
  function drawTiles() {
    document.querySelectorAll('[data-source="totals"]').forEach((el) => {
      const t = dash.data('totals');
      if (t.status !== 'ok') return;
      el.classList.remove('dash-skeleton');
      const r = (t.data || []).find((x) => x.metric === el.dataset.where.split('=')[1]);
      el.textContent = r ? int(r.value) : '—';
    });
  }
  const ssColor = (n) => C()[[2, 4, 6].indexOf(n)];

  function overview() {
    const el = document.getElementById('ov-cda');
    if (!failIf(el, 'cda_lab_waves')) {
      const R = rows('cda_lab_waves').filter((r) => r.lab === 'All labs');
      EX.line(el, { height: 300, zero: true, shade: [300, 900], vlines: [0], xLabel: 'ms from memory array', yLabel: 'µV',
        series: [2, 4, 6].map((n) => { const q = R.filter((r) => r.setsize === n); return { name: `${n} items`, color: ssColor(n), pts: q.map((r) => [r.t_ms, r.uv]), band: q.map((r) => [r.t_ms, r.uv - r.se, r.uv + r.se]) }; }),
        xTip: (v) => `${v} ms`, tfmt: (v) => `${f2(v)} µV` });
      EX.legend(el, [2, 4, 6].map((n) => ({ name: `${n} items`, color: ssColor(n) })));
    }
    const e2 = document.getElementById('ov-eng');
    if (!failIf(e2, 'labs')) {
      EX.scatter(e2, { xLabel: 'Attention (share of 2-item trials encoded)', yLabel: 'K', xfmt: pct, yfmt: f1,
        points: rows('labs').map((r) => ({ x: r.attention, y: r.K, label: short(r.lab), color: C()[0], tip: [r.lab, `attention ${pct(r.attention)}`, `K ${f2(r.K)}`, `${r.included} analysed`] })) });
    }
    const dl = document.getElementById('ov-ideas');
    if (ok('ideas')) {
      const pick = ['Lab-mean attention vs lab-mean K', 'CDA ~ attention, within lab (std. beta)', "Brier skill of trial EEG + eyes over the person's own accuracy",
        'CDA after a 6-item trial (vs after 2)', 'RT variance before lapses', 'Eye-tracker load slope vs K within lab', 'cda: disengaged minus engaged trials', 'exponent: r with K within lab'];
      dl.replaceChildren(...pick.map((t) => rows('ideas').find((r) => r.test === t)).filter(Boolean).map((r) => {
        const d = document.createElement('div'); const dt = document.createElement('dt'); const dd = document.createElement('dd');
        dt.textContent = `${r.idea}: ${r.test}`; dd.textContent = `${num(r.value, f3)}${r.unit && r.unit !== 'r' && r.unit !== 'beta' ? ' ' + r.unit : ''}${r.t != null ? `  (t ${f2(r.t)})` : ''}`;
        dd.dataset.source = 'ideas'; dd.dataset.field = 'value'; dd.dataset.where = 'test=' + enc(r.test);
        d.append(dt, dd); return d;
      }));
    }
  }

  function labsPage() {
    const t = document.getElementById('labs-table');
    if (ok('labs')) fillTable(t, rows('labs'), { age: (v) => num(v, f1), acc: (v) => num(v, pct), attention: (v) => num(v, pct), K: (v) => num(v), cda2: (v) => num(v), cda4: (v) => num(v), cda6: (v) => num(v) });
    const n = document.getElementById('labs-n');
    if (!failIf(n, 'labs')) EX.hbars(n, { fmt: int, bars: rows('labs').map((r) => ({ label: `${short(r.lab)} (of ${r.n})`, value: r.included, color: C()[0] })) });
    const s = document.getElementById('labs-cdak');
    if (!failIf(s, 'labs')) EX.scatter(s, { xLabel: 'K', yLabel: 'CDA at 4 items (µV)', xfmt: f1, yfmt: f1,
      points: rows('labs').map((r) => ({ x: r.K, y: r.cda4, label: short(r.lab), color: C()[1], tip: [r.lab, `K ${f2(r.K)}`, `CDA 4 ${f2(r.cda4)} µV`] })) });
  }

  function perLabLines(el, id, field, hl, o) {
    if (failIf(el, id)) return;
    const R = rows(id).filter((r) => (o.filter ? o.filter(r) : true));
    const labs = [...new Set(R.map((r) => r.lab))];
    const series = labs.map((l) => {
      const q = R.filter((r) => r.lab === l).sort((a, b) => a[o.x] - b[o.x]);
      const color = l === hl ? C()[0] : l === 'All labs' ? C()[1] : EX.MUTED;
      return { name: short(l), color, width: l === hl || l === 'All labs' ? 2.5 : 1.2, dots: l === hl, pts: q.map((r) => [r[o.x], r[field]]) };
    });
    EX.line(el, { ...o, series });
    EX.legend(el, [...(hl ? [{ name: short(hl), color: C()[0] }] : []), ...(labs.includes('All labs') ? [{ name: 'All labs', color: C()[1] }] : []), { name: 'Other labs', color: EX.MUTED }]);
  }

  function behaviour() {
    const hl = state.behLab;
    perLabLines(document.getElementById('beh-acc'), 'lab_setsize', 'acc', hl, { x: 'setsize', xTicks: [2, 4, 6], xLabel: 'Items to remember', yLabel: 'Accuracy', yfmt: pct, tfmt: (v) => pct(v), xTip: (v) => `${v} items` });
    perLabLines(document.getElementById('beh-rt'), 'lab_setsize', 'rt', hl, { x: 'setsize', xTicks: [2, 4, 6], xLabel: 'Items to remember', yLabel: 'Seconds', yfmt: f2, xTip: (v) => `${v} items` });
    perLabLines(document.getElementById('beh-tot'), 'time_on_task', 'acc', hl, { x: 'minute5', xLabel: 'Minutes into the session', yLabel: 'Accuracy', yfmt: pct, tfmt: (v) => pct(v), xTip: (v) => `${v}–${v + 5} min` });
    perLabLines(document.getElementById('beh-alpha'), 'time_on_task', 'alpha_pre', hl, { x: 'minute5', xLabel: 'Minutes into the session', yLabel: 'Log alpha (relative)', zero: true, yfmt: f2, xTip: (v) => `${v}–${v + 5} min` });
    perLabLines(document.getElementById('beh-eye'), 'lab_setsize', 'heog', hl, { x: 'setsize', xTicks: [2, 4, 6], xLabel: 'Items to remember', yLabel: 'µV toward cue', zero: true, yfmt: f1, xTip: (v) => `${v} items` });
    const k = document.getElementById('beh-k');
    if (!failIf(k, 'subjects', 'labs')) {
      EX.scatter(k, { height: 320, xCat: rows('labs').map((r) => r.lab), yLabel: 'K', yfmt: f1,
        onClick: (p) => { state.sel = p.sub; show('subjects'); },
        points: rows('subjects').filter((s) => !s.excluded).map((s) => ({ x: s.lab, y: s.K, sub: s.sub, color: s.lab === hl ? C()[0] : EX.MUTED, r: 3.5, tip: [s.sub, s.lab, `K ${f2(s.K)}`, `attention ${pct(s.attention)}`, 'Click to open'] })) });
    }
  }

  function cdaPage() {
    const w = document.getElementById('cda-wave');
    if (!failIf(w, 'cda_lab_waves')) {
      const R = rows('cda_lab_waves').filter((r) => r.lab === state.cdaLab);
      EX.line(w, { height: 320, zero: true, shade: [300, 900], vlines: [0], xLabel: 'ms from memory array', yLabel: 'µV',
        series: [2, 4, 6].map((n) => { const q = R.filter((r) => r.setsize === n); return { name: `${n} items`, color: ssColor(n), pts: q.map((r) => [r.t_ms, r.uv]), band: q.map((r) => [r.t_ms, r.uv - r.se, r.uv + r.se]) }; }),
        xTip: (v) => `${v} ms`, tfmt: (v) => `${f2(v)} µV` });
      EX.legend(w, [2, 4, 6].map((n) => ({ name: `${n} items`, color: ssColor(n) })));
    }
    const h = document.getElementById('cda-hist');
    if (!failIf(h, 'cda_hist')) {
      const R = rows('cda_hist');
      EX.line(h, { xLabel: 'Single-trial CDA (µV)', yLabel: 'Trials', yfmt: d3.format('~s'), tfmt: int, xTip: (v) => `${v} µV`, vlines: [0],
        series: [2, 4, 6].map((n) => ({ name: `${n} items`, color: ssColor(n), pts: R.filter((r) => r.setsize === n).map((r) => [(r.lo + r.hi) / 2, r.count]) })) });
      EX.legend(h, [2, 4, 6].map((n) => ({ name: `${n} items`, color: ssColor(n) })));
    }
    const c = document.getElementById('cda-coh');
    if (!failIf(c, 'coherence_spectrum')) EX.line(c, { xLabel: 'Hz', yLabel: 'Coherent fraction', yfmt: d3.format('.0%'), tfmt: d3.format('.2%'), xTip: (v) => `${f1(v)} Hz`,
      series: [{ name: 'Coherent fraction', color: C()[0], dots: true, pts: rows('coherence_spectrum').map((r) => [r.f, r.c]) }] });
    const hl = state.cdaLab === 'All labs' ? null : state.cdaLab;
    perLabLines(document.getElementById('cda-labss'), 'lab_setsize', 'cda', hl || '', { x: 'setsize', xTicks: [2, 4, 6], xLabel: 'Items to remember', yLabel: 'µV', zero: true, yfmt: f1, xTip: (v) => `${v} items` });
  }

  const SUBF = [['lab', 'Lab'], ['excluded', 'Excluded'], ['reason', 'Why excluded'], ['maker', 'System'], ['fs', 'Rate (Hz)'], ['n_eeg', 'EEG channels'], ['heog', 'Eye channel'],
    ['n_trials', 'Trials'], ['kept', 'Clean trials'], ['acc', 'Accuracy'], ['K2', 'K at 2'], ['K4', 'K at 4'], ['K6', 'K at 6'], ['K', 'K (4, 6)'], ['attention', 'Attention'],
    ['cda2', 'CDA 2 (µV)'], ['cda4', 'CDA 4 (µV)'], ['cda6', 'CDA 6 (µV)'], ['cda4_po', 'CDA 4, PO7/PO8'], ['heog_resid', 'Residual eye drift (µV)'], ['c_time', 'Coherent fraction'],
    ['c_slow', 'Coherent fraction <2 Hz'], ['c_alpha_wave', 'Alpha waveform c'], ['c_alpha_env', 'Alpha envelope c'], ['alpha_lat', 'Alpha lateralization'],
    ['hyst_cda', 'CDA carry-over after 6'], ['hyst_rt', 'RT carry-over after 6 (s)'], ['frac_dis', 'Share of trials disengaged'], ['dwell_dis', 'Disengaged run (trials)'],
    ['exponent', '1/f exponent'], ['offset', '1/f offset'], ['iaf', 'Alpha frequency (Hz)'], ['dfa_alpha', 'Alpha DFA exponent'], ['hf_floor', 'High-frequency floor']];

  function subjectsPage() {
    if (!ok('subjects')) return;
    const S = rows('subjects');
    if (!state.sel) state.sel = (S.find((s) => !s.excluded) || S[0]).sub;
    const q = state.subQ.toLowerCase();
    const list = S.filter((s) => (!state.subLab || s.lab === state.subLab) && (!q || s.sub.toLowerCase().includes(q)) &&
      (state.subInc === 'all' || (state.subInc === 'in' ? !s.excluded : s.excluded)));
    const pf = (v) => num(v, pct);
    fillTable(document.getElementById('sub-table'), list, { excluded: (v) => (v ? 'yes' : 'no'), reason: (v) => v || '', kept: pf, acc: pf, attention: pf, K: (v) => num(v),
      cda2: (v) => num(v), cda4: (v) => num(v), cda6: (v) => num(v), alpha_lat: (v) => num(v, f3), c_time: (v) => num(v, d3.format('.1%')), exponent: (v) => num(v), iaf: (v) => num(v, f1) },
      (r) => { state.sel = r.sub; draw(); document.getElementById('sub-dossier-card').scrollIntoView({ block: 'nearest' }); }, state.sel);
    const s = S.find((x) => x.sub === state.sel);
    document.getElementById('sub-name').textContent = s ? `${s.sub} · ${s.lab}` : 'Person';
    const dl = document.getElementById('sub-dl');
    dl.replaceChildren(...SUBF.filter(([f]) => s && f in s).map(([f, lab]) => {
      const d = document.createElement('div'); const dt = document.createElement('dt'); const dd = document.createElement('dd');
      const v = s[f]; dt.textContent = lab;
      dd.textContent = typeof v === 'boolean' ? (v ? 'yes' : 'no') : typeof v === 'number' ? (['fs', 'n_eeg', 'n_trials'].includes(f) ? int(v) : ['kept', 'acc', 'attention', 'frac_dis'].includes(f) ? pct(v) : f3(v)) : v || '—';
      dd.dataset.source = 'subjects'; dd.dataset.field = f; dd.dataset.where = 'sub=' + enc(s.sub);
      d.append(dt, dd); return d;
    }));
    const w = document.getElementById('sub-wave');
    if (!failIf(w, 'cda_subject_waves')) {
      const R = rows('cda_subject_waves').filter((r) => r.sub === state.sel);
      EX.line(w, { height: 260, zero: true, shade: [300, 900], vlines: [0], xLabel: 'ms from memory array', yLabel: 'µV', xTip: (v) => `${v} ms`, tfmt: (v) => `${f2(v)} µV`,
        series: [2, 4, 6].map((n) => { const r = R.find((x) => x.setsize === n); return { name: `${n} items`, color: ssColor(n), pts: r ? r.w.map((v, i) => [r.t0_ms + i * r.dt_ms, v]) : [] }; }) });
      EX.legend(w, [2, 4, 6].map((n) => ({ name: `${n} items`, color: ssColor(n) })));
    }
    const p = document.getElementById('sub-psd');
    if (!failIf(p, 'psd_subjects', 'lab_psd')) {
      const r = rows('psd_subjects').find((x) => x.sub === state.sel);
      const L = rows('lab_psd').filter((x) => s && x.lab === s.lab);
      EX.line(p, { height: 260, xLabel: 'Hz (as recorded)', yLabel: 'log10 power', yfmt: f1, xTip: (v) => `${v} Hz`,
        series: [{ name: 'Lab median', color: C()[1], width: 1.5, pts: L.map((x) => [x.f_hz, x.median_log10_power]) },
          { name: 'This person', color: C()[0], pts: r ? r.log10_power.map((v, i) => [r.f0_hz + i * r.df_hz, v]) : [] }] });
      EX.legend(p, [{ name: 'This person', color: C()[0] }, { name: 'Lab median', color: C()[1] }]);
    }
  }

  function spectraPage() {
    const hl = state.spLab;
    perLabLines(document.getElementById('sp-psd'), 'lab_psd', 'median_log10_power', hl, { x: 'f_hz', height: 320, xLabel: 'Hz', yLabel: 'log10 power', yfmt: f1, xTip: (v) => `${v} Hz` });
    const e = document.getElementById('sp-exp');
    if (!failIf(e, 'subjects')) EX.scatter(e, { xLabel: '1/f exponent', yLabel: 'K', xfmt: f1, yfmt: f1, onClick: (p) => { state.sel = p.sub; show('subjects'); },
      points: rows('subjects').filter((s) => !s.excluded).map((s) => ({ x: s.exponent, y: s.K, sub: s.sub, r: 3.5, color: s.lab === hl ? C()[0] : EX.MUTED, tip: [s.sub, s.lab, `exponent ${f2(s.exponent)}`, `K ${f2(s.K)}`] })) });
    const i = document.getElementById('sp-iaf');
    if (!failIf(i, 'subjects', 'labs')) EX.scatter(i, { xCat: rows('labs').map((r) => r.lab), yLabel: 'Hz', yfmt: f1, onClick: (p) => { state.sel = p.sub; show('subjects'); },
      points: rows('subjects').filter((s) => !s.excluded).map((s) => ({ x: s.lab, y: s.iaf, sub: s.sub, r: 3.5, color: s.lab === hl ? C()[0] : EX.MUTED, tip: [s.sub, s.lab, `alpha ${f1(s.iaf)} Hz`] })) });
  }

  function analysesPage() {
    const pr = document.getElementById('an-pred');
    if (!failIf(pr, 'predict')) {
      const M = [['crpss_cda', 'CDA', 0], ['crpss_cda_labc', 'CDA centred in lab', 1], ['crpss_lab_clim', 'Lab only', 2]];
      EX.hbars(pr, { fmt: d3.format('+.1%'), tickFmt: d3.format('.0%'), bars: rows('predict').flatMap((r) => M.map(([k, n, i]) => ({ label: `${r.feature} · ${n}`, value: r[k], color: C()[i] }))) });
    }
    const bands = ['1-4', '4-8', '8-13', '13-30', '30-60', '60-125'];
    const meth = [['spline', 'Spherical spline', 2], ['wiener', 'Wiener (averaging)', 0], ['sampler', 'Gaussian draw', 1]];
    ['c', 'p'].forEach((k) => {
      const el = document.getElementById(k === 'c' ? 'an-spat' : 'an-spatp');
      if (failIf(el, 'spatial')) return;
      EX.line(el, { xTicks: bands.map((_, i) => i), xfmt: (i) => bands[i], xTip: (i) => `${bands[i]} Hz`, xLabel: 'Band (Hz)', yLabel: k === 'c' ? 'Coherent fraction' : 'Output / true energy', yDomain: [0, 1.05], yfmt: f1, tfmt: f2,
        series: meth.map(([m, n, i]) => ({ name: n, color: C()[i], dots: true, pts: bands.map((b, j) => { const r = rows('spatial').find((x) => x.method === m && x.band === b); return [j, r ? r[k] : null]; }) })) });
      EX.legend(el, meth.map(([, n, i]) => ({ name: n, color: C()[i] })));
    });
    const b = document.getElementById('an-bwe');
    if (!failIf(b, 'bwe')) {
      const R = rows('bwe'); const keys = Object.keys(R[0]).filter((k) => k.startsWith('def_'));
      const arms = [['det_paper | pt', 'L1 loss (LISA)', 3], ['det | pt', 'L1 + spectral loss', 0], ['es_marg | pt', 'Energy score, one draw', 1], ['shaped_noise', 'Unconditional noise', 2]];
      EX.line(b, { xLog: true, xTicks: keys.map((k) => +k.slice(4)), xLabel: 'Band centre (Hz)', yLabel: 'dB', zero: true, yfmt: d3.format('~g'), tfmt: (v) => `${f1(v)} dB`, xTip: (v) => `${v} Hz`,
        series: arms.map(([c, n, i]) => { const r = R.find((x) => x.condition === c); return { name: n, color: C()[i], dots: true, pts: r ? keys.map((k) => [+k.slice(4), r[k]]) : [] }; }) });
      EX.legend(b, arms.map(([, n, i]) => ({ name: n, color: C()[i] })));
      fillTable(document.getElementById('bwe-table'), R, { deficit: (v) => num(v, f1), coh_above_40: (v) => num(v, d3.format('.3f')), snr: (v) => num(v, f1), crps: (v) => num(v, f3), loud: (v) => num(v, f1), quiet: (v) => num(v, f1) });
    }
  }

  function ideasPage() {
    if (!ok('ideas')) return;
    fillTable(document.getElementById('ideas-table'), rows('ideas'), { value: (v) => num(v, d3.format('.4~f')), t: (v) => num(v), p: (v) => (v == null ? '—' : d3.format('.2~g')(v)) });
  }

  function draw() {
    ctrls.forEach((c) => c());
    drawHeader(); drawTiles();
    ({ overview, labs: labsPage, behaviour, cda: cdaPage, subjects: subjectsPage, spectra: spectraPage, analyses: analysesPage, ideas: ideasPage })[state.tab]();
  }
  dash.onData(draw);
  const named = location.hash.slice(1);
  if (tabs.some((b) => b.dataset.tab === named)) show(named);
})();
