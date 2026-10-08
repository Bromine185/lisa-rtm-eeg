// Shared chart helpers: line, scatter/strip, horizontal bars. Everything sized from clientWidth, colors from tokens.
(function () {
  const EX = (window.EX = window.EX || {});
  const MUTED = 'var(--cds-chart-muted)';
  EX.MUTED = MUTED;

  function tipFor(el) {
    const card = el.closest('.ex-card') || el.parentElement;
    let t = card.querySelector(':scope > .ex-tip');
    if (!t) { t = document.createElement('div'); t.className = 'ex-tip'; t.hidden = true; card.appendChild(t); }
    return t;
  }
  function showTip(el, tip, event, lines) {
    tip.replaceChildren();
    lines.forEach((ln, i) => {
      const d = document.createElement('div');
      if (i === 0) { d.className = 'h'; d.textContent = ln.text; }
      else {
        if (ln.color) { const sw = document.createElement('i'); sw.style.background = ln.color; d.appendChild(sw); }
        d.appendChild(document.createTextNode(ln.text));
      }
      tip.appendChild(d);
    });
    tip.hidden = false;
    const card = tip.parentElement.getBoundingClientRect();
    let x = event.clientX - card.left + 12, y = event.clientY - card.top + 12;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > card.width - 4) x = event.clientX - card.left - w - 12;
    if (x < 4) x = 4;
    if (y + h > card.height - 4) y = event.clientY - card.top - h - 12;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  EX.fail = (el, msg) => { el.replaceChildren(); const d = document.createElement('div'); d.className = 'fail'; d.textContent = msg || 'Failed to load'; el.appendChild(d); };

  function frame(el, height, yTicks, yfmt, xLabel, yLabel) {
    el.replaceChildren();
    const W = Math.max(280, el.clientWidth || 600), H = height;
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`).style('display', 'block');
    const probe = svg.append('text').style('font-size', 'var(--cds-font-size-caption)');
    let wmax = 0; yTicks.forEach((t) => { probe.text(yfmt(t)); wmax = Math.max(wmax, probe.node().getComputedTextLength()); });
    probe.remove();
    const m = { l: Math.ceil(wmax) + (yLabel ? 26 : 10), r: 14, t: 10, b: xLabel ? 40 : 24 };
    return { svg, W, H, m };
  }
  function axes(g, f, x, y, opts) {
    const { svg, W, H, m } = f;
    const yt = opts.yTicks, xt = opts.xTicks;
    svg.append('g').selectAll('line').data(yt).join('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y).attr('y2', y)
      .style('stroke', 'var(--cds-chart-grid)').attr('stroke-width', 1);
    svg.append('g').selectAll('text').data(yt).join('text').attr('x', m.l - 6).attr('y', (d) => y(d) + 4).attr('text-anchor', 'end').text(opts.yfmt);
    svg.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', H - m.b).attr('y2', H - m.b).style('stroke', 'var(--cds-chart-axis)');
    if (xt) svg.append('g').selectAll('text').data(xt).join('text').attr('x', x).attr('y', H - m.b + 15).attr('text-anchor', 'middle').text(opts.xfmt);
    if (opts.xLabel) svg.append('text').attr('x', m.l + (W - m.l - m.r) / 2).attr('y', H - 4).attr('text-anchor', 'middle').text(opts.xLabel);
    if (opts.yLabel) svg.append('text').attr('transform', `translate(11,${m.t + (H - m.t - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').text(opts.yLabel);
  }

  // series: [{name, color, pts: [[x, y], ...], band?: [[x, lo, hi]], width?, dash?, hover?: true}]
  EX.line = function (el, o) {
    const all = o.series.flatMap((s) => s.pts.filter((p) => p[1] != null && isFinite(p[1])));
    if (!all.length) return EX.fail(el, 'No data for this selection');
    const ys = all.map((p) => p[1]).concat((o.series.flatMap((s) => s.band || [])).flatMap((b) => [b[1], b[2]]));
    let yd = o.yDomain || d3.extent(ys); if (o.zero) yd = [Math.min(0, yd[0]), Math.max(0, yd[1])];
    const y0 = d3.scaleLinear(yd, [0, 1]).nice(5);
    const yTicks = y0.ticks(5), yfmt = o.yfmt || d3.format('~g');
    const f = frame(el, o.height || 260, yTicks, yfmt, o.xLabel, o.yLabel);
    const { svg, W, H, m } = f;
    const xd = o.xDomain || d3.extent(all, (p) => p[0]);
    const x = (o.xLog ? d3.scaleLog : d3.scaleLinear)(xd, [m.l, W - m.r]);
    const y = d3.scaleLinear(y0.domain(), [H - m.b, m.t]);
    const xTicks = o.xTicks || x.ticks(Math.max(3, Math.floor((W - m.l) / 80)));
    axes(svg, f, x, y, { yTicks, xTicks, yfmt, xfmt: o.xfmt || d3.format('~g'), xLabel: o.xLabel, yLabel: o.yLabel });
    if (o.shade) svg.insert('rect', ':first-child').attr('x', x(o.shade[0])).attr('width', x(o.shade[1]) - x(o.shade[0])).attr('y', m.t).attr('height', H - m.b - m.t).style('fill', 'var(--cds-chart-grid)').style('opacity', 0.5);
    if (o.zero && yd[0] < 0) svg.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(0)).attr('y2', y(0)).style('stroke', 'var(--cds-chart-reference)');
    (o.vlines || []).forEach((v) => svg.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', m.t).attr('y2', H - m.b).style('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '3 3'));
    const order = o.series.slice().sort((a, b) => (a.color === MUTED) - (b.color === MUTED)).reverse();
    order.forEach((s) => {
      if (s.band) svg.append('path').datum(s.band.filter((b) => b[1] != null)).attr('d', d3.area().x((b) => x(b[0])).y0((b) => y(b[1])).y1((b) => y(b[2]))).style('fill', s.color).style('opacity', 0.15);
      const ln = d3.line().defined((p) => p[1] != null && isFinite(p[1])).x((p) => x(p[0])).y((p) => y(p[1]));
      svg.append('path').datum(s.pts).attr('d', ln).style('fill', 'none').style('stroke', s.color).attr('stroke-width', s.width || 2)
        .attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round').attr('stroke-dasharray', s.dash || null);
      if (s.dots) svg.append('g').selectAll('circle').data(s.pts.filter((p) => p[1] != null)).join('circle').attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1])).attr('r', 3.5).style('fill', s.color);
    });
    const hov = o.series.filter((s) => s.hover !== false && s.color !== MUTED);
    const rule = svg.append('line').attr('y1', m.t).attr('y2', H - m.b).style('stroke', 'var(--cds-chart-axis)').style('opacity', 0);
    const tip = tipFor(el);
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', W - m.l - m.r).attr('height', H - m.t - m.b).style('fill', 'transparent')
      .on('mousemove', (ev) => {
        const [mx] = d3.pointer(ev); const xv = x.invert(mx);
        const lines = [];
        let xs = null;
        hov.forEach((s) => {
          const p = s.pts.reduce((a, b) => (Math.abs(b[0] - xv) < Math.abs(a[0] - xv) ? b : a));
          xs = p[0]; if (p[1] != null) lines.push({ color: hov.length > 1 ? s.color : null, text: `${s.name}: ${(o.tfmt || yfmt)(p[1])}` });
        });
        if (xs == null) return;
        rule.attr('x1', x(xs)).attr('x2', x(xs)).style('opacity', 1);
        showTip(el, tip, ev, [{ text: (o.xTip || ((v) => v))(xs) }, ...lines]);
      })
      .on('mouseleave', () => { tip.hidden = true; rule.style('opacity', 0); });
  };

  // points: [{x, y, label?, color, r?, tip: [lines]}]; xCat: categories for a strip chart
  EX.scatter = function (el, o) {
    const pts = o.points.filter((p) => p.y != null && isFinite(p.y) && (o.xCat || (p.x != null && isFinite(p.x))));
    if (!pts.length) return EX.fail(el, 'No data for this selection');
    const y0 = d3.scaleLinear(d3.extent(pts, (p) => p.y), [0, 1]).nice(5);
    const yTicks = y0.ticks(5), yfmt = o.yfmt || d3.format('~g');
    const f = frame(el, o.height || 260, yTicks, yfmt, o.xLabel, o.yLabel);
    const { svg, W, H, m } = f;
    if (o.xCat) m.b += 0;
    const y = d3.scaleLinear(y0.domain(), [H - m.b, m.t]);
    let x, xTicks, xfmt;
    if (o.xCat) {
      x = d3.scaleBand(o.xCat, [m.l, W - m.r]).padding(0.2);
      axes(svg, f, null, y, { yTicks, yfmt, xLabel: o.xLabel, yLabel: o.yLabel });
      const short = (s) => s.replace('Zürich (Prof. ', 'ZH-').replace(')', '').replace('NorthDakota', 'N Dakota');
      svg.append('g').selectAll('text').data(o.xCat).join('text').attr('x', (d) => x(d) + x.bandwidth() / 2).attr('y', H - m.b + 15).attr('text-anchor', 'middle')
        .text((d) => (x.bandwidth() > 46 ? short(d) : short(d).slice(0, 4)));
    } else {
      const xd = d3.scaleLinear(d3.extent(pts, (p) => p.x), [0, 1]).nice(5).domain();
      x = d3.scaleLinear(xd, [m.l + 6, W - m.r - 6]); xTicks = x.ticks(Math.max(3, Math.floor((W - m.l) / 80))); xfmt = o.xfmt || d3.format('~g');
      axes(svg, f, x, y, { yTicks, xTicks, yfmt, xfmt, xLabel: o.xLabel, yLabel: o.yLabel });
    }
    const jit = (p, i) => (o.xCat ? x(p.x) + x.bandwidth() / 2 + (((i * 7919) % 100) / 100 - 0.5) * x.bandwidth() * 0.7 : x(p.x));
    const sorted = pts.map((p, i) => ({ ...p, i })).sort((a, b) => (a.color === MUTED ? -1 : 0) - (b.color === MUTED ? -1 : 0));
    const tip = tipFor(el);
    svg.append('g').selectAll('circle').data(sorted).join('circle').attr('cx', (p) => jit(p, p.i)).attr('cy', (p) => y(p.y)).attr('r', (p) => p.r || 4)
      .style('fill', (p) => p.color).style('stroke', 'var(--color-panel)').attr('stroke-width', 1.5).style('cursor', o.onClick ? 'pointer' : null)
      .on('mousemove', (ev, p) => showTip(el, tip, ev, p.tip.map((t) => ({ text: t }))))
      .on('mouseleave', () => { tip.hidden = true; })
      .on('click', (ev, p) => o.onClick && o.onClick(p));
    if (!o.xCat) svg.append('g').selectAll('text').data(pts.filter((p) => p.label)).join('text').attr('x', (p) => x(p.x) + 7).attr('y', (p) => y(p.y) + 4).text((p) => p.label);
  };

  // bars: [{label, value, color}] horizontal, from a zero line
  EX.hbars = function (el, o) {
    const bars = o.bars.filter((b) => b.value != null && isFinite(b.value));
    if (!bars.length) return EX.fail(el, 'No data');
    el.replaceChildren();
    const W = Math.max(280, el.clientWidth || 600), rowH = 26, H = bars.length * rowH + 34;
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`).style('display', 'block');
    const probe = svg.append('text'); let lw = 0; bars.forEach((b) => { probe.text(b.label); lw = Math.max(lw, probe.node().getComputedTextLength()); }); probe.remove();
    const m = { l: Math.min(W * 0.45, lw + 12), r: 48, t: 6, b: 26 };
    const ext = d3.extent([0, ...bars.map((b) => b.value)]);
    const x = d3.scaleLinear(ext, [m.l, W - m.r]).nice(4);
    const fmt = o.fmt || d3.format('~g');
    svg.append('g').selectAll('line').data(x.ticks(4)).join('line').attr('x1', x).attr('x2', x).attr('y1', m.t).attr('y2', H - m.b).style('stroke', 'var(--cds-chart-grid)');
    svg.append('g').selectAll('text').data(x.ticks(4)).join('text').attr('x', x).attr('y', H - 8).attr('text-anchor', 'middle').text(o.tickFmt || fmt);
    const g = svg.append('g').selectAll('g').data(bars).join('g').attr('transform', (b, i) => `translate(0,${m.t + i * rowH})`);
    g.append('text').attr('x', m.l - 8).attr('y', rowH / 2 + 4).attr('text-anchor', 'end').text((b) => b.label);
    g.append('rect').attr('x', (b) => x(Math.min(0, b.value))).attr('width', (b) => Math.max(1, Math.abs(x(b.value) - x(0)))).attr('y', 5).attr('height', rowH - 10).attr('rx', 4).style('fill', (b) => b.color);
    g.append('text').attr('x', (b) => x(Math.max(0, b.value)) + 5).attr('y', rowH / 2 + 4).text((b) => fmt(b.value));
    svg.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', m.t).attr('y2', H - m.b).style('stroke', 'var(--cds-chart-axis)');
  };

  EX.legend = function (el, items) {
    let lg = el.parentElement.querySelector(':scope > .ex-legend');
    if (!lg) { lg = document.createElement('div'); lg.className = 'ex-legend'; el.after(lg); }
    lg.replaceChildren(...items.map((it) => { const s = document.createElement('span'); const i = document.createElement('i'); i.style.background = it.color; s.append(i, document.createTextNode(it.name)); return s; }));
  };
})();
