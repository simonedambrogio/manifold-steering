/* =========================================================================
   Bandit explorable: scroll-driven figure walkthrough.

   Each scroll step (a `#sections > div`) carries a data-scene key. graph-scroll
   fires `active` with the step index; updateScene() crossfades to that step's
   scene. Iframed figures (Plotly/SVG) are lazy-loaded on first view and scaled
   to fit the graphic; scenes without a `src` render as a placeholder.
   ========================================================================= */

// Bump this whenever a figure file in assets/ is regenerated, so browsers fetch
// the new version instead of a stale cached iframe.
var ASSET_VERSION = '2';

// Scene definitions. `src` (relative to this page) marks an iframed figure with
// its native pixel size; otherwise the scene is a placeholder. An optional
// `scale` (0–1) shrinks a figure below its fit-to-panel size.
var SCENES = {
  task:     { src: 'assets/task.svg',               w: 480, h: 290 },
};

// The scene shown at each scroll step, read from the section's data-scene.
var sectionSel = d3.selectAll('.container-1 #sections > div');
var STEP_SCENE = sectionSel.nodes().map(function (n) { return n.getAttribute('data-scene'); });

// Build one stacked layer per scene inside #scene.
var sceneEl = document.getElementById('scene');
var layers = {};                         // scene key -> { el, iframe?, w?, h?, loaded? }

Object.keys(SCENES).forEach(function (key) {
  var def = SCENES[key];
  var layer = document.createElement('div');
  layer.className = 'layer';
  layer.dataset.scene = key;

  if (def.src) {
    var fit = document.createElement('div');
    fit.className = 'fig-fit';
    var iframe = document.createElement('iframe');
    iframe.setAttribute('scrolling', 'no');
    iframe.width = def.w;                       // fixed internal viewport (native size)
    iframe.height = def.h;
    fit.appendChild(iframe);
    layer.appendChild(fit);
    layers[key] = { el: layer, fit: fit, iframe: iframe, w: def.w, h: def.h, scale: def.scale, loaded: false };
  } else {
    layer.innerHTML =
      '<div class="placeholder">' +
        '<div class="placeholder-step">placeholder</div>' +
        '<div class="placeholder-label">' + def.placeholder + '</div>' +
        '<div class="placeholder-hint">' + def.hint + '</div>' +
      '</div>';
    layers[key] = { el: layer };
  }
  sceneEl.appendChild(layer);
});

// Scale .fig-fit uniformly to fit the scene box, then center it. Uniform scale
// preserves the figure's native aspect, so its framing matches the PDF.
function fitLayer(key) {
  var L = layers[key];
  if (!L || !L.iframe) return;
  var box = sceneEl.getBoundingClientRect();
  if (!box.width || !box.height) return;
  var scale = Math.min(box.width / L.w, box.height / L.h);
  if (L.scale) scale *= L.scale;                // per-scene shrink (e.g. task: 0.8)
  L.fit.style.transform = 'scale(' + scale + ')';
  L.fit.style.left = ((box.width - L.w * scale) / 2) + 'px';
  L.fit.style.top = ((box.height - L.h * scale) / 2) + 'px';
}

function showScene(key) {
  Object.keys(layers).forEach(function (k) {
    layers[k].el.classList.toggle('active', k === key);
  });
  var L = layers[key];
  if (L && L.iframe && !L.loaded) {           // lazy-load the figure on first view
    L.iframe.src = SCENES[key].src + '?v=' + ASSET_VERSION;
    L.loaded = true;
  }
  if (L && L.iframe) fitLayer(key);
}

function updateScene(i) {
  showScene(STEP_SCENE[i] || STEP_SCENE[0]);
}

// Activation line: a step lights up at ~55% of the viewport height.
function triggerOffset() { return Math.round(window.innerHeight * 0.55); }

var gs = d3.graphScroll()
  .container(d3.select('.container-1'))
  .graph(d3.select('.container-1 #graph'))
  .eventId('bandit')
  .offset(triggerOffset())
  .sections(sectionSel)
  .on('active', updateScene);

/* ----- Unified keyboard navigation: scrolly steps + the full-width step 2 -----
   graph-scroll's built-in arrow keys only move between #sections steps, so they
   can't reach the full-width .fullstep block below the scrolly container. Disable
   them and replace with one handler over the full ordered stop list (steps +
   .fullstep), scrolling to LIVE element positions so down/up always reach step 2. */
d3.select(window).on('keydown.gscrollbandit', null);

function navStops() {
  var els = d3.selectAll('.container-1 #sections > div').nodes().slice();
  document.querySelectorAll('.fullstep, .splitstep').forEach(function (el) { els.push(el); });
  return els;
}

// Absolute Y that places a stop's top near the top of the viewport. Sections carry
// their own 10vh padding-top (heading drops level with the centred figure); the
// full-width step gets a small nudge so its heading isn't flush to the top.
function stopTargetY(el) {
  var top = el.getBoundingClientRect().top + window.pageYOffset;
  if (el.classList.contains('fullstep') || el.classList.contains('splitstep'))
    top -= Math.round(window.innerHeight * 0.08);
  return Math.max(0, Math.round(top));
}

function currentStopIndex(els) {
  var y = window.pageYOffset, best = 0, bestD = Infinity;
  els.forEach(function (el, i) {
    var d = Math.abs(stopTargetY(el) - y);
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

d3.select(window).on('keydown.bnav', function () {
  var e = d3.event, delta;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  switch (e.keyCode) {
    case 40: case 34: delta = 1; break;            // down arrow / page down
    case 38: case 33: delta = -1; break;           // up arrow / page up
    case 32: delta = e.shiftKey ? -1 : 1; break;   // space / shift-space
    default: return;
  }
  var els = navStops();
  var i = currentStopIndex(els);
  var ni = Math.max(0, Math.min(els.length - 1, i + delta));
  if (ni === i) return;
  e.preventDefault();
  window.scrollTo({ top: stopTargetY(els[ni]), behavior: 'smooth' });
});

// Full-width steering animation (native 1280×620): scale its .fig-fit to fill the
// step's width and reserve matching height (the .fig-fit inside is absolutely
// positioned). Same fit technique as the scrolly scenes; never upscales past 1:1.
var ANIM_W = 1280, ANIM_H = 620;
function fitAnim() {
  var fig = document.querySelector('.anim-fig');
  if (!fig) return;
  var fit = fig.querySelector('.fig-fit');
  var iframe = fig.querySelector('iframe');
  if (!fit || !iframe) return;
  if (!iframe.src) iframe.src = 'assets/steer_animation_CD_L19_t95-105.html?v=' + ASSET_VERSION;
  var width = fig.getBoundingClientRect().width;
  if (!width) return;
  var scale = Math.min(1, width / ANIM_W);
  fit.style.transform = 'scale(' + scale + ')';
  fit.style.left = ((width - ANIM_W * scale) / 2) + 'px';
  fit.style.top = '0px';
  fig.style.height = (ANIM_H * scale) + 'px';   // reserve space for the scaled figure
}

d3.select(window).on('resize.bandit', function () {
  gs.offset(triggerOffset());
  Object.keys(layers).forEach(fitLayer);      // re-fit figures to the new panel size
  fitAnim();
});

updateScene(0);

// Pre-load every figure up front (hidden layers, opacity 0) so Plotly's one-time
// post-render reflow happens off-screen, instead of as a visible left-shift when a
// step first scrolls into view.
function preloadFigures() {
  Object.keys(layers).forEach(function (key) {
    var L = layers[key];
    if (L.iframe && !L.loaded) {
      L.iframe.src = SCENES[key].src + '?v=' + ASSET_VERSION;
      L.loaded = true;
    }
  });
  Object.keys(layers).forEach(fitLayer);
}
window.addEventListener('load', function () { preloadFigures(); fitAnim(); });
