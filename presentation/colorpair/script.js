/* =========================================================================
   Color pair explorable: scroll-driven figure walkthrough.

   The page is two scrolly containers around a full-width "model" interlude:
     - Part A (#part-a / #scene):  steps 1-2, the task.
     - Step 3 interlude:           full-width text + a full-width model.svg.
     - Part B (#part-b / #scene2): steps 4-10, the results.
   Each scroll step (a section <div>) carries a data-scene key. graph-scroll
   fires `active` with the step index; updateScene() crossfades to that step's
   scene. Figures are iframed (lazy-loaded on first view) and scaled to fit.
   ========================================================================= */

// Bump this whenever a figure file in assets/ is regenerated, so browsers fetch
// the new version instead of a stale cached iframe.
var ASSET_VERSION = '21';

// Scene definitions. `src` (relative to this page) marks an iframed figure with
// its native pixel size. `model` is not here: step 3 renders it inline (below).
var SCENES = {
  task:     { src: 'assets/task.svg',               w: 600,  h: 550, scale: 0.8 },
  pca:      { src: 'assets/pca3d.html',             w: 820,  h: 720 },
  behavior: { src: 'assets/behavior_manifold.html', w: 820,  h: 720 },
  steer1:   { src: 'assets/steer1.html',            w: 1180, h: 1060 },  // manifold only (top)
  steer2:   { src: 'assets/steer2.html',            w: 1180, h: 1060 },  // manifold + linear (2x2)
  steer3:   { src: 'assets/steer3.html',            w: 1180, h: 1060 },  // multi: manifold only (top)
  steer4:   { src: 'assets/steer4.html',            w: 1180, h: 1060 },  // multi: manifold + linear (2x2)
};

// Activation line: a step lights up at ~55% of the viewport height.
function triggerOffset() { return Math.round(window.innerHeight * 0.55); }

// Build a scrolly instance: stacked crossfading scene layers in `mountId`,
// driven by graph-scroll over the section divs in `sectionsSel`.
function createScrolly(opts) {
  var sceneEl = document.getElementById(opts.mountId);
  var layers = {};                       // scene key -> { el, iframe?, w?, h?, ... }

  opts.sceneKeys.forEach(function (key) {
    var def = SCENES[key];
    var layer = document.createElement('div');
    layer.className = 'layer';
    layer.dataset.scene = key;

    var fit = document.createElement('div');
    fit.className = 'fig-fit';
    var iframe = document.createElement('iframe');
    iframe.setAttribute('scrolling', 'no');
    iframe.width = def.w;                       // fixed internal viewport (native size)
    iframe.height = def.h;
    fit.appendChild(iframe);
    layer.appendChild(fit);
    layers[key] = { el: layer, fit: fit, iframe: iframe, w: def.w, h: def.h, scale: def.scale, loaded: false };
    sceneEl.appendChild(layer);
  });

  var sectionSel = d3.selectAll(opts.sectionsSel);
  var STEP_SCENE = sectionSel.nodes().map(function (n) { return n.getAttribute('data-scene'); });

  // Scale .fig-fit uniformly to fit the scene box, then center it. Uniform scale
  // preserves the figure's native aspect, so its framing matches the PDF.
  function fitLayer(key) {
    var L = layers[key];
    if (!L || !L.iframe) return;
    var box = sceneEl.getBoundingClientRect();
    if (!box.width || !box.height) return;
    var scale = Math.min(box.width / L.w, box.height / L.h);
    if (L.scale) scale *= L.scale;              // per-scene shrink (e.g. task: 0.8)
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

  function updateScene(i) { showScene(STEP_SCENE[i] || STEP_SCENE[0]); }

  var gs = d3.graphScroll()
    .container(d3.select(opts.containerSel))
    .graph(d3.select(opts.graphSel))
    .eventId(opts.eventId)
    .offset(triggerOffset())
    .sections(sectionSel)
    .on('active', updateScene);

  updateScene(0);

  return {
    layers: layers,
    fitAll: function () { Object.keys(layers).forEach(fitLayer); },
    refreshOffset: function () { gs.offset(triggerOffset()); },
    preload: function () {
      Object.keys(layers).forEach(function (key) {
        var L = layers[key];
        if (L.iframe && !L.loaded) {
          L.iframe.src = SCENES[key].src + '?v=' + ASSET_VERSION;
          L.loaded = true;
        }
      });
      Object.keys(layers).forEach(fitLayer);
    },
  };
}

// Step 3 interlude: scale the native-size model iframe to fill its full-width
// box, and reserve matching height (the .fig-fit inside is absolutely positioned).
// Run this BEFORE the scrolly instances initialize so the interlude already
// occupies its real height when graph-scroll measures Part B's position.
var MODEL_W = 1289, MODEL_H = 500;
function fitModel() {
  var fig = document.querySelector('.interlude-fig');
  var iframe = document.getElementById('model-fig');
  if (!fig || !iframe) return;
  var fit = iframe.parentNode;                  // the .fig-fit wrapper
  var width = fig.getBoundingClientRect().width;
  if (!width) return;
  var scale = width / MODEL_W;
  fit.style.transform = 'scale(' + scale + ')';
  fit.style.left = '0px';
  fit.style.top = '0px';
  fig.style.height = (MODEL_H * scale) + 'px';  // reserve space for the scaled figure
}

(function loadModel() {
  var iframe = document.getElementById('model-fig');
  if (iframe && !iframe.src) iframe.src = 'assets/model.svg?v=' + ASSET_VERSION;
  fitModel();
})();

var partA = createScrolly({
  mountId: 'scene', containerSel: '#part-a', graphSel: '#part-a #graph',
  sectionsSel: '#part-a #sections > div', eventId: 'colorpair-a',
  sceneKeys: ['task'],
});

var partB = createScrolly({
  mountId: 'scene2', containerSel: '#part-b', graphSel: '#part-b #graph2',
  sectionsSel: '#part-b #sections2 > div', eventId: 'colorpair-b',
  sceneKeys: ['pca', 'behavior', 'steer1', 'steer2', 'steer3', 'steer4'],
});

/* ----- Unified keyboard navigation across all stops -----
   The page is three blocks (Part A sections, the step-3 interlude, Part B
   sections), so graph-scroll's built-in per-instance arrow keys can't cross the
   interlude and would also land on stale positions. Replace them with one
   handler over the full ordered stop list, scrolling to LIVE element positions
   so it always crosses the interlude and lands correctly. */
d3.select(window).on('keydown.gscrollcolorpair-a', null).on('keydown.gscrollcolorpair-b', null);

function navStops() {
  var els = d3.selectAll('#part-a #sections > div').nodes().slice();
  els.push(document.querySelector('.interlude'));
  return els.concat(d3.selectAll('#part-b #sections2 > div').nodes());
}

// Absolute Y that places a stop's top near the top of the viewport. Sections
// carry their own 10vh padding-top, so aligning their border-top to the viewport
// top drops the heading ~10vh down (level with the vertically-centred figure);
// the interlude gets an explicit nudge so its heading isn't flush to the top.
function stopTargetY(el) {
  var top = el.getBoundingClientRect().top + window.pageYOffset;
  if (el.classList.contains('interlude')) top -= Math.round(window.innerHeight * 0.08);
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

d3.select(window).on('keydown.cpnav', function () {
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

d3.select(window).on('resize.colorpair', function () {
  partA.refreshOffset();
  partB.refreshOffset();
  partA.fitAll();
  partB.fitAll();
  fitModel();
});

// Pre-load every figure up front (hidden layers, opacity 0) so each figure's
// one-time post-render reflow happens off-screen, instead of as a visible
// left-shift when a step first scrolls into view. Then force a resize so both
// graph-scroll instances re-measure with the interlude at its final height
// (their free-scroll highlighting relies on those cached positions).
window.addEventListener('load', function () {
  partA.preload();
  partB.preload();
  fitModel();
  window.dispatchEvent(new Event('resize'));
});
