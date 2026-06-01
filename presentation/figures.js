/* =========================================================================
   Presentation manifest.

   The ordered list of figure explorables. The landing page (index.html)
   renders these as links in this order; add a new entry here when you build
   another figure folder. Each `dir` is a folder containing its own
   index.html / style.css / script.js.
   ========================================================================= */
window.FIGURES = [
  {
    dir: 'colorpair',
    title: 'The color pair experiment',
    blurb: 'A network folds colors into a ring with no prompting, seen in its representation, its behavior, and how it steers.',
  },
  {
    dir: 'bandit',
    title: 'Value steering in a bandit',
    blurb: 'An agent\'s latent value beliefs over a four-armed bandit live on a manifold; steering along it moves the agent\'s choices.',
  },
];
