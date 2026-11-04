// Bundles app.js (and its imports) + style.css into a single self-contained dist/index.html.
// KaTeX is still loaded from the CDN; see README note if you need a fully offline file.
import {build} from 'esbuild';
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';

const bundle = await build({entryPoints: ['app.js'], bundle: true, minify: true, format: 'iife', write: false});
const js = bundle.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const css = readFileSync('style.css', 'utf8');

const swap = (html, from, to) => {
    if (!html.includes(from)) throw new Error(`index.html no longer contains: ${from}`);
    return html.replace(from, () => to); // function form so "$&" etc. in the bundle are not interpreted
};
let html = readFileSync('index.html', 'utf8');
html = swap(html, '<link rel="stylesheet" href="style.css">', `<style>${css}</style>`);
// type="module" keeps the script deferred, so it still runs after the DOM and the (deferred) KaTeX script
html = swap(html, '<script type="module" src="app.js"></script>', `<script type="module">${js}</script>`);

mkdirSync('dist', {recursive: true});
writeFileSync('dist/index.html', html);
console.log(`dist/index.html  ${(html.length / 1024).toFixed(1)} kB`);
