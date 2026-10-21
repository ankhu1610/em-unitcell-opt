// Offline static SVG QA. This does not launch a browser or access the network.
const fs = require('node:fs');
const path = require('node:path');
const [input, output, modules] = process.argv.slice(2);
if (!input || !output || !modules) throw new Error('Usage: input.html output.png absolute-node-modules');
if (fs.existsSync(output)) throw new Error('Refusing to overwrite an existing raster');
const sharp = require(require.resolve('sharp', {paths:[modules]}));
const html = fs.readFileSync(input,'utf8');
const match = html.match(/<svg\b[\s\S]*?<\/svg>/);
if (!match) throw new Error('No SVG figure found');
if (/\b(?:href|src)\s*=|<script\b|<foreignObject\b|<image\b/i.test(match[0])) throw new Error('Only self-contained vector figures are supported');
let svg = match[0].replace('<svg ','<svg xmlns="http://www.w3.org/2000/svg" ');
svg = svg.replace(/(<svg[^>]*>)/,'$1<style>.panel-title{font:bold 18px Arial}.tick{font:12px Arial}.axis{font:14px Arial}</style>');
sharp(Buffer.from(svg),{density:144}).flatten({background:'#ffffff'}).png().toFile(path.resolve(output))
  .then(info => console.log(JSON.stringify({output:path.resolve(output),width:info.width,height:info.height})))
  .catch(error => {console.error(error);process.exitCode=1;});
