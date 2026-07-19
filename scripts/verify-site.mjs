import { existsSync, readFileSync } from "node:fs";

const html = readFileSync("index.html", "utf8");
const visibleText = html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ");
const failures = [];

const requiredSnippets = [
  ["teaser section", 'id="teaser"'],
  ["overview section", 'id="overview"'],
  ["contributions block", 'id="contributions"'],
  ["data section", 'id="data"'],
  ["model section", 'id="model"'],
  ["results section", 'id="results"'],
  ["citation section", 'id="citation"'],
  ["main video source", "./static/OMG-demo.mp4"],
  ["language dinosaur demo source", "./static/results/language-dinosaur.mp4"],
  ["language aeroplane demo source", "./static/results/language-aeroplane.mp4"],
  ["language zombie demo source", "./static/results/language-zombie.mp4"],
  ["audio demo source", "./static/results/audio-liebestraume.mp4"],
  ["human reference demo source", "./static/results/human-reference-motion.mp4"],
  ["composition demo source", "./static/results/composition-realtime-switching.mp4"],
  ["composition demo poster", "./static/results/composition-realtime-switching-poster.jpg"],
  ["arxiv link", "https://arxiv.org/abs/2606.10340"],
  ["github code link", "https://github.com/Tsinghua-MARS-Lab/OMG"],
  ["hugging face model link", "https://huggingface.co/THU-MARS/OMG"],
  ["hugging face dataset link", "https://huggingface.co/datasets/THU-MARS/OMG-Data"],
  ["cite link", 'href="#citation"'],
  ["arxiv icon", "button-icon-arxiv"],
  ["github icon", "button-icon-github"],
  ["hugging face icon", "button-icon-huggingface"],
  ["cite icon", "button-icon-cite"],
  ["dataset hours", "1174.66"],
  ["model family", "OMG-DiT"]
];

const requiredText = [
  ["hero title", "OMG: Omni-Modal Motion Generation for Generalist Humanoid Control"]
];

for (const [label, snippet] of requiredText) {
  if (!visibleText.includes(snippet)) {
    failures.push(`Missing visible ${label}: ${snippet}`);
  }
}

for (const [label, snippet] of requiredSnippets) {
  if (!html.includes(snippet)) {
    failures.push(`Missing ${label}: ${snippet}`);
  }
}

const forbiddenSnippets = [
  "Text FID",
  "Audio FIDk",
  "fall rate",
  "Base model",
  "Large model",
  "XL model",
  "multi-demo-grid",
  "Paper PDF",
  "Code soon",
  "coming soon",
  "Coming soon"
];

for (const snippet of forbiddenSnippets) {
  if (html.includes(snippet)) {
    failures.push(`Unexpected old metric copy remains: ${snippet}`);
  }
}

const resultsSection = html.match(/<section class="section results-section"[\s\S]*?<\/section>/)?.[0] ?? "";
const resultsVideoSources = [...resultsSection.matchAll(/<source src="([^"]+)"/g)].map((match) => match[1]);
const carouselCount = (resultsSection.match(/data-demo-carousel/g) ?? []).length;

if (resultsVideoSources.length !== 8) {
  failures.push(`Expected 8 results videos, found ${resultsVideoSources.length}`);
}

if (carouselCount !== 2) {
  failures.push(`Expected 2 results carousels, found ${carouselCount}`);
}

const hrefs = [...html.matchAll(/\b(?:href|src)="([^"]+)"/g)].map((match) => match[1]);
for (const href of hrefs) {
  if (/^(https?:|mailto:|#)/.test(href)) continue;
  const path = href.replace(/^\.\//, "").split("#")[0].split("?")[0];
  if (path && !existsSync(path)) {
    failures.push(`Referenced asset does not exist: ${href}`);
  }
}

const requiredAssets = [
  "static/OMG-demo.mp4",
  "static/css/index.css",
  "static/images/omg/favicon.svg",
  "static/images/omg/icons/arxiv.svg",
  "static/images/omg/icons/github.svg",
  "static/images/omg/icons/cite.svg",
  "static/images/omg/overview.png",
  "static/images/omg/main-video-poster.jpg",
  "static/images/omg/dataset-statistics.png",
  "static/images/omg/method.png",
  "static/results/language-dinosaur.mp4",
  "static/results/language-aeroplane.mp4",
  "static/results/language-zombie.mp4",
  "static/results/audio-liebestraume.mp4",
  "static/results/audio-scherzo.mp4",
  "static/results/audio-paper-rings.mp4",
  "static/results/human-reference-motion.mp4",
  "static/results/composition-realtime-switching.mp4",
  "static/results/composition-realtime-switching-poster.jpg"
];

for (const asset of requiredAssets) {
  if (!existsSync(asset)) {
    failures.push(`Missing required asset: ${asset}`);
  }
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("Site verification passed.");
