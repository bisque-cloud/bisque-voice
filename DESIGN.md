---
colors:
  ground: "#ebe5f4"
  blush: "#f9d2e2"
  sky: "#cddbfb"
  panel: "#fbfaf7"
  ink: "#1c1a27"
  muted: "#4f4a63"
  accent: "#5a3fc0"
  rule: "#b9b2cc"
typography:
  title:
    fontFamily: "Hanken Grotesk"
    fontSize: 112
    fontWeight: 700
    lineHeight: 1.02
    letterSpacing: "-0.02em"
  heading:
    fontFamily: "Hanken Grotesk"
    fontSize: 72
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "-0.015em"
  body:
    fontFamily: "Hanken Grotesk"
    fontSize: 36
    fontWeight: 500
    lineHeight: 1.35
  code:
    fontFamily: "Space Mono"
    fontSize: 28
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Space Mono"
    fontSize: 24
    fontWeight: 700
    letterSpacing: "0.06em"
spacing:
  scale: [8, 16, 24, 40, 64, 96]
rounded:
  panel: 18
  chip: 999
---

# bisque-voice design

How bisque-voice looks in its pull request and release explainers: one calm,
light system for every slide, so a viewer who has seen one explainer
recognises the next.

## Overview

bisque-voice turns text into speech and gives back the start and end time of
every word. The look comes from that: a quiet, light page, words set large,
and timings in a monospace face beside them. Every slide sits on the same
light ground. There are no dark slides, and no slide changes theme partway
through.

## Colors

- **ground** `#ebe5f4`, **blush** `#f9d2e2`, **sky** `#cddbfb`: the
  background, as one dithered mesh used on every slide (below). All three sit
  within 5 L\* of each other, so the hue drifts and nothing reads as a shape.
- **panel** `#fbfaf7`: cards, code blocks and screenshots sit on it.
- **ink** `#1c1a27`: headings and body text. About 13:1 on the ground.
- **muted** `#4f4a63`: secondary text and captions. About 6.5:1 on the ground.
- **accent** `#5a3fc0`: one highlighted word or number per slide, and word
  timings. About 5.5:1 on the ground.
- **rule** `#b9b2cc`: hairlines and the timing ruler. Never text.

Every slide's background is this spec, unchanged:

```
data-background='dither:{"type":"mesh","background":"#ebe5f4","mesh":[{"color":"#f9d2e2","x":0.82,"y":0.3,"radius":0.95},{"color":"#cddbfb","x":0.15,"y":0.8,"radius":0.95}],"dither":{"mode":"noise","scale":3,"levels":14},"grain":0.05,"motion":{"kind":"mesh","speed":0.1,"amp":0.04}}'
```

## Typography

- **Hanken Grotesk** for titles, headings and body. Download the woff2 files
  from Google Fonts into `assets/`.
- **Space Mono** for code, commands, file names, versions and every timing.
- Sizes at 1920 × 1080 are in the front matter. Nothing is smaller than 24 px.
  The vertical layout keeps the same faces and steps the title down to 96 px.

## Layout

- A 120 px margin on landscape and 80 px on vertical.
- A top strip in **label** style, in muted ink, naming where the change lives
  (`install.sh`, `.github/workflows/release.yml`, `v0.1.38`), with a 2 px
  rule under it.
- One idea per slide. The heading says the change in a sentence; the content
  below shows it.

## Elevation & Depth

Flat. Panels are `panel` with a 1 px `rule` border and no shadow. The dither
ground is the only texture.

## Shapes

Panels have an 18 px radius. Chips (a version, a platform, a voice name) are
fully rounded, `panel` fill, `label` type.

## Components

- **Word-timing strip, the signature.** A short sentence set large in ink,
  each word with its start and end time under it in accent Space Mono
  (`0.30–0.68`), on a thin `rule` ruler. Use it where the change touches what
  is spoken or when, and nowhere else.
- **Code and commands** sit on a `panel` with ink text, so the highlighter's
  light theme reads correctly. Show the real command and its real output: the
  JSON it prints, the error it gives.
- **Before and after** are two panels side by side on landscape, stacked on
  vertical, each with a label chip (`before`, `after`).

## Do's and Don'ts

- Do keep every slide on the light ground. A dark slide is the one way this
  system fails: ink on near-black is unreadable.
- Do check contrast against the pixels behind the text: 4.5:1 for body text,
  3:1 for text above 32 px.
- Do show real output and real files from the change.
- Don't put text directly on the dither where a block of it runs long. Put
  it on a panel.
- Don't add a second accent colour, gradients on text, glows or drop
  shadows.
- Don't change the background from slide to slide.
