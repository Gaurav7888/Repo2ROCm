---
name: paper_navigation
description: How to navigate a paper PDF without reading it cover-to-cover
when_to_use: At the start of the paper-research agent's turn, after PaperFetch and before any PaperRead calls
allowed_tools: ["PaperOutline", "PaperRead"]
---

# Navigating a paper efficiently

Papers are 8–60+ pages. Don't read them linearly. Use `PaperOutline` first,
then use `PaperRead` to pull *only* the sections you need.

## The default reading order

For a reproducibility task, almost everything you need lives in 3 places:

1. **The results tables** — for the headline number to reproduce.
2. **The §Experimental Setup / §Implementation Details / §Hyperparameters
   section** — for the config tuple (model checkpoint, dataset subset, all
   knobs).
3. **The Appendix** — when (1) and (2) are silent on a knob, the appendix
   almost always names it. Do not skip the appendix.

Read in this order: outline → setup section → headline-claim table → appendix
for any missing knob. Optional: §Experiments intro paragraph (sometimes
specifies the prompt template).

## What to ignore

Skip these unless something downstream forces you back:

- §Abstract — already in the agent's system prompt; never enough on its own.
- §Introduction — marketing.
- §Related Work — about other papers, not this one's setup.
- §Discussion / §Limitations / §Conclusion — narrative, no configs.
- Anything labelled "preliminaries" / "background".

## Outline-first protocol

```
1. PaperOutline(pdf_path="papers/<arxiv>.pdf")
   -> note: page_count, sections list, tables list, setup_hint_offsets
```

Decide:

- **Which table holds the headline result?** Look at table captions; pick the
  one whose caption matches the paper's main claim (e.g. "Performance on
  LongBench" for a KV-cache paper).
- **Which section names the setup?** Look for any of: "Experimental Setup",
  "Implementation Details", "Hyperparameters", "Setup", "Training Details".
  If the outline shows multiple §4.x subsections, the setup is usually §4.1.
- **Is there an appendix?** If `page_count > sections[last].page_hint + 5`,
  the trailing pages are almost certainly an appendix worth scanning.

Then:

```
2. PaperRead(pdf_path=..., section="4.1 Experimental Setup")
3. PaperRead(pdf_path=..., section="5 Results")        # or whichever table-bearing section
4. PaperRead(pdf_path=..., chunk=N) for appendix pages, if any
```

For tables that didn't survive PDF extraction cleanly, retry with
`PaperRead(source="html", section=...)`. The arXiv rendered HTML is much
better than PDF for structured tables.

## Paging through long sections

If a section is bigger than `chars_per_chunk` (default 12000), `PaperRead`
returns `has_next=True`. Keep calling with `chunk=N+1` until `has_next=False`.
Do not give up because chunk 0 didn't contain the number you wanted.

## When the PDF is hostile

- PDFs with 2-column layout sometimes interleave columns. If `PaperRead`
  output looks scrambled, retry with `source="html"`.
- LaTeX-typeset equations may show up as garbage. Ignore — equations aren't
  config knobs.
- Some papers split a single result into a figure + table. The figure caption
  often has the number you need; check `figures` in the outline.

## What you should hand off

After navigation, you should be able to say (in your own narration):

- "The headline experiment is X on dataset Y; the relevant table is Table N."
- "The Setup section is §M; I've read it; the hyperparameters listed are
  H1, H2, H3."
- "The appendix adds H4 and H5."

If you can't say all three, you haven't read enough — go back.
