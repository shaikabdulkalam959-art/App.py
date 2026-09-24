# Sai University — Research Intelligence Dashboard

An interactive Streamlit dashboard for exploring Sai University's research
output, publication quality, institutional contribution, and impact — built
for university leadership, faculty, and researchers.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app fetches live data on every load (cached for 1 hour) from:
`https://sai-publications-dashboard.vercel.app/api/publications`

No manual data entry — everything is pulled and cleaned dynamically.

## What's inside

**Pipeline:** Publications API → `requests` → `pandas` cleaning →
`matplotlib`/`seaborn` visuals → Streamlit UI.

**Data handling decisions (for accuracy):**
- Records are **de-duplicated per unique publication** (by DOI, falling back
  to Title+Year) before counting publication-level KPIs, since the same
  paper can appear once per co-authoring SaiU faculty member in the raw
  feed. Author/faculty-level views intentionally use the un-deduplicated
  records, since each row represents one author's contribution.
- `Indexing Status` (e.g. `"Scopus | WoS"`) is split into a list so a paper
  indexed in both databases is correctly counted in both KPIs without
  double-counting the publication itself.
- `SJR Quartile` falls back to `Year Wise Quartile` when blank; publications
  with neither (common for conference papers, books, and law journals) are
  reported separately rather than silently dropped or miscounted.
- SDG tags come from `SaiU SDG Indexing`, falling back to the Scopus/WoS
  "Achieved SDG" fields when the SaiU field is empty; multi-value and
  trailing-pipe fields (`"12|9|"`) are parsed defensively.
- Rows with an unparsable `Year` are excluded from year-based analysis and
  reported rather than silently coerced.

**Sections:**
1. **KPIs** — total publications, Scopus/WoS-indexed counts, Q1 share,
   schools, faculty authors, SDGs represented — all filter-aware.
2. **Auto-generated insights** — plain-language takeaways computed from the
   current filtered view (YoY change, leading school/author, indexing and
   SDG highlights).
3. **Visualizations** (tabbed) — publication trend over time, indexed vs.
   non-indexed trend, publications by school & document type, a school ×
   year activity heatmap, SJR quartile distribution, indexing status,
   top-N faculty contribution with a per-researcher profile drill-down, and
   SDG distribution with an SDG × school heatmap.
4. **Filters** (sidebar) — year range, school, author, document type,
   indexing status, SJR quartile, SDG, and publisher/source search. All
   KPIs, charts, and the explorer respond to these.
5. **Publication Explorer** — searchable, sortable table with direct links
   (DOI/article/journal, whichever is available) and CSV export of the
   current filtered view.

## Bonus features implemented
- Year-over-year table and insight
- Researcher profile view (select any faculty member)
- School × Year and SDG × School heatmaps (drill-down style views)
- Advanced free-text search across title, authors, keywords, publisher
- CSV export of the current filtered view
- Automatically generated narrative insights

## File structure
```
app.py              Streamlit application (single entry point)
requirements.txt    Python dependencies
test_clean.py       Standalone unit check of the cleaning/dedup logic
README.md           This file
```
