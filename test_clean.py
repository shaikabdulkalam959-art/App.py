import pandas as pd
import numpy as np

sample = [
    {"Authors": "A, B", "SaiU Authors": "Abhishek Chakravarty ", "School": "School of Law",
     "Title": "Sacred Groves", "Year": "2021", "Source title": "NLU Assam Law Review",
     "SJR Quartile": "", "Year Wise Quartile": "", "DOI": "", "DOI Link": "",
     "Article Link": "https://example.com/a", "Journal Link": "", "Document Type": "Journal Article",
     "Indexing Status": "Non Indexed", "Achieved SDG (WoS)": "", "Achieved SDG (Scopus)": "",
     "SaiU SDG Indexing": "15", "Publisher": "NLU"},
    {"Authors": "C, D", "SaiU Authors": "Ajith Abraham", "School": "School of Artificial Intelligence",
     "Title": "Pistachio SC", "Year": "2025", "Source title": "IJSSOL",
     "SJR Quartile": "Q1", "Year Wise Quartile": "Q1", "DOI": "10.1/xyz", "DOI Link": "https://doi.org/10.1/xyz",
     "Article Link": "", "Journal Link": "", "Document Type": "Journal Article",
     "Indexing Status": "Scopus | WoS", "Achieved SDG (WoS)": "12", "Achieved SDG (Scopus)": "9|12",
     "SaiU SDG Indexing": "12|9|", "Publisher": "Taylor & Francis"},
    {"Authors": "E", "SaiU Authors": "Ajith Abraham", "School": "School of Artificial Intelligence",
     "Title": "Edited Book Vol 1", "Year": "2025", "Source title": "LNNS",
     "SJR Quartile": "Q4", "Year Wise Quartile": "Q4", "DOI": "10.1/book1", "DOI Link": "https://doi.org/10.1/book1",
     "Article Link": "", "Journal Link": "", "Document Type": "Edited Book",
     "Indexing Status": "Scopus", "Achieved SDG (WoS)": "", "Achieved SDG (Scopus)": "",
     "SaiU SDG Indexing": "", "Publisher": "Springer"},
    # duplicate publication co-authored by 2 SaiU faculty -> should dedupe by DOI
    {"Authors": "F, G", "SaiU Authors": "Faculty Two", "School": "School of Artificial Intelligence",
     "Title": "Pistachio SC", "Year": "2025", "Source title": "IJSSOL",
     "SJR Quartile": "Q1", "Year Wise Quartile": "Q1", "DOI": "10.1/xyz", "DOI Link": "https://doi.org/10.1/xyz",
     "Article Link": "", "Journal Link": "", "Document Type": "Journal Article",
     "Indexing Status": "Scopus | WoS", "Achieved SDG (WoS)": "12", "Achieved SDG (Scopus)": "9|12",
     "SaiU SDG Indexing": "12|9|", "Publisher": "Taylor & Francis"},
    {"Authors": "H", "SaiU Authors": "", "School": "", "Title": "", "Year": "not-a-year",
     "Source title": "", "SJR Quartile": "", "Year Wise Quartile": "", "DOI": "", "DOI Link": "",
     "Article Link": "", "Journal Link": "", "Document Type": "", "Indexing Status": "",
     "Achieved SDG (WoS)": "", "Achieved SDG (Scopus)": "", "SaiU SDG Indexing": "", "Publisher": ""},
]

df = pd.DataFrame(sample)


def _split_pipe(val):
    if pd.isna(val) or str(val).strip() in ("", "-"):
        return []
    return [p.strip() for p in str(val).split("|") if p.strip()]


def _split_indexing(val):
    if pd.isna(val) or str(val).strip() in ("", "-"):
        return ["Non Indexed"]
    return [p.strip() for p in str(val).split("|") if p.strip()]


df = df.replace(r"^\s*$", np.nan, regex=True).replace("-", np.nan)
df["Year"] = pd.to_numeric(df.get("Year"), errors="coerce")
df = df[df["Year"].notna()].copy()
df["Year"] = df["Year"].astype(int)

df["Quartile"] = df.get("SJR Quartile")
df["Quartile"] = df["Quartile"].fillna(df["Year Wise Quartile"])
df["Quartile"] = df["Quartile"].where(df["Quartile"].isin(["Q1", "Q2", "Q3", "Q4"]), np.nan)

df["Indexing List"] = df.get("Indexing Status").apply(_split_indexing)
df["Is Scopus"] = df["Indexing List"].apply(lambda l: "Scopus" in l)
df["Is WoS"] = df["Indexing List"].apply(lambda l: "WoS" in l)
df["Is Indexed"] = df["Indexing List"].apply(lambda l: any(x != "Non Indexed" for x in l))


def combined_sdgs(row):
    for col in ("SaiU SDG Indexing", "Achieved SDG (Scopus)", "Achieved SDG (WoS)"):
        vals = _split_pipe(row.get(col))
        if vals:
            return sorted(set(v for v in vals if v.isdigit()))
    return []


df["SDG List"] = df.apply(combined_sdgs, axis=1)
df["SaiU Authors"] = df.get("SaiU Authors", "").fillna("")
df["SaiU Author List"] = df["SaiU Authors"].apply(lambda v: [a.strip() for a in str(v).split(",") if a.strip()])
df["School"] = df.get("School").fillna("Unspecified")
df["DOI"] = df.get("DOI")
df["Pub Key"] = df["DOI"].fillna(df["Title"].fillna("untitled").str.lower().str.strip() + "||" + df["Year"].astype(str))

print("Rows after year cleaning:", len(df))
print(df[["Title", "Year", "Quartile", "Indexing List", "Is Scopus", "Is WoS", "Is Indexed", "SDG List", "Pub Key"]])

dedup = df.drop_duplicates(subset="Pub Key")
print("\nUnique publications after dedup:", len(dedup))
assert len(df) == 4, f"expected 4 rows with valid year, got {len(df)}"
assert len(dedup) == 3, f"expected 3 unique pubs after DOI dedup, got {len(dedup)}"
assert df[df.Title == "Pistachio SC"]["Is WoS"].all()
assert set(df[df.Title == "Pistachio SC"]["SDG List"].iloc[0]) == {"9", "12"}
print("\nALL ASSERTIONS PASSED")
