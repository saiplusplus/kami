"""
Analyze TikTok-10M (sample) to infer trending haircuts.

Because the dataset does not contain explicit haircut, face shape, gender,
or race labels, this script uses keyword-based heuristics on captions and
hashtags to approximate:

- overall top 10 trending haircuts (by total play_count)
- top 10 trending haircuts for men and for women
- top 10 haircuts by inferred face shape
- top 10 haircuts by inferred race

The results are printed to stdout and saved into an HTML report.
"""

from __future__ import annotations
import time

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from datasets import load_dataset  # type: ignore


DatasetRow = Mapping[str, Any]


# Haircut categories and the phrases/hashtags that map to them.
HAIRCUT_KEYWORDS: Dict[str, Sequence[str]] = {
    "fade": [
        "fade",
        "skin fade",
        "low fade",
        "mid fade",
        "high fade",
        "taper fade",
    ],
    "taper": [
        "taper",
        "tapered cut",
    ],
    "buzz cut": [
        "buzz cut",
        "buzzcut",
    ],
    "crew cut": [
        "crew cut",
        "crewcut",
    ],
    "undercut": [
        "undercut",
        "disconnected undercut",
    ],
    "mullet": [
        "mullet",
        "wolf cut",
    ],
    "bob": [
        "bob cut",
        "bob haircut",
        "bob hairstyle",
        "bob",
        "lob",
    ],
    "pixie": [
        "pixie cut",
        "pixie haircut",
        "pixie",
    ],
    "curly": [
        "curly hair",
        "curly haircut",
        "curl cut",
        "curly hairstyle",
    ],
    "layers": [
        "layered cut",
        "layers",
        "layered hair",
        "face framing layers",
    ],
    "bangs / fringe": [
        "bangs",
        "fringe",
        "curtain bangs",
        "baby bangs",
    ],
    "shag": [
        "shag cut",
        "shaggy cut",
        "shag",
    ],
}


GENDER_KEYWORDS: Dict[str, Sequence[str]] = {
    "male": [
        "men haircut",
        "mens haircut",
        "men hair",
        "mens hair",
        "guy haircut",
        "boy haircut",
        "boys haircut",
        "male haircut",
        "for men",
        "for guys",
    ],
    "female": [
        "women haircut",
        "womens haircut",
        "woman haircut",
        "girl haircut",
        "girls haircut",
        "female haircut",
        "for women",
        "for girls",
        "ladies haircut",
    ],
}


RACE_KEYWORDS: Dict[str, Sequence[str]] = {
    # These are very rough proxies based on common hair-content hashtags.
    "black": [
        "black hair",
        "afro hair",
        "4c hair",
        "4b hair",
        "natural hair",
        "protective style",
        "box braids",
        "cornrows",
        "locs",
        "dreadlocks",
        "twists",
    ],
    "asian": [
        "asian hair",
        "korean perm",
        "korean hair",
        "japanese hair",
        "chinese hair",
    ],
    "latino / hispanic": [
        "latino hair",
        "hispanic hair",
    ],
    "white / caucasian": [
        "blonde hair",
        "brunette hair",
        "european hair",
    ],
}


FACE_SHAPE_KEYWORDS: Dict[str, Sequence[str]] = {
    "round": [
        "round face",
    ],
    "oval": [
        "oval face",
    ],
    "square": [
        "square face",
    ],
    "heart": [
        "heart face",
        "heart-shaped face",
    ],
    "diamond": [
        "diamond face",
    ],
}


@dataclass
class HaircutRecord:
    haircut: str
    gender: str
    race: str
    face_shape: str
    play_count: int


def load_tiktok10m_dataframe() -> pd.DataFrame:
    """
    Load a 2,000-row sample of the TikTok-10M dataset as a pandas DataFrame.
    """
    ds = load_dataset(
        "The-data-company/TikTok-10M",
        split="train[:2000]",
    ).select_columns(
        ["desc", "challenges", "play_count"]
    ).shuffle(seed=time.time())

    return ds.to_pandas()


def _to_lower_text(row: DatasetRow) -> str:
    # Support both Shofo-style ("description", "hashtags") and TikTok-10M-style ("desc", "challenges").
    desc = (row.get("description") or row.get("desc") or "") or ""

    hashtags = row.get("hashtags")
    if hashtags is None:
        hashtags = row.get("challenges") or []

    if isinstance(hashtags, str):
        # In some environments this may already be JSON-encoded.
        try:
            parsed = json.loads(hashtags)
            if isinstance(parsed, list):
                hashtags = parsed
        except Exception:
            hashtags = [hashtags]

    if not isinstance(hashtags, Iterable) or isinstance(hashtags, (str, bytes)):
        hashtags_iter: Iterable[Any] = []
    else:
        hashtags_iter = hashtags

    tags_text = " ".join(str(h) for h in hashtags_iter)
    combined = f"{desc} {tags_text}"
    return combined.lower()


def _detect_labels(text: str, mapping: Mapping[str, Sequence[str]], allow_multiple: bool = True) -> List[str]:
    labels: List[str] = []
    for label, patterns in mapping.items():
        for pattern in patterns:
            if pattern in text:
                labels.append(label)
                break
        if not allow_multiple and labels:
            break
    return labels


def _safe_play_count(row: DatasetRow) -> int:
    # TikTok-10M stores play_count as a top-level integer column.
    if "play_count" in row:
        try:
            return int(row.get("play_count") or 0)
        except Exception:
            return 0

    # Fallback for structures where engagement metrics are nested.
    metrics = row.get("engagement_metrics")
    if isinstance(metrics, dict):
        value = metrics.get("play_count")
        try:
            return int(value or 0)
        except Exception:
            return 0

    if isinstance(metrics, str):
        try:
            parsed = json.loads(metrics)
            if isinstance(parsed, dict):
                value = parsed.get("play_count")
                return int(value or 0)
        except Exception:
            return 0

    return 0


def build_haircut_records(df: pd.DataFrame) -> List[HaircutRecord]:
    records: List[HaircutRecord] = []

    for _, row in df.iterrows():
        row_map: DatasetRow = row.to_dict()
        text = _to_lower_text(row_map)
        haircuts = _detect_labels(text, HAIRCUT_KEYWORDS, allow_multiple=True)
        if not haircuts:
            continue

        genders = _detect_labels(text, GENDER_KEYWORDS, allow_multiple=False)
        races = _detect_labels(text, RACE_KEYWORDS, allow_multiple=False)
        face_shapes = _detect_labels(text, FACE_SHAPE_KEYWORDS, allow_multiple=False)

        gender = genders[0] if genders else "unknown"
        race = races[0] if races else "unknown"
        face_shape = face_shapes[0] if face_shapes else "unknown"
        play_count = _safe_play_count(row_map)

        for h in haircuts:
            records.append(
                HaircutRecord(
                    haircut=h,
                    gender=gender,
                    race=race,
                    face_shape=face_shape,
                    play_count=play_count,
                )
            )

    return records


def records_to_dataframe(records: Sequence[HaircutRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "haircut": r.haircut,
                "gender": r.gender,
                "race": r.race,
                "face_shape": r.face_shape,
                "play_count": r.play_count,
            }
            for r in records
        ]
    )


def aggregate_top_overall(haircut_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    grouped = (
        haircut_df.groupby("haircut")["play_count"]
        .agg(total_play_count="sum", video_count="count")
        .reset_index()
        .sort_values("total_play_count", ascending=False)
        .head(n)
    )
    return grouped


def aggregate_top_by_dimension(haircut_df: pd.DataFrame, dimension: str, n: int = 10) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    for value, sub in haircut_df.groupby(dimension):
        if value == "unknown":
            continue
        grouped = (
            sub.groupby("haircut")["play_count"]
            .agg(total_play_count="sum", video_count="count")
            .reset_index()
            .sort_values("total_play_count", ascending=False)
            .head(n)
        )
        if not grouped.empty:
            results[str(value)] = grouped
    return results


def build_html_report(
    overall: pd.DataFrame,
    by_gender: Dict[str, pd.DataFrame],
    by_face_shape: Dict[str, pd.DataFrame],
    by_race: Dict[str, pd.DataFrame],
) -> str:
    parts: List[str] = []
    parts.append("<html><head><title>Trending Haircuts Analysis (TikTok-10M)</title></head><body>")
    parts.append("<h1>Trending Haircuts (TikTok-10M sample)</h1>")
    parts.append("<p><em>Note: haircut, gender, race, and face shape are inferred heuristically from captions and hashtags; results are approximate.</em></p>")

    parts.append("<h2>Top 10 Trending Haircuts (Overall)</h2>")
    parts.append(overall.to_html(index=False, border=1))

    parts.append("<h2>Top 10 Trending Haircuts by Gender</h2>")
    for gender, df in by_gender.items():
        parts.append(f"<h3>{gender.title()}</h3>")
        parts.append(df.to_html(index=False, border=1))

    parts.append("<h2>Top 10 Trending Haircuts by Face Shape</h2>")
    for face_shape, df in by_face_shape.items():
        parts.append(f"<h3>{face_shape.title()}</h3>")
        parts.append(df.to_html(index=False, border=1))

    parts.append("<h2>Top 10 Trending Haircuts by Race (Heuristic)</h2>")
    for race, df in by_race.items():
        parts.append(f"<h3>{race.title()}</h3>")
        parts.append(df.to_html(index=False, border=1))

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    print("Loading TikTok-10M dataset (2,000-row sample)...")
    df = load_tiktok10m_dataframe()

    print(f"Loaded {len(df):,} videos. Inferring haircut-related records...")
    records = build_haircut_records(df)
    haircut_df = records_to_dataframe(records)

    if haircut_df.empty:
        print("No haircut-related content detected with current keyword heuristics.")
        return

    print(f"Constructed {len(haircut_df):,} haircut records.")

    overall = aggregate_top_overall(haircut_df, n=10)
    by_gender = aggregate_top_by_dimension(haircut_df, "gender", n=10)
    by_face_shape = aggregate_top_by_dimension(haircut_df, "face_shape", n=10)
    by_race = aggregate_top_by_dimension(haircut_df, "race", n=10)

    print("\n=== Top 10 Trending Haircuts (Overall, by total play_count) ===")
    print(overall.to_string(index=False))

    print("\n=== Top 10 Trending Haircuts by Gender ===")
    for gender, table in by_gender.items():
        print(f"\n-- {gender.title()} --")
        print(table.to_string(index=False))

    print("\n=== Top 10 Trending Haircuts by Face Shape ===")
    for face_shape, table in by_face_shape.items():
        print(f"\n-- {face_shape.title()} --")
        print(table.to_string(index=False))

    print("\n=== Top 10 Trending Haircuts by Race (Heuristic) ===")
    for race, table in by_race.items():
        print(f"\n-- {race.title()} --")
        print(table.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = build_html_report(overall, by_gender, by_face_shape, by_race)
    output_path = f"trending_hairstyles_tiktok10m_{timestamp}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML report written to: {output_path}")


if __name__ == "__main__":
    main()

