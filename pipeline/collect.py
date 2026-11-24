"""Collect podcast episode audio URLs for training data.

Takes the Apple top-podcasts chart, resolves each show's public RSS feed, and picks a few
recent episodes per show. Shows that appear in the MinusPod benchmark corpus are skipped so
the evaluation stays held out.

Output: data/episodes.jsonl  ({id, show, title, url, published})
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import feedparser
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "episodes.jsonl"
UA = {"User-Agent": "sponsorsieve-data/0.1"}

# Shows in the held-out MinusPod benchmark corpus (matched loosely on the name).
HELD_OUT = [
    "crime junkie", "tim dillon", "tosh show", "brilliant idiots", "drink champs",
    "daily tech news", "security now", "oxide and friends", "it's a thing", "daily gist",
    "andy and ari", "on air with dan", "cloud essentials", "glt",
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def chart(country: str, n: int) -> list[dict]:
    # Legacy iTunes chart feed (the newer marketing-tools API is unreliable).
    url = f"https://itunes.apple.com/{country}/rss/toppodcasts/limit={n}/json"
    entries = requests.get(url, headers=UA, timeout=30).json()["feed"]["entry"]
    return [{"name": e["im:name"]["label"], "id": e["id"]["attributes"]["im:id"]} for e in entries]


def feed_url(apple_id: str) -> str | None:
    r = requests.get("https://itunes.apple.com/lookup", params={"id": apple_id}, headers=UA, timeout=30)
    res = r.json().get("results", [])
    return res[0].get("feedUrl") if res else None


def main(per_show: int = 3, country: str = "us", n: int = 100) -> None:
    seen = set()
    if OUT.exists():
        seen = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    added = 0
    with OUT.open("a", encoding="utf-8") as out:
        for show in chart(country, n):
            name = show["name"]
            if any(h in name.lower() for h in HELD_OUT):
                print(f"skip held-out: {name}")
                continue
            try:
                url = feed_url(show["id"])
                if not url:
                    continue
                f = feedparser.parse(requests.get(url, headers=UA, timeout=30).content)
            except Exception as e:  # noqa: BLE001
                print(f"feed failed: {name}: {e}")
                continue
            picked = 0
            # Skip the very newest entry (often a trailer or bonus) and take the next ones.
            for e in f.entries[1:]:
                enc = next((l for l in e.get("links", []) if l.get("rel") == "enclosure"), None)
                if not enc or "audio" not in enc.get("type", "audio"):
                    continue
                eid = f"{slug(name)}-{hashlib.sha1(enc['href'].encode()).hexdigest()[:10]}"
                if eid in seen:
                    picked += 1
                else:
                    out.write(json.dumps({"id": eid, "show": name, "title": e.get("title", ""),
                                          "url": enc["href"], "published": e.get("published", "")}) + "\n")
                    seen.add(eid)
                    picked += 1
                    added += 1
                if picked >= per_show:
                    break
            print(f"{name}: {picked}")
    print(f"added {added} episodes -> {OUT}")


if __name__ == "__main__":
    main(*(int(a) if a.isdigit() else a for a in sys.argv[1:]))
