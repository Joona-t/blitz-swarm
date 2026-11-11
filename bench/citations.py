"""M1: deterministic citation resolution (Phase 1, WP4).

Extract arXiv IDs, DOIs and URLs from a report and check that each one
exists, against public key-less endpoints:

  arXiv   https://export.arxiv.org/api/query?id_list=...   (Atom, batched)
  DOI     https://api.crossref.org/works/<doi>            (JSON)
  URL     HEAD (GET on 403/405), then the Wayback availability API

Classes per citation:
  resolved        exists, and the report does not claim a different title
  title_mismatch  exists, but the title the report gives near it does not
                  match the real one (a real ID attached to the wrong paper)
  not_found       the registry says it does not exist
  unverifiable    network error / blocked / rate-limited: excluded from the
                  ratio and reported, never counted as a pass or a fail

Why this exists: the v1 composite scored citations on FORMAT only, so a
fabricated arXiv ID in valid format earned full marks; the tool-less swarm
averaged 0.90 on it (docs/audit/2026-11-11/PHASE1-RESEARCH.md §1).

Only the cited identifiers leave the machine. No key, no account. Results
are cached in a JSON file so reruns are offline and reproducible;
`unverifiable` is never cached. Standard library only.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

RESOLVED, TITLE_MISMATCH, NOT_FOUND, UNVERIFIABLE = "resolved", "title_mismatch", "not_found", "unverifiable"

DEFAULT_CACHE = Path(__file__).parent / "cache" / "citations.json"
USER_AGENT = "blitz-swarm-bench/0.4 (citation existence check; stdlib urllib)"
ARXIV_DELAY_S = 3.0          # arXiv API courtesy delay between requests
TITLE_MATCH_MIN = 0.6        # share of the real title's content words found near the citation

_ARXIV_RE = re.compile(r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>\]\)]+)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\"<>\]\)]+", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "and", "for", "with", "from", "that", "this", "are", "via", "using", "into", "over", "their"}

# fetch(url, method) -> (status_code, body_text). status 0 = network failure.
Fetch = Callable[[str, str], tuple[int, str]]


@dataclass(slots=True)
class Citation:
    kind: str            # "arxiv" | "doi" | "url"
    ident: str           # normalised identifier
    context: str         # text around the citation in the report (for the title check)
    status: str = ""
    real_title: str = ""
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.ident}"


def _clean(token: str) -> str:
    return token.rstrip(".,;:!?'\"*_`>")


def extract(text: str) -> list[Citation]:
    """Distinct citations in order of first appearance. A doi.org / arxiv.org
    URL is recorded once, as the DOI / arXiv ID it carries."""
    text = text or ""
    found: dict[str, Citation] = {}

    def add(kind: str, ident: str, start: int, end: int) -> None:
        key = f"{kind}:{ident}"
        if key in found:
            return
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        line_end = len(text) if line_end == -1 else line_end
        lo, hi = max(line_start, start - 300), min(line_end, end + 300)
        context = text[lo:hi]
        # Wrapped prose often puts the title on the line above the link. Only
        # when this line says almost nothing itself do we borrow the previous
        # non-empty line; a reference list (one full entry per line) never
        # borrows its neighbour's title.
        if len(_content_words(context.replace(ident, " "))) < 4 and line_start > 0:
            prev_end = line_start - 1
            prev_start = text.rfind("\n", 0, prev_end) + 1
            prev = text[prev_start:prev_end]
            if prev.strip():
                context = prev[-300:] + " " + context
        found[key] = Citation(kind=kind, ident=ident, context=context)

    for m in _ARXIV_RE.finditer(text):
        add("arxiv", m.group(1), m.start(), m.end())
    for m in _DOI_RE.finditer(text):
        add("doi", _clean(m.group(1)).lower(), m.start(), m.end())
    for m in _URL_RE.finditer(text):
        url = _clean(m.group(0))
        low = url.lower()
        if "arxiv.org/" in low or "doi.org/" in low:
            continue        # already captured above as an ID
        add("url", url, m.start(), m.end())
    return list(found.values())


def _content_words(s: str) -> set[str]:
    return {w for w in _WORD_RE.findall((s or "").lower()) if len(w) > 3 and w not in _STOP}


def title_check(real_title: str, context: str, ident: str) -> str:
    """resolved | title_mismatch for an identifier that exists.

    The report "claims a title" when the text around the identifier has at
    least 4 content words besides the identifier itself. Then at least
    TITLE_MATCH_MIN of the real title's content words must appear there.
    A bare identifier with no descriptive text cannot mismatch.
    """
    real = _content_words(real_title)
    if not real:
        return RESOLVED
    ctx = _content_words(context.replace(ident, " "))
    ctx -= {"arxiv", "https", "http", "preprint"}
    if len(ctx) < 4:
        return RESOLVED
    share = len(real & ctx) / len(real)
    return RESOLVED if share >= TITLE_MATCH_MIN else TITLE_MISMATCH


# ---------------------------------------------------------------------------
# Network layer (injectable)
# ---------------------------------------------------------------------------


def http_fetch(url: str, method: str = "GET", *, timeout: float = 20.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = "" if method == "HEAD" else resp.read(2_000_000).decode("utf-8", errors="replace")
            return int(resp.status), body
    except urllib.error.HTTPError as exc:
        return int(exc.code), ""
    except Exception:
        return 0, ""


def _arxiv_titles(ids: list[str], fetch: Fetch) -> dict[str, str] | None:
    """{id: title} for the ids that exist; None on a network failure."""
    url = ("https://export.arxiv.org/api/query?id_list=" + ",".join(ids)
           + f"&max_results={len(ids)}")
    status, body = fetch(url, "GET")
    if status != 200 or "<feed" not in body:
        return None
    titles: dict[str, str] = {}
    for entry in re.findall(r"<entry>(.*?)</entry>", body, re.DOTALL):
        mid = re.search(r"<id>\s*https?://arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?\s*</id>", entry)
        mtitle = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        if mid and mtitle and mtitle.group(1).strip() and mtitle.group(1).strip().lower() != "error":
            titles[mid.group(1)] = re.sub(r"\s+", " ", mtitle.group(1)).strip()
    return titles


def _resolve_doi(doi: str, fetch: Fetch) -> tuple[str, str]:
    status, body = fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/"), "GET")
    if status == 200:
        try:
            title = (json.loads(body).get("message", {}).get("title") or [""])[0]
        except (json.JSONDecodeError, AttributeError, IndexError):
            title = ""
        return RESOLVED, title
    if status == 404:
        return NOT_FOUND, ""
    return UNVERIFIABLE, ""


def _resolve_url(url: str, fetch: Fetch) -> tuple[str, str]:
    status, _ = fetch(url, "HEAD")
    if status in (403, 405, 400, 501):
        status, _ = fetch(url, "GET")
    if 200 <= status < 400:
        return RESOLVED, ""
    if status in (404, 410):
        wstatus, wbody = fetch("https://archive.org/wayback/available?url=" + urllib.parse.quote(url, safe=""), "GET")
        if wstatus == 200:
            try:
                snap = json.loads(wbody).get("archived_snapshots", {}).get("closest") or {}
            except json.JSONDecodeError:
                snap = {}
            if snap.get("available"):
                return RESOLVED, ""
            return NOT_FOUND, ""
        return UNVERIFIABLE, ""
    return UNVERIFIABLE, ""       # 0 (network), 401/403 after GET, 429, 5xx


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def load_cache(path: Path | None) -> dict:
    if path and Path(path).exists():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(path: Path | None, cache: dict) -> None:
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_report(
    text: str,
    *,
    fetch: Fetch | None = None,
    cache_path: Path | None = DEFAULT_CACHE,
    offline: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Resolve every citation in `text`. Returns the M1 score block:

      {n_cited, resolved, title_mismatch, not_found, unverifiable,
       ratio, items: [{kind, ident, status, real_title, note}]}

    ratio = resolved / (n_cited - unverifiable); None when nothing could be
    verified (including a report that cites nothing). `offline=True` uses
    the cache only: anything not cached is `unverifiable`.
    """
    fetch = fetch or http_fetch
    cache = load_cache(cache_path)
    cites = extract(text)
    dirty = False

    def from_cache(c: Citation) -> bool:
        hit = cache.get(c.key)
        if not hit:
            return False
        c.real_title = hit.get("title", "")
        exists = hit.get("exists")
        c.status = title_check(c.real_title, c.context, c.ident) if exists else NOT_FOUND
        c.note = "cache"
        return True

    pending = [c for c in cites if not from_cache(c)]
    if offline:
        for c in pending:
            c.status, c.note = UNVERIFIABLE, "offline, not cached"
        pending = []

    arxiv_pending = [c for c in pending if c.kind == "arxiv"]
    for i in range(0, len(arxiv_pending), 40):
        batch = arxiv_pending[i:i + 40]
        if i:
            sleep(ARXIV_DELAY_S)
        titles = _arxiv_titles([c.ident for c in batch], fetch)
        for c in batch:
            if titles is None:
                c.status, c.note = UNVERIFIABLE, "arxiv api unreachable"
                continue
            exists = c.ident in titles
            c.real_title = titles.get(c.ident, "")
            cache[c.key] = {"exists": exists, "title": c.real_title}
            dirty = True
            c.status = title_check(c.real_title, c.context, c.ident) if exists else NOT_FOUND

    for c in pending:
        if c.kind == "arxiv":
            continue
        status, title = (_resolve_doi if c.kind == "doi" else _resolve_url)(c.ident, fetch)
        if status == UNVERIFIABLE:
            c.status, c.note = UNVERIFIABLE, "network / blocked"
            continue
        exists = status == RESOLVED
        c.real_title = title
        cache[c.key] = {"exists": exists, "title": title}
        dirty = True
        c.status = title_check(title, c.context, c.ident) if exists else NOT_FOUND

    if dirty:
        save_cache(cache_path, cache)

    counts = {s: sum(1 for c in cites if c.status == s) for s in (RESOLVED, TITLE_MISMATCH, NOT_FOUND, UNVERIFIABLE)}
    verifiable = len(cites) - counts[UNVERIFIABLE]
    return {
        "n_cited": len(cites),
        **counts,
        "ratio": (counts[RESOLVED] / verifiable) if verifiable > 0 else None,
        "items": [{k: v for k, v in asdict(c).items() if k != "context"} for c in cites],
    }


def citations_v2_score(check: dict | None) -> float | None:
    """The composite-v2 citation term: resolved ratio x richness (3 distinct
    RESOLVED citations for full richness). 0.0 for a report that cites
    nothing; None when nothing was verifiable (offline miss, network down),
    so an outage never reads as a zero."""
    if check is None:
        return None
    if check.get("n_cited", 0) == 0:
        return 0.0
    if check.get("ratio") is None:
        return None
    return float(check["ratio"]) * min(1.0, check.get("resolved", 0) / 3.0)


__all__ = [
    "Citation", "RESOLVED", "TITLE_MISMATCH", "NOT_FOUND", "UNVERIFIABLE",
    "extract", "title_check", "check_report", "citations_v2_score", "http_fetch",
    "DEFAULT_CACHE",
]
