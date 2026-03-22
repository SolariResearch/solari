#!/usr/bin/env python3
"""
Knowledge Base Builder — Ingest content from the web into searchable FAISS vector indices.

Scrapes URLs, downloads PDFs, transcribes YouTube videos, fetches Wikipedia articles
and Project Gutenberg books, then chunks the text, embeds it with sentence-transformers,
and stores everything in a local FAISS index for fast semantic retrieval.

Each "mind" is a named directory containing:
  - index.faiss        Dense inner-product index (all-MiniLM-L6-v2, 384-dim)
  - metadata.json.gz   Compressed JSON array of {content, hash, source, ingested_at, ...}
  - manifest.json      Summary stats, top keywords, inferred domain tags

Usage examples:
    # Ingest a web page
    python ingest.py --url "https://example.com/article" --mind research

    # Ingest a local PDF
    python ingest.py --pdf /path/to/paper.pdf --mind papers

    # Download and ingest a remote PDF
    python ingest.py --pdf-url "https://arxiv.org/pdf/2301.00001.pdf" --mind papers

    # Transcribe and ingest a YouTube video
    python ingest.py --youtube "https://youtube.com/watch?v=dQw4w9WgXcQ" --mind lectures

    # Search YouTube and ingest top results
    python ingest.py --youtube-search "machine learning fundamentals" --mind ml --max 5

    # Ingest a YouTube playlist
    python ingest.py --youtube-playlist "https://youtube.com/playlist?list=..." --mind lectures

    # Ingest a Wikipedia article
    python ingest.py --wikipedia "Transformer (machine learning model)" --mind ml

    # Search Wikipedia and ingest matching articles
    python ingest.py --wikipedia-search "reinforcement learning" --mind ml --max 5

    # Ingest a Project Gutenberg book
    python ingest.py --gutenberg "The Art of War" --mind strategy

    # Ingest a local text file
    python ingest.py --file notes.txt --mind research

    # Batch mode (file with URLs, one per line)
    python ingest.py --batch urls.txt --mind research

    # Custom storage directory
    python ingest.py --url "https://example.com" --mind demo --minds-dir /data/indices

Dependencies:
    pip install faiss-cpu sentence-transformers

Optional (for YouTube):
    pip install youtube-transcript-api
    yt-dlp (CLI tool)

Optional (for PDF):
    pdftotext (from poppler-utils)
"""

import argparse
try:
    import fcntl
except ImportError:
    fcntl = None  # Windows — file locking handled via fallback
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MINDS_DIR = "./minds"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_MAX_CHARS = 1500
PRE_CHUNK_MAX_CHARS = 1200
COOKIE_FILE = os.path.join(tempfile.gettempdir(), "youtube_cookies.txt")

_cached_encoder = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Print a timestamped log message."""
    print(f"[ingest] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Encoder (lazy-loaded, cached)
# ---------------------------------------------------------------------------

def _get_encoder():
    """Lazy-load and cache the SentenceTransformer encoder."""
    global _cached_encoder
    if _cached_encoder is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _cached_encoder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _cached_encoder


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split text into meaningful chunks at section or paragraph boundaries."""
    sections = re.split(r"\n(?=## )", text.strip())

    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 > max_chars and current:
                    chunks.append(current.strip())
                    current = para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip():
                chunks.append(current.strip())

    return [c for c in chunks if len(c) > 20]


def pre_chunk(text: str, max_chars: int = PRE_CHUNK_MAX_CHARS) -> str:
    """Insert paragraph breaks into unstructured text at sentence boundaries.

    Transcripts and long-form prose often lack structural markers.  This adds
    double-newline breaks every *max_chars* characters at sentence boundaries so
    that ``chunk_text`` later produces well-sized FAISS entries.
    """
    if "\n## " in text or text.count("\n\n") > len(text) / (max_chars * 2):
        return text

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    parts: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            parts.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current.strip():
        parts.append(current.strip())

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Hashing & deduplication
# ---------------------------------------------------------------------------

def compute_hash(text: str) -> str:
    """Return a short blake2b hash for deduplication."""
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------

def _mind_lock(mind_dir: Path):
    """Acquire an exclusive file lock for a mind directory."""
    lock_path = mind_dir / ".ingest.lock"
    lock_fd = open(lock_path, "w")
    if fcntl is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    else:
        # Windows fallback — msvcrt locking
        try:
            import msvcrt
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_LOCK, 1)
        except (ImportError, OSError):
            pass  # Best-effort locking on platforms without fcntl/msvcrt
    return lock_fd


def _mind_unlock(lock_fd) -> None:
    """Release a mind directory lock."""
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        else:
            try:
                import msvcrt
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (ImportError, OSError):
                pass
        lock_fd.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Metadata recovery
# ---------------------------------------------------------------------------

def _recover_metadata(meta_gz: Path) -> list[dict]:
    """Best-effort recovery of a truncated/corrupted gzip metadata file."""
    import zlib

    try:
        raw = meta_gz.read_bytes()
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        text = decompressor.decompress(raw).decode("utf-8")
        if text.startswith("["):
            last_brace = text.rfind("}")
            if last_brace > 0:
                return json.loads(text[: last_brace + 1] + "]")
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Keyword / domain-tag extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "is", "are", "was", "were", "been",
    "has", "had", "does", "did", "more", "also", "use", "used", "using",
    "than", "then", "into", "some", "could", "true", "false", "null",
    "none", "return", "import", "class", "def", "function", "var", "let",
    "const", "self", "public", "private", "void", "file", "line", "type",
    "name", "value", "data", "result", "test", "new",
})

_DOMAIN_HINTS = [
    ({"security", "vuln", "exploit", "attack", "audit"}, "security"),
    ({"defi", "protocol", "liquidity", "oracle", "swap"}, "defi"),
    ({"solidity", "contract", "evm", "ethereum", "erc"}, "solidity"),
    ({"python", "javascript", "rust", "golang", "java", "typescript"}, "code"),
    ({"math", "theorem", "proof", "algebra", "calculus"}, "mathematics"),
    ({"physics", "quantum", "thermodynamic", "relativity"}, "physics"),
    ({"business", "strategy", "market", "growth", "pricing"}, "business"),
    ({"biology", "immune", "neural", "brain", "genetic"}, "biology"),
    ({"game_theory", "nash", "equilibrium", "mechanism", "auction"}, "game_theory"),
    ({"cybernetics", "homeostasis", "feedback", "variety", "viable"}, "cybernetics"),
]


def _extract_keywords_and_tags(
    mind_name: str,
    new_chunks: list[tuple[str, str]],
    combined_meta: list[dict],
) -> tuple[list[str], list[str]]:
    """Derive top keywords and domain tags from ingested content."""
    sample_texts = [c[0] for c in new_chunks[:30]]
    sample_texts += [
        m.get("content", m.get("text", ""))
        for m in combined_meta[-20:]
        if m.get("content") or m.get("text")
    ]
    word_counts: Counter = Counter()
    for text in sample_texts:
        words = re.findall(r"[a-z_]{4,30}", text.lower())
        word_counts.update(
            w for w in words if w not in _STOP_WORDS and not w.startswith("http")
        )
    top_kw = [w for w, _ in word_counts.most_common(50)]

    domain_tags: set[str] = set()
    for kw in [mind_name] + top_kw[:20]:
        for keywords, tag in _DOMAIN_HINTS:
            if any(s in kw for s in keywords):
                domain_tags.add(tag)

    return top_kw, sorted(domain_tags)


# ---------------------------------------------------------------------------
# Core FAISS ingestion
# ---------------------------------------------------------------------------

def ingest_into_mind(
    mind_name: str,
    texts: list[str],
    source: str = "manual",
    minds_dir: str = DEFAULT_MINDS_DIR,
) -> int:
    """Chunk, embed, and store *texts* in the named FAISS mind.

    Creates the mind if it does not exist.  Deduplicates against existing
    entries via blake2b hashes.  Writes are atomic (temp file + rename) and
    protected by an exclusive file lock.

    Returns the number of *new* entries added.
    """
    import faiss
    import numpy as np

    mind_dir = Path(minds_dir) / mind_name
    mind_dir.mkdir(parents=True, exist_ok=True)

    lock_fd = _mind_lock(mind_dir)
    try:
        return _ingest_locked(mind_name, texts, source, mind_dir)
    finally:
        _mind_unlock(lock_fd)


def _ingest_locked(
    mind_name: str,
    texts: list[str],
    source: str,
    mind_dir: Path,
) -> int:
    """Core ingestion logic, called while holding the directory lock."""
    import faiss
    import numpy as np

    index_path = mind_dir / "index.faiss"
    meta_gz = mind_dir / "metadata.json.gz"
    meta_json = mind_dir / "metadata.json"
    manifest_path = mind_dir / "manifest.json"

    # -- Load existing data ------------------------------------------------
    existing_meta: list[dict] = []
    existing_hashes: set[str] = set()
    existing_index = None

    if meta_gz.exists():
        try:
            with gzip.open(meta_gz, "rt") as f:
                existing_meta = json.load(f)
        except Exception:
            existing_meta = _recover_metadata(meta_gz)
    elif meta_json.exists():
        with open(meta_json) as f:
            existing_meta = json.load(f)

    for entry in existing_meta:
        h = entry.get("hash")
        if h:
            existing_hashes.add(h)

    if index_path.exists():
        existing_index = faiss.read_index(str(index_path))

    # -- Chunk & deduplicate -----------------------------------------------
    all_chunks: list[str] = []
    for text in texts:
        all_chunks.extend(chunk_text(text))

    new_chunks: list[tuple[str, str]] = []
    for chunk in all_chunks:
        h = compute_hash(chunk)
        if h not in existing_hashes:
            new_chunks.append((chunk, h))
            existing_hashes.add(h)

    if not new_chunks:
        log(f"No new entries (all {len(all_chunks)} chunks already exist)")
        return 0

    # -- Encode ------------------------------------------------------------
    log(f"Encoding {len(new_chunks)} new chunks...")
    encoder = _get_encoder()
    chunk_texts = [c[0] for c in new_chunks]
    embeddings = encoder.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=512,
    ).astype("float32")

    now = datetime.now(timezone.utc).isoformat()

    new_meta = [
        {
            "content": chunk,
            "hash": h,
            "ingested_at": now,
            "source": source,
            "category": mind_name,
            "domain_tags": [],
        }
        for chunk, h in new_chunks
    ]

    # -- Merge into FAISS index --------------------------------------------
    dim = embeddings.shape[1]

    if existing_index is not None and existing_index.ntotal > 0:
        new_index = faiss.IndexFlatIP(dim)
        existing_vectors = existing_index.reconstruct_n(0, existing_index.ntotal)
        new_index.add(existing_vectors)
        new_index.add(embeddings)
    else:
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(embeddings)

    combined_meta = existing_meta + new_meta

    # -- Atomic write (temp + rename) --------------------------------------
    tmp_idx = index_path.with_suffix(".faiss.tmp")
    tmp_meta = meta_gz.with_suffix(".gz.tmp")
    faiss.write_index(new_index, str(tmp_idx))
    with gzip.open(tmp_meta, "wt") as f:
        json.dump(combined_meta, f)
    tmp_idx.rename(index_path)
    tmp_meta.rename(meta_gz)

    # -- Manifest ----------------------------------------------------------
    manifest: dict = {
        "mind_id": mind_name,
        "estimated_size_mb": round(index_path.stat().st_size / 1024 / 1024, 1),
        "entry_count": len(combined_meta),
        "last_updated": now,
        "created_at": now,
    }
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                old_manifest = json.load(f)
            manifest["created_at"] = old_manifest.get("created_at", now)
            for keep in (
                "description", "domain_tags", "sample_topics",
                "quality_metrics", "related_minds", "sources",
                "last_enriched",
            ):
                if keep in old_manifest:
                    manifest[keep] = old_manifest[keep]
        except Exception:
            pass

    try:
        top_kw, domain_tags = _extract_keywords_and_tags(
            mind_name, new_chunks, combined_meta
        )
        manifest["top_keywords"] = top_kw
        if domain_tags:
            manifest["domain_tags"] = domain_tags
    except Exception:
        pass

    if "description" not in manifest:
        manifest["description"] = f"Knowledge base: {mind_name}"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"Ingested {len(new_chunks)} new entries into '{mind_name}' "
        f"(total: {len(combined_meta)})")
    return len(new_chunks)


# ---------------------------------------------------------------------------
# High-level ingest helper (text -> pre-chunk -> store)
# ---------------------------------------------------------------------------

def ingest_text(
    text: str,
    mind_name: str,
    source_label: str,
    minds_dir: str = DEFAULT_MINDS_DIR,
) -> int:
    """Pre-chunk raw text and ingest it into the named mind."""
    if not text or len(text.strip()) < 50:
        log(f"Skipping empty/tiny text for {source_label}")
        return 0
    text = pre_chunk(text)
    return ingest_into_mind(mind_name, [text], source=source_label, minds_dir=minds_dir)


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------

def _get_youtube_title(url: str) -> str:
    """Fetch video title via yt-dlp."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", "--no-warnings", url],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip() or "Unknown"
    except Exception:
        return "Unknown"


def _parse_json3(path: Path) -> str:
    """Parse YouTube json3 subtitle file to plain text."""
    try:
        data = json.loads(path.read_text())
        events = data.get("events", [])
        texts: list[str] = []
        for event in events:
            for seg in event.get("segs", []):
                t = seg.get("utf8", "").strip()
                if t and t != "\n":
                    texts.append(t)
        return " ".join(texts)
    except Exception as e:
        log(f"json3 parse failed: {e}")
        return ""


def _parse_vtt(path: Path) -> str:
    """Parse VTT subtitle file to plain text."""
    lines: list[str] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if (
                not line
                or line == "WEBVTT"
                or "-->" in line
                or line.startswith("Kind:")
                or line.startswith("Language:")
            ):
                continue
            clean = re.sub(r"<[^>]+>", "", line)
            if clean and clean not in seen:
                seen.add(clean)
                lines.append(clean)
    return " ".join(lines)


def _youtube_transcript_ytdlp(url: str) -> tuple[str | None, str | None]:
    """Fallback: use yt-dlp to download auto-generated subtitles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            cmd = [
                "yt-dlp", "--write-auto-sub", "--sub-lang", "en",
                "--skip-download", "--sub-format", "json3",
                "-o", f"{tmpdir}/%(title)s.%(ext)s",
                "--no-warnings",
            ]
            if os.path.exists(COOKIE_FILE):
                cmd.extend(["--cookies", COOKIE_FILE])
            cmd.extend(["--format", "sb0", url])
            subprocess.run(cmd, capture_output=True, timeout=30)

            json3_files = list(Path(tmpdir).glob("*.json3"))
            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            sub_file = (
                json3_files[0] if json3_files
                else (vtt_files[0] if vtt_files else None)
            )
            if not sub_file:
                return None, None

            title = sub_file.stem.replace(".en", "")
            text = (
                _parse_json3(sub_file)
                if sub_file.suffix == ".json3"
                else _parse_vtt(sub_file)
            )
            return title, text
        except Exception as e:
            log(f"yt-dlp fallback failed: {e}")
            return None, None


def youtube_transcript(url: str) -> tuple[str | None, str | None]:
    """Extract transcript from a YouTube video.

    Tries the ``youtube-transcript-api`` library first, then falls back to
    downloading auto-generated subtitles with ``yt-dlp``.

    Returns ``(title, text)`` or ``(None, None)`` on failure.
    """
    video_id = None
    for pattern in [r"v=([^&]+)", r"youtu\.be/([^?]+)", r"shorts/([^?]+)"]:
        m = re.search(pattern, url)
        if m:
            video_id = m.group(1)
            break
    if not video_id:
        log(f"Cannot extract video ID from: {url}")
        return None, None

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        result = api.fetch(video_id)
        text = " ".join(snippet.text for snippet in result.snippets)
        title = _get_youtube_title(url)
        return title, text
    except Exception as e:
        log(f"Transcript API failed for {video_id}: {e}")
        return _youtube_transcript_ytdlp(url)


def youtube_search(query: str, max_results: int = 10) -> list[str]:
    """Search YouTube and return video URLs via yt-dlp."""
    try:
        result = subprocess.run(
            [
                "yt-dlp", f"ytsearch{max_results}:{query}",
                "--get-url", "--get-title", "--flat-playlist",
                "--no-warnings",
            ],
            capture_output=True, text=True, timeout=30,
        )
        lines = result.stdout.strip().split("\n")
        urls = [line for line in lines if line.startswith("http")]
        return urls[:max_results]
    except Exception as e:
        log(f"YouTube search failed: {e}")
        return []


def youtube_playlist(url: str, max_results: int = 50) -> list[str]:
    """Extract video URLs from a YouTube playlist via yt-dlp."""
    try:
        result = subprocess.run(
            [
                "yt-dlp", "--flat-playlist", "--get-url",
                "--no-warnings", "--playlist-end", str(max_results), url,
            ],
            capture_output=True, text=True, timeout=60,
        )
        return [
            line.strip()
            for line in result.stdout.strip().split("\n")
            if line.strip().startswith("http")
        ]
    except Exception as e:
        log(f"Playlist extraction failed: {e}")
        return []


def process_youtube_batch(
    urls: list[str], mind_name: str, minds_dir: str
) -> int:
    """Transcribe and ingest a list of YouTube videos."""
    total = 0
    for i, url in enumerate(urls):
        log(f"[{i + 1}/{len(urls)}] Processing: {url}")
        title, text = youtube_transcript(url)
        if text:
            source = re.sub(
                r"[^a-zA-Z0-9_-]", "_",
                f"youtube_{title[:50]}_{int(time.time())}",
            )
            entries = ingest_text(
                f"# {title}\nSource: YouTube\nURL: {url}\n\n{text}",
                mind_name, source, minds_dir,
            )
            total += entries
            log(f"  -> '{title[:60]}': {entries} entries")
        else:
            log("  -> No transcript available")
    return total


# ---------------------------------------------------------------------------
# Web page scraping
# ---------------------------------------------------------------------------

def fetch_url_text(url: str) -> str | None:
    """Fetch and extract visible text from a web page.

    Uses only the standard library (``urllib`` + ``html.parser``).
    """
    try:
        result = subprocess.run(
            [
                "python3", "-c",
                f"""
import urllib.request, html.parser, re

class TextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.skip = True
    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.skip = False
    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)

req = urllib.request.Request('{url}', headers={{'User-Agent': 'Mozilla/5.0'}})
html_content = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')
parser = TextExtractor()
parser.feed(html_content)
text = ' '.join(parser.text)
text = re.sub(r'\\s+', ' ', text).strip()
print(text[:50000])
""",
            ],
            capture_output=True, text=True, timeout=20,
        )
        return result.stdout.strip() or None
    except Exception as e:
        log(f"URL fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def ingest_pdf(
    path: str, mind_name: str, source_label: str, minds_dir: str
) -> int:
    """Extract text from a local PDF (via ``pdftotext``) and ingest it."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, timeout=120,
        )
        text = result.stdout
        if not text or len(text.strip()) < 100:
            log(f"PDF extraction empty for {path}")
            return 0
        log(f"PDF extracted: {len(text):,} chars from {Path(path).name}")
        return ingest_text(text, mind_name, source_label, minds_dir)
    except Exception as e:
        log(f"PDF extraction failed: {e}")
        return 0


def download_and_ingest_pdf(
    url: str, mind_name: str, source_label: str, minds_dir: str
) -> int:
    """Download a PDF from a URL and ingest its text."""
    import urllib.request

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        tmp_path = f.name
    try:
        log(f"Downloading PDF: {url[:80]}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        log(f"Downloaded {size_mb:.1f} MB")
        return ingest_pdf(tmp_path, mind_name, source_label, minds_dir)
    except Exception as e:
        log(f"PDF download failed: {e}")
        return 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

def wikipedia_article(title: str) -> tuple[str, str | None]:
    """Fetch the full text of a Wikipedia article via the MediaWiki API."""
    import urllib.parse
    import urllib.request

    api_url = (
        f"https://en.wikipedia.org/w/api.php?action=query"
        f"&titles={urllib.parse.quote(title)}"
        f"&prop=extracts&explaintext=true&redirects=true&format=json"
    )
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        pages = data["query"]["pages"]
        for pid, page in pages.items():
            if int(pid) < 0:
                continue
            text = page.get("extract", "")
            if text:
                return page.get("title", title), text
    except Exception as e:
        log(f"Wikipedia fetch failed for '{title}': {e}")
    return title, None


def wikipedia_search(query: str, max_results: int = 10) -> list[str]:
    """Search Wikipedia via the OpenSearch API.

    Uses progressive query simplification: full query, then stripped of filler
    phrases, then individual significant words.
    """
    import urllib.parse
    import urllib.request

    def _opensearch(q: str) -> list[str]:
        url = (
            f"https://en.wikipedia.org/w/api.php?action=opensearch"
            f"&search={urllib.parse.quote(q)}"
            f"&limit={max_results}&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return data[1] if len(data) > 1 else []

    filler_re = re.compile(
        r"\b(best practices|techniques|strategies|approaches|fundamentals|"
        r"basics|advanced|introduction to|overview of|guide to|"
        r"methods for|tools for|tips for|how to|in depth|"
        r"fix |modern |practical |comprehensive )\b",
        re.IGNORECASE,
    )

    try:
        results = _opensearch(query)
        if results:
            return results

        cleaned = filler_re.sub(" ", query).strip()
        cleaned = " ".join(cleaned.split())
        if cleaned and cleaned != query:
            log(f"  Retrying simplified: '{cleaned}'")
            results = _opensearch(cleaned)
            if results:
                return results

        noise = {"about", "their", "these", "those", "which", "where",
                 "would", "could", "should"}
        words = [
            w for w in query.split()
            if len(w) > 4 and w.lower() not in noise
        ]
        all_results: list[str] = []
        for word in words[:3]:
            hits = _opensearch(word)
            all_results.extend(hits)
            if len(all_results) >= max_results:
                break

        seen: set[str] = set()
        deduped: list[str] = []
        for r in all_results:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped[:max_results]
    except Exception as e:
        log(f"Wikipedia search failed: {e}")
        return []


def wikipedia_batch(
    titles: list[str], mind_name: str, minds_dir: str
) -> int:
    """Ingest multiple Wikipedia articles into a single mind."""
    total = 0
    for i, title in enumerate(titles):
        log(f"[{i + 1}/{len(titles)}] Wikipedia: {title}")
        real_title, text = wikipedia_article(title)
        if text:
            source = f"wikipedia_{re.sub(r'[^a-zA-Z0-9_-]', '_', real_title[:40])}"
            n = ingest_text(
                f"# {real_title}\nSource: Wikipedia\n\n{text}",
                mind_name, source, minds_dir,
            )
            total += n
            log(f"  -> {n} entries")
        else:
            log("  -> Not found")
    return total


# ---------------------------------------------------------------------------
# Project Gutenberg
# ---------------------------------------------------------------------------

def search_gutenberg(query: str) -> str | None:
    """Search Project Gutenberg and return the plain-text URL of the first hit."""
    import urllib.parse
    import urllib.request

    search_url = (
        f"https://www.gutenberg.org/ebooks/search/"
        f"?query={urllib.parse.quote(query)}&submit_search=Go%21"
    )
    try:
        req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=15).read().decode()
        m = re.search(r"/ebooks/(\d+)", html)
        if m:
            book_id = m.group(1)
            return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    except Exception as e:
        log(f"Gutenberg search failed: {e}")
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge Base Builder: ingest content from the web into "
                    "searchable FAISS vector indices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mind", required=True,
        help="Name of the target FAISS mind (index directory)",
    )
    parser.add_argument(
        "--minds-dir", default=DEFAULT_MINDS_DIR,
        help=f"Root directory for mind storage (default: {DEFAULT_MINDS_DIR})",
    )

    # Source selectors (mutually exclusive in practice, first match wins)
    parser.add_argument("--url", help="Fetch and ingest a web page")
    parser.add_argument("--pdf", help="Ingest a local PDF file")
    parser.add_argument("--pdf-url", help="Download and ingest a PDF from a URL")
    parser.add_argument("--file", help="Ingest a local text file")
    parser.add_argument("--youtube", help="Ingest a single YouTube video transcript")
    parser.add_argument("--youtube-search", help="Search YouTube, ingest top results")
    parser.add_argument("--youtube-playlist", help="Ingest videos from a YouTube playlist")
    parser.add_argument("--wikipedia", help="Ingest a single Wikipedia article by title")
    parser.add_argument("--wikipedia-search", help="Search Wikipedia, ingest top results")
    parser.add_argument("--gutenberg", help="Search Project Gutenberg for a book")
    parser.add_argument("--batch", help="File with URLs (one per line, # for comments)")
    parser.add_argument(
        "--max", type=int, default=50,
        help="Max items for search/playlist modes (default: 50)",
    )

    args = parser.parse_args()
    minds_dir: str = args.minds_dir
    start = time.time()
    total = 0

    if args.youtube:
        title, text = youtube_transcript(args.youtube)
        if text:
            total = ingest_text(
                f"# {title}\nSource: YouTube\n\n{text}",
                args.mind, f"youtube_{int(time.time())}", minds_dir,
            )
            log(f"Ingested '{title}': {total} entries")

    elif args.youtube_search:
        log(f"Searching YouTube: '{args.youtube_search}' (max {args.max})")
        urls = youtube_search(args.youtube_search, args.max)
        log(f"Found {len(urls)} videos")
        total = process_youtube_batch(urls, args.mind, minds_dir)

    elif args.youtube_playlist:
        log(f"Extracting playlist (max {args.max})")
        urls = youtube_playlist(args.youtube_playlist, args.max)
        log(f"Found {len(urls)} videos")
        total = process_youtube_batch(urls, args.mind, minds_dir)

    elif args.wikipedia:
        title, text = wikipedia_article(args.wikipedia)
        if text:
            total = ingest_text(
                f"# {title}\nSource: Wikipedia\n\n{text}",
                args.mind, f"wikipedia_{int(time.time())}", minds_dir,
            )
            log(f"Ingested Wikipedia '{title}': {total} entries")

    elif args.wikipedia_search:
        log(f"Searching Wikipedia: '{args.wikipedia_search}' (max {args.max})")
        titles = wikipedia_search(args.wikipedia_search, args.max)
        log(f"Found {len(titles)} articles")
        total = wikipedia_batch(titles, args.mind, minds_dir)

    elif args.gutenberg:
        log(f"Searching Gutenberg: '{args.gutenberg}'")
        url = search_gutenberg(args.gutenberg)
        if url:
            import urllib.request
            text = urllib.request.urlopen(url, timeout=30).read().decode(
                "utf-8", errors="ignore"
            )
            total = ingest_text(
                text, args.mind, f"gutenberg_{args.gutenberg[:30]}", minds_dir,
            )
            log(f"Ingested book: {total} entries")
        else:
            log("Book not found")

    elif args.url:
        text = fetch_url_text(args.url)
        if text:
            total = ingest_text(text, args.mind, f"web_{int(time.time())}", minds_dir)
            log(f"Ingested page: {total} entries")

    elif args.pdf:
        total = ingest_pdf(
            args.pdf, args.mind, f"pdf_{Path(args.pdf).stem}", minds_dir,
        )
        log(f"Ingested PDF: {total} entries")

    elif args.pdf_url:
        total = download_and_ingest_pdf(
            args.pdf_url, args.mind, f"pdf_{int(time.time())}", minds_dir,
        )
        log(f"Ingested PDF from URL: {total} entries")

    elif args.file:
        with open(args.file) as f:
            content = f.read()
        total = ingest_text(
            content, args.mind, f"file_{Path(args.file).stem}", minds_dir,
        )
        log(f"Ingested file: {total} entries")

    elif args.batch:
        with open(args.batch) as f:
            urls = [
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
            ]
        for url in urls:
            if "youtube.com" in url or "youtu.be" in url:
                title, text = youtube_transcript(url)
                if text:
                    n = ingest_text(
                        f"# {title}\n\n{text}",
                        args.mind, f"batch_{int(time.time())}", minds_dir,
                    )
                    total += n
                    log(f"  {title[:50]}: {n} entries")
            else:
                text = fetch_url_text(url)
                if text:
                    n = ingest_text(
                        text, args.mind, f"batch_{int(time.time())}", minds_dir,
                    )
                    total += n
                    log(f"  {url[:50]}: {n} entries")

    else:
        parser.print_help()
        sys.exit(1)

    elapsed = time.time() - start
    log(f"Done. {total} entries ingested into '{args.mind}' in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
