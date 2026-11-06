"""Ingestion : PDF, Word, EPUB, texte/Markdown, HTML et pages web -> documents + passages."""

import hashlib
import logging
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.parse import unquote

log = logging.getLogger(__name__)

FILE_TYPES = {".pdf": "pdf", ".docx": "docx", ".epub": "epub", ".md": "text",
              ".txt": "text", ".html": "web", ".htm": "web"}
USER_AGENT = "agentic-search-prototype/0.1 (recherche documentaire locale)"
_BLOCKS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre"]


@dataclass
class Extracted:
    title: str = ""
    author: str = ""
    date: str | None = None
    date_origin: str = ""
    sections: list = field(default_factory=list)      # [(titre de section, texte)]
    warnings: list = field(default_factory=list)


# --- dates ---------------------------------------------------------------

def norm_date(value) -> str | None:
    """Convertit divers formats en 'AAAA-MM-JJ' (ou None)."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (datetime, date)):
            return value.strftime("%Y-%m-%d")
        s = str(value).strip()
        m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?", s)
        if m:
            y, mo, d = m.groups()
        else:
            m = re.match(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", s)
            if not m:
                return None
            y, mo, d = m.groups()
        iso = f"{y}-{mo or '01'}-{d or '01'}"
        date.fromisoformat(iso)
        return iso
    except (ValueError, TypeError):
        return None


# --- découpage en passages -------------------------------------------------

def _units(text: str, target: int) -> list[str]:
    units = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        if len(line.split()) <= target:
            units.append(line)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            w = sent.split()
            for k in range(0, len(w), target):
                units.append(" ".join(w[k:k + target]))
    return units


def chunk_text(text: str, target: int = 250, overlap: int = 40) -> list[str]:
    """Passages d'environ `target` mots, avec un chevauchement de `overlap` mots."""
    chunks: list[str] = []
    cur: list[str] = []          # contenu courant (avec chevauchement)
    fresh: list[str] = []        # unités nouvelles depuis la dernière émission
    n = 0
    for u in _units(text, target):
        w = len(u.split())
        if cur and n + w > target:
            chunk = "\n".join(cur)
            chunks.append(chunk)
            tail = " ".join(chunk.split()[-overlap:]) if overlap else ""
            cur, fresh, n = ([tail] if tail else []), [], len(tail.split())
        cur.append(u)
        fresh.append(u)
        n += w
    if fresh:
        fresh_words = sum(len(u.split()) for u in fresh)
        if chunks and fresh_words < 30:
            chunks[-1] += "\n" + "\n".join(fresh)
        else:
            chunks.append("\n".join(cur))
    return chunks


# --- utilitaires HTML ----------------------------------------------------------

def _soup(markup):
    from bs4 import BeautifulSoup, FeatureNotFound
    try:
        return BeautifulSoup(markup, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(markup, "html.parser")


def html_blocks(root) -> str:
    lines = []
    for el in root.find_all(_BLOCKS):
        if el.find_parent(["p", "li", "blockquote", "pre"]) is not None:
            continue
        t = " ".join(el.get_text(" ", strip=True).split())
        if t:
            lines.append(t)
    return "\n".join(lines) if lines else "\n".join(
        x.strip() for x in root.get_text("\n").splitlines() if x.strip())


def _meta(soup, *names) -> str:
    for n in names:
        el = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n}) \
            or soup.find("meta", attrs={"itemprop": n})
        if el and el.get("content"):
            return el["content"].strip()
    return ""


def extract_html(markup, url: str | None = None) -> Extracted:
    soup = _soup(markup)
    ex = Extracted()
    ex.title = _meta(soup, "og:title", "twitter:title") or (soup.title.get_text(strip=True) if soup.title else "")
    ex.author = _meta(soup, "author", "article:author")
    raw = _meta(soup, "article:published_time", "datePublished", "date", "pubdate", "DC.date.issued")
    if not raw:
        t = soup.find("time", attrs={"datetime": True})
        raw = t["datetime"] if t else ""
    if not raw:
        m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', str(markup)[:200000])
        raw = m.group(1) if m else ""
    ex.date, ex.date_origin = norm_date(raw), "meta_web"
    if not ex.date and url:
        m = re.search(r"/(20\d{2}|19\d{2})/(\d{2})/(\d{2})?", url)
        if m:
            ex.date, ex.date_origin = norm_date("-".join(g for g in m.groups() if g)), "url"
    if not ex.date:
        ex.date_origin = ""

    text = ""
    try:                                     # extraction du corps de l'article, si disponible
        import trafilatura
        text = trafilatura.extract(str(markup) if not isinstance(markup, str) else markup) or ""
    except Exception:
        text = ""
    if len(text.split()) < 50:
        for t in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"]):
            t.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        text = html_blocks(root)
    ex.sections = [("", text)]
    return ex


# --- lecteurs par format ---------------------------------------------------------

def reflow(lines: list[str]) -> list[str]:
    """Reconstitue des paragraphes à partir des lignes d'un PDF."""
    paras, cur = [], ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            if cur:
                paras.append(cur)
                cur = ""
            continue
        if not cur:
            cur = ln
        elif cur.endswith("-") and ln[:1].islower():
            cur = cur[:-1] + ln
        elif re.search(r"[.!?:»”)]$", cur) and ln[:1].isupper():
            paras.append(cur)
            cur = ln
        else:
            cur += " " + ln
    if cur:
        paras.append(cur)
    return paras


def strip_repeated_lines(pages: list[list[str]]) -> list[list[str]]:
    """Retire en-têtes, pieds de page et numéros qui se répètent sur plus de la moitié des pages."""
    if len(pages) < 4:
        return pages
    key = lambda s: re.sub(r"\d+", "#", s.strip())
    count: dict[str, int] = {}
    for pg in pages:
        for k in {key(x) for x in pg if x.strip() and len(x.strip()) < 80}:
            count[k] = count.get(k, 0) + 1
    limit = len(pages) / 2
    return [[x for x in pg if not (x.strip() and len(x.strip()) < 80 and count.get(key(x), 0) > limit)]
            for pg in pages]


def _clean_title(t: str, fallback: str) -> str:
    t = (t or "").strip()
    return fallback if len(t) < 4 or t.lower().startswith(("microsoft word", "untitled", "sans titre")) else t


def read_pdf(path: Path) -> Extracted:
    from pypdf import PdfReader
    r = PdfReader(str(path))
    if r.is_encrypted:
        try:
            r.decrypt("")
        except Exception:
            raise ValueError("PDF protégé par mot de passe")
    pages = []
    for pg in r.pages:
        try:
            pages.append((pg.extract_text() or "").splitlines())
        except Exception:
            pages.append([])
    paras = reflow([ln for pg in strip_repeated_lines(pages) for ln in pg])
    ex = Extracted(sections=[("", "\n".join(paras))])
    meta, title, author = None, "", ""
    try:
        meta = r.metadata
        title, author = (meta.title or ""), (meta.author or "")
        ex.date, ex.date_origin = norm_date(meta.creation_date), "metadonnee_fichier"
    except Exception:
        pass
    ex.title, ex.author = _clean_title(title, path.stem), author.strip()
    if not ex.date:
        ex.date_origin = ""
    if sum(len(p.split()) for p in paras) < 30 * max(1, len(pages)) / 3:
        ex.warnings.append("très peu de texte extrait (PDF scanné ? une OCR serait nécessaire)")
    return ex


def read_docx(path: Path) -> Extracted:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx.Document(str(path))
    sections: list[list] = [["", []]]
    first_heading = ""
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            txt = p.text.strip()
            if not txt:
                continue
            style = (p.style.name if p.style is not None and p.style.name else "").lower()
            if style.startswith(("heading", "titre", "title")):
                first_heading = first_heading or txt
                sections.append([txt, []])
            else:
                sections[-1][1].append(txt)
        elif tag == "tbl":
            for row in Table(child, d).rows:
                cells = []
                for c in row.cells:
                    t = c.text.strip()
                    if t and (not cells or cells[-1] != t):
                        cells.append(t)
                if cells:
                    sections[-1][1].append(" | ".join(cells))
    cp = d.core_properties
    ex = Extracted(sections=[(t, "\n".join(ls)) for t, ls in sections if ls],
                   author=(cp.author or "").strip())
    ex.title = _clean_title(cp.title or first_heading, path.stem)
    ex.date, ex.date_origin = norm_date(cp.created), "metadonnee_fichier"
    if not ex.date:
        ex.date_origin = ""
    return ex


def read_epub(path: Path) -> Extracted:
    with zipfile.ZipFile(path) as z:
        if "META-INF/encryption.xml" in z.namelist():
            enc = z.read("META-INF/encryption.xml").decode("utf-8", "replace")
            algos = set(re.findall(r'Algorithm="([^"]+)"', enc))
            fonts = {"http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"}
            if algos - fonts:
                raise ValueError("EPUB protégé (DRM) : extraction impossible")
        root = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = root.find(".//{*}rootfile").get("full-path")
        opf = ET.fromstring(z.read(opf_path))
        base = posixpath.dirname(opf_path)

        def md(tag):
            el = opf.find(f".//{{*}}metadata/{{*}}{tag}")
            return (el.text or "").strip() if el is not None else ""

        manifest = {i.get("id"): i for i in opf.iterfind(".//{*}manifest/{*}item")}
        ex = Extracted(title=_clean_title(md("title"), path.stem), author=md("creator"))
        ex.date, ex.date_origin = norm_date(md("date")), "metadonnee_fichier"
        if not ex.date:
            ex.date_origin = ""
        for ref in opf.iterfind(".//{*}spine/{*}itemref"):
            item = manifest.get(ref.get("idref"))
            if item is None or "nav" in (item.get("properties") or "").split():
                continue
            name = posixpath.normpath(posixpath.join(base, unquote(item.get("href"))))
            try:
                soup = _soup(z.read(name))
            except KeyError:
                continue
            for t in soup(["script", "style"]):
                t.decompose()
            head = soup.find(["h1", "h2", "h3"])
            title = head.get_text(" ", strip=True) if head else (soup.title.get_text(strip=True) if soup.title else "")
            text = html_blocks(soup.body or soup)
            if len(text.split()) >= 20:
                ex.sections.append((title[:150], text))
    return ex


def read_text(path: Path) -> Extracted:
    raw = path.read_text(encoding="utf-8", errors="replace")
    sections, title, cur_t, cur = [], "", "", []
    for line in raw.splitlines():
        m = re.match(r"#{1,6}\s+(.*)", line)
        if m:
            if cur:
                sections.append((cur_t, "\n".join(cur)))
            cur_t, cur = m.group(1).strip(), []
            title = title or cur_t
        else:
            cur.append(line)
    if cur:
        sections.append((cur_t, "\n".join(cur)))
    return Extracted(title=_clean_title(title, path.stem), sections=sections)


def read_html_file(path: Path) -> Extracted:
    ex = extract_html(path.read_bytes())
    ex.title = _clean_title(ex.title, path.stem)
    return ex


READERS = {".pdf": read_pdf, ".docx": read_docx, ".epub": read_epub, ".md": read_text,
           ".txt": read_text, ".html": read_html_file, ".htm": read_html_file}


def fetch_url(url: str) -> Extracted:
    import requests
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    if "pdf" in r.headers.get("content-type", "").lower():
        raise ValueError("lien vers un PDF : téléchargez-le et placez-le dans le dossier du corpus")
    ex = extract_html(r.content, url)
    ex.title = ex.title or url
    return ex


# --- assemblage ------------------------------------------------------------------------

def _add(docs, passages, source, dtype, ex: Extracted, words, overlap, warnings, resolver=None):
    if resolver is not None:
        resolver(ex, source, dtype)
    doc_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    n = 0
    for sec_title, text in ex.sections:
        for chunk in chunk_text(text, words, overlap):
            passages.append({"id": f"{doc_id}-{n:04d}", "doc_id": doc_id, "position": n,
                             "section": sec_title, "text": chunk})
            n += 1
    for w in ex.warnings:
        warnings.append(f"{source} : {w}")
    if n == 0:
        warnings.append(f"{source} : aucun texte exploitable, document ignoré")
        return
    docs.append({"id": doc_id, "title": ex.title or source, "type": dtype, "source": source,
                 "author": ex.author, "date": ex.date, "date_origine": ex.date_origin if ex.date else "",
                 "n_passages": n, "n_words": sum(len(p["text"].split()) for p in passages[-n:])})


def ingest_corpus(corpus_dir, urls_file=None, words=250, overlap=40, fetch=fetch_url, date_resolver=None):
    """Retourne (documents, passages, avertissements)."""
    root = Path(corpus_dir)
    docs, passages, warnings = [], [], []
    urls_path = Path(urls_file) if urls_file else root / "urls.txt"
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.name.startswith(".") or path.resolve() == urls_path.resolve()
                or path.suffix.lower() not in FILE_TYPES):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            ex = READERS[path.suffix.lower()](path)
        except Exception as e:
            warnings.append(f"{rel} : {type(e).__name__}: {e}")
            continue
        _add(docs, passages, rel, FILE_TYPES[path.suffix.lower()], ex, words, overlap, warnings, date_resolver)
    if urls_path.exists():
        for line in urls_path.read_text(encoding="utf-8").splitlines():
            url = line.split("#", 1)[0].strip()
            if not url:
                continue
            try:
                ex = fetch(url)
            except Exception as e:
                warnings.append(f"{url} : {type(e).__name__}: {e}")
                continue
            _add(docs, passages, url, "web", ex, words, overlap, warnings, date_resolver)
    return docs, passages, warnings
