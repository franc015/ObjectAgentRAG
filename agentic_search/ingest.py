"""Ingestion : PDF, Word, EPUB, texte/Markdown, HTML et pages web -> documents + passages."""

import hashlib
import io
import json
import logging
import posixpath
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

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
    meta: dict = field(default_factory=dict)          # métadonnées variables (site, langue, ...)
    themes: list = field(default_factory=list)        # étiquettes de thèmes (voir themes.py)
    figures: list = field(default_factory=list)       # figures repérées (voir FigureCollector)


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


# --- figures ---------------------------------------------------------------------------

MAX_FIGURE_BYTES = 8 * 1024 * 1024
_MAGIC = ((b"\x89PNG\r\n\x1a\n", "png", "image/png"), (b"\xff\xd8\xff", "jpg", "image/jpeg"),
          (b"GIF87a", "gif", "image/gif"), (b"GIF89a", "gif", "image/gif"))
_PLACEHOLDER = re.compile(r"\[\[figure:(\d+)\]\]")
_CAPTION = re.compile(r"^(figure|fig\.|schéma|schema|graphique|illustration|tableau)\b", re.I)


def sniff_image(data: bytes) -> tuple[str, str] | None:
    """(extension, type MIME) d'après la signature du contenu (jamais d'après le nom du fichier)."""
    for sig, ext, mime in _MAGIC:
        if data.startswith(sig):
            return ext, mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    head = data[:600].lstrip().lower()
    if head.startswith((b"<svg", b"<?xml")) and b"<svg" in data[:2000].lower():
        return "svg", "image/svg+xml"
    return None


def image_size(data: bytes) -> tuple[int, int]:
    """(largeur, hauteur) en pixels, (0, 0) si inconnues (SVG, Pillow absent)."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return 0, 0


class FigureCollector:
    """Rassemble les figures d'un document. `add` renvoie une ligne de repère `[[figure:N]]` à insérer
    dans le texte (remplacée par `_materialize` une fois l'identifiant du document connu), ou None si
    l'image est écartée (trop petite : logo, icône ; format non pris en charge ; trop lourde ; quota)."""

    def __init__(self, min_px: int = 120, max_items: int = 40):
        self.min_px, self.max_items = min_px, max_items
        self.items: list[dict] = []
        self.skipped = {"petites": 0, "format": 0, "trop lourdes": 0, "quota": 0}

    def add(self, data: bytes | None = None, url: str | None = None, alt: str = "", legende: str = "",
            page: int = 0) -> str | None:
        ext = mime = None
        if data is not None:
            kind = sniff_image(data)
            if kind is None:
                self.skipped["format"] += 1            # EMF, WMF, TIFF... : non affichables dans un navigateur
                return None
            ext, mime = kind
            if len(data) > MAX_FIGURE_BYTES:
                self.skipped["trop lourdes"] += 1
                return None
            w, h = image_size(data)
            if w and (w < self.min_px or h < self.min_px):
                self.skipped["petites"] += 1
                return None
        elif not url:
            return None
        if len(self.items) >= self.max_items:
            self.skipped["quota"] += 1
            return None
        self.items.append({"data": data, "url": url, "ext": ext, "mime": mime, "alt": alt or "",
                           "legende": legende or "", "page": page})
        return f"[[figure:{len(self.items) - 1}]]"

    def warning(self) -> str | None:
        parts = [f"{n} {k}" for k, n in self.skipped.items() if n]
        return ("figures écartées : " + ", ".join(parts)) if parts else None


def _clean_alt(text: str) -> str:
    return " ".join(re.sub(r"[\[\]]", " ", text or "").split())[:200]


def figure_label(alt: str, legende: str) -> str:
    alt, legende = _clean_alt(alt), _clean_alt(legende)
    if legende and alt and alt.lower() not in legende.lower():
        return f"{legende} — {alt}"
    return legende or alt


def _materialize(ex: "Extracted", doc_id: str, figures_dir) -> list[dict]:
    """Enregistre les images, remplace les repères du texte par des lignes markdown `![texte](cible)`
    et retourne les métadonnées des figures (une par image distincte)."""
    infos: list[dict] = []
    seen: dict[str, str] = {}

    def line(m) -> str:
        i = int(m.group(1))
        if i >= len(ex.figures):
            return ""
        f = ex.figures[i]
        label = figure_label(f["alt"], f["legende"])
        if f["data"] is not None:
            sha = hashlib.sha256(f["data"]).hexdigest()
            name = f"{sha[:12]}.{f['ext']}"
            target = f"figures/{doc_id}/{name}"
            if sha not in seen:
                if figures_dir:
                    out = Path(figures_dir) / doc_id
                    out.mkdir(parents=True, exist_ok=True)
                    if not (out / name).exists():
                        (out / name).write_bytes(f["data"])
                w, h = image_size(f["data"])
                infos.append({"sha256": sha, "mime": f["mime"], "width": w, "height": h,
                              "octets": len(f["data"]), "alt": _clean_alt(f["alt"]),
                              "legende": _clean_alt(f["legende"]), "page": f["page"], "chemin": target, "url": ""})
                seen[sha] = target
        else:
            target = f["url"]
            if target not in seen:
                infos.append({"sha256": "", "mime": "", "width": 0, "height": 0, "octets": 0,
                              "alt": _clean_alt(f["alt"]), "legende": _clean_alt(f["legende"]),
                              "page": f["page"], "chemin": "", "url": target})
                seen[target] = target
        return f"![{label}]({target})"

    ex.sections = [(t, _PLACEHOLDER.sub(line, text)) for t, text in ex.sections]
    return infos


_FIG_LINE = re.compile(r"^!\[([^\]]*)\]\((?:figures/[^)\s]+|https?://[^)\s]+)\)$")


def figure_passage_line(line: str) -> str | None:
    """Texte indexable d'une ligne figure : sa légende (vide : la ligne est omise). None : pas une figure."""
    m = _FIG_LINE.match(line.strip())
    if not m:
        return None
    label = m.group(1).strip()
    if not label:
        return ""
    return label if _CAPTION.match(label) else f"Figure : {label}"


def make_on_img(figures: FigureCollector, resolve):
    """Rappel pour `html_blocks` : retourne la ligne de repère d'une balise <img> (ou None)."""
    def on_img(el) -> str | None:
        src = (el.get("src") or el.get("data-src") or "").strip()
        if not src or src.startswith("data:"):
            return None
        for dim in ("width", "height"):
            v = str(el.get(dim, "")).strip()
            if v.isdigit() and int(v) < figures.min_px:
                figures.skipped["petites"] += 1
                return None
        data, url = resolve(src)
        if data is None and not url:
            return None
        fig = el.find_parent("figure")
        cap = fig.find("figcaption") if fig else None
        return figures.add(data=data, url=url, alt=el.get("alt", ""),
                           legende=cap.get_text(" ", strip=True) if cap else "")
    return on_img


# --- utilitaires HTML ----------------------------------------------------------

def _soup(markup):
    from bs4 import BeautifulSoup, FeatureNotFound
    try:
        return BeautifulSoup(markup, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(markup, "html.parser")


def html_blocks(root, on_img=None) -> str:
    """Texte des blocs. Avec `on_img`, les figures (<img>) deviennent des lignes de repère et les
    légendes (<figcaption>) sont rattachées à leur image."""
    lines = []
    wanted = _BLOCKS + (["img", "figcaption"] if on_img else [])
    for el in root.find_all(wanted):
        if el.name == "img":
            ph = on_img(el)
            if ph:
                lines.append(ph)
                continue
            fig = el.find_parent("figure")             # image écartée : on garde au moins sa légende
            cap = fig.find("figcaption") if fig else None
            if cap and cap.get_text(strip=True):
                lines.append(" ".join(cap.get_text(" ", strip=True).split()))
            continue
        if el.name == "figcaption":
            fig = el.find_parent("figure")
            if fig is not None and fig.find("img") is not None:
                continue                               # déjà portée par la ligne de l'image
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


def extract_html(markup, url: str | None = None, figures: FigureCollector | None = None, resolve=None) -> Extracted:
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
    body = soup.find("article") or soup.find("main")
    on_img = make_on_img(figures, resolve) if figures is not None and resolve is not None else None
    # avec figures : on lit le corps <article>/<main> bloc par bloc (trafilatura supprimerait les images)
    use_blocks = on_img is not None and body is not None and body.find("img") is not None
    if not use_blocks:
        try:                                 # extraction du corps de l'article, si disponible
            import trafilatura
            text = trafilatura.extract(str(markup) if not isinstance(markup, str) else markup) or ""
        except Exception:
            text = ""
    if use_blocks or len(text.split()) < 50:
        for t in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"]):
            t.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        text = html_blocks(root, on_img)
    ex.sections = [("", text)]
    html_tag = soup.find("html")
    ex.meta = {k: v for k, v in {
        "site": _meta(soup, "og:site_name"),
        "langue": (html_tag.get("lang", "") if html_tag else "")[:10],
        "description": _meta(soup, "og:description", "description")[:300],
    }.items() if v}
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


def read_pdf(path: Path, figures: FigureCollector | None = None) -> Extracted:
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
    if figures is not None:                  # images raster seulement ; position dans la page non conservée
        lines = []
        for no, pg in enumerate(r.pages, 1):
            try:
                images = list(pg.images)
                caps = [ln.strip() for ln in (pg.extract_text() or "").splitlines() if _CAPTION.match(ln.strip())]
            except Exception:
                continue
            for k, im in enumerate(images):
                try:
                    ph = figures.add(data=im.data, page=no,
                                     legende=caps[k] if len(caps) == len(images) else "")
                except Exception:
                    continue
                if ph:
                    lines.append(ph)
        if lines:
            ex.sections.append(("Figures", "\n".join(lines)))
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


_A_BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
_R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
_WP_DOCPR = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr"


def read_docx(path: Path, figures: FigureCollector | None = None) -> Extracted:
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
            if figures is not None:
                for blip in child.iter(_A_BLIP):
                    try:
                        part = d.part.related_parts[blip.get(_R_EMBED)]
                    except KeyError:
                        continue
                    pr = next(child.iter(_WP_DOCPR), None)
                    alt = ((pr.get("descr") or pr.get("title") or "") if pr is not None else "")
                    ph = figures.add(data=part.blob, alt=alt)
                    if ph:
                        sections[-1][1].append(ph)
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


def read_epub(path: Path, figures: FigureCollector | None = None) -> Extracted:
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
            on_img = None
            if figures is not None:
                def resolve(src, _dir=posixpath.dirname(name)):
                    if re.match(r"^[a-z][a-z0-9+.-]*:", src, re.I):
                        return None, None                    # lien externe : non suivi dans un EPUB
                    inner = posixpath.normpath(posixpath.join(_dir, unquote(src.split("#")[0].split("?")[0])))
                    try:
                        return z.read(inner), None
                    except KeyError:
                        return None, None
                on_img = make_on_img(figures, resolve)
            text = html_blocks(soup.body or soup, on_img)
            if len(text.split()) >= 20:
                ex.sections.append((title[:150], text))
    return ex


def read_text(path: Path, figures: FigureCollector | None = None) -> Extracted:
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


def read_html_file(path: Path, figures: FigureCollector | None = None) -> Extracted:
    root = path.parent.resolve()

    def resolve(src):
        if re.match(r"^[a-z][a-z0-9+.-]*:|^//", src, re.I):
            return None, None                                # pas de téléchargement pour un fichier local
        target = (root / unquote(src.split("#")[0].split("?")[0])).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            return None, None                                # jamais en dehors du dossier de la page
        return target.read_bytes(), None

    ex = extract_html(path.read_bytes(), None, figures, resolve if figures is not None else None)
    ex.title = _clean_title(ex.title, path.stem)
    return ex


READERS = {".pdf": read_pdf, ".docx": read_docx, ".epub": read_epub, ".md": read_text,
           ".txt": read_text, ".html": read_html_file, ".htm": read_html_file}


def fetch_url(url: str, figures: FigureCollector | None = None) -> Extracted:
    import requests
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    if "pdf" in r.headers.get("content-type", "").lower():
        raise ValueError("lien vers un PDF : téléchargez-le et placez-le dans le dossier du corpus")
    # les figures d'une page web sont référencées par leur adresse (aucun téléchargement d'images)
    resolve = (lambda src: (None, urljoin(url, src)) if urlparse(urljoin(url, src)).scheme in ("http", "https")
               else (None, None))
    ex = extract_html(r.content, url, figures, resolve if figures is not None else None)
    ex.title = ex.title or url
    return ex


# --- markdown reconstitué -------------------------------------------------------------

def to_markdown(title: str, sections: list) -> str:
    """Document complet en markdown : `# titre`, puis `## section` suivi du texte de chaque section."""
    parts = []
    clean = " ".join((title or "").split())
    if clean:
        parts.append(f"# {clean}")
    for sec_title, text in sections:
        sec = " ".join((sec_title or "").split())
        if sec:
            parts.append(f"## {sec}")
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts) + "\n"


def markdown_to_sections(md: str) -> list:
    """Inverse de `to_markdown` : retrouve [(titre de section, texte)] (le titre `# ...` est ignoré)."""
    sections, title, buf = [], "", []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((title, text))

    for k, line in enumerate(md.splitlines()):
        if k == 0 and line.startswith("# "):
            continue
        if line.startswith("## "):
            flush()
            title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    flush()
    return sections


def build_passages(doc_id: str, sections: list, words: int = 250, overlap: int = 40, figures: bool = False) -> list:
    """Passages d'un document. `figures=True` : les lignes `![légende](cible)` deviennent « Figure : légende »
    (ou disparaissent sans légende), pour que ni l'adresse ni le chemin ne soient indexés."""
    out = []
    for sec_title, text in sections:
        if figures:
            kept = []
            for ln in text.splitlines():
                fl = figure_passage_line(ln)
                if fl is None:
                    kept.append(ln)
                elif fl:
                    kept.append(fl)
            text = "\n".join(kept)
        for chunk in chunk_text(text, words, overlap):
            n = len(out)
            out.append({"id": f"{doc_id}-{n:04d}", "doc_id": doc_id, "position": n,
                        "section": sec_title, "text": chunk})
    return out


# --- assemblage ------------------------------------------------------------------------

def _add(docs, passages, source, dtype, ex: Extracted, words, overlap, warnings, resolver=None,
         tagger=None, keep_markdown=False, meta_extra=None, keep_figures=False, figures_dir=None):
    if resolver is not None:
        resolver(ex, source, dtype)
    if tagger is not None:
        tagger(ex, source, dtype)
    doc_id = doc_id_for(source)
    infos = _materialize(ex, doc_id, figures_dir) if keep_figures else []
    new = build_passages(doc_id, ex.sections, words, overlap, figures=keep_figures)
    n = len(new)
    passages.extend(new)
    for w in ex.warnings:
        warnings.append(f"{source} : {w}")
    if n == 0:
        warnings.append(f"{source} : aucun texte exploitable, document ignoré")
        return
    doc = {"id": doc_id, "title": ex.title or source, "type": dtype, "source": source,
           "author": ex.author, "date": ex.date, "date_origine": ex.date_origin if ex.date else "",
           "n_passages": n, "n_words": sum(len(p["text"].split()) for p in new)}
    if ex.themes:
        doc["themes"] = list(ex.themes)
    if keep_markdown:
        md = to_markdown(doc["title"], ex.sections)
        doc["markdown"] = md
        doc["content_sha256"] = hashlib.sha256(md.encode("utf-8")).hexdigest()
        doc["meta"] = {**(meta_extra or {}), **ex.meta}
    if infos:
        doc["figures"] = infos
    docs.append(doc)


def doc_id_for(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _file_meta(path: Path, rel: str, sha: str | None = None) -> dict:
    return {"fichier": rel, "extension": path.suffix.lower().lstrip("."),
            "taille_octets": path.stat().st_size, "sha256_fichier": sha or file_sha256(path)}


def _drop_figures(figures_dir, doc_id: str) -> None:
    """Supprime le dossier d'images d'un document (jamais autre chose qu'un dossier à 8 hexadécimaux)."""
    if figures_dir and re.fullmatch(r"[0-9a-f]{8}", doc_id):
        shutil.rmtree(Path(figures_dir) / doc_id, ignore_errors=True)


MANIFEST = "manifest.json"


def params_id(**params) -> str:
    """Empreinte des réglages qui changent le résultat de l'ingestion (extraction, découpage, dates, thèmes)."""
    return hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
                        .encode("utf-8")).hexdigest()[:12]


def load_previous(data_dir, pid: str) -> tuple[dict | None, str]:
    """Résultat de l'ingestion précédente (documents, passages, manifeste) et la raison d'un retraitement complet."""
    from .backend import read_neutral
    d = Path(data_dir)
    needed = [d / n for n in ("documents.json", "passages.jsonl", MANIFEST)]
    if not all(p.exists() for p in needed):
        return None, "première ingestion"
    try:
        manifest = json.loads(needed[2].read_text(encoding="utf-8"))
        docs, passages = read_neutral(d)
    except (OSError, ValueError):
        return None, "ingestion précédente illisible : retraitement complet"
    if manifest.get("params") != pid:
        return None, "réglages modifiés depuis la dernière ingestion : retraitement complet"
    return {"documents": docs, "passages": passages, "manifest": manifest}, ""


def write_manifest(data_dir, manifest: dict) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    (Path(data_dir) / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


def _finish_figures(ex: Extracted, coll: FigureCollector | None) -> None:
    if coll is None:
        return
    ex.figures = coll.items
    if coll.warning():
        ex.warnings.append(coll.warning())


def ingest_corpus(corpus_dir, urls_file=None, words=250, overlap=40, fetch=fetch_url, date_resolver=None,
                  theme_tagger=None, keep_markdown=False, keep_figures=False, figures_dir=None,
                  figures_min_px=120, figures_max_per_doc=40, previous=None, report=None, pid=""):
    """Retourne (documents, passages, avertissements).

    `keep_markdown` ajoute à chaque document le markdown complet, son empreinte et des métadonnées.
    `keep_figures` (avec keep_markdown) repère les figures : lignes `![légende](figures/...)` dans le
    markdown, fichiers image dans `figures_dir`, métadonnées dans `doc["figures"]`.

    Ingestion incrémentale : avec `previous` (voir `load_previous`), un fichier dont l'empreinte (SHA-256)
    n'a pas changé n'est PAS relu : son document, ses passages et son markdown sont repris tels quels
    (corrections manuelles de date comprises). Une adresse web déjà ingérée n'est pas retéléchargée.
    Seuls les fichiers nouveaux ou modifiés sont traités ; ceux qui ont disparu sont retirés.
    `report` (dict) reçoit : nouveaux, modifies, inchanges, supprimes, manifest.
    """
    keep_figures = bool(keep_figures and keep_markdown)
    root = Path(corpus_dir)
    docs, passages, warnings = [], [], []
    urls_path = Path(urls_file) if urls_file else root / "urls.txt"
    prev_docs = {d["source"]: d for d in previous["documents"]} if previous else {}
    prev_pass: dict[str, list] = {}
    for p in (previous["passages"] if previous else []):
        prev_pass.setdefault(p["doc_id"], []).append(p)
    known = previous["manifest"].get("sources", {}) if previous else {}
    rep = {"nouveaux": [], "modifies": [], "inchanges": 0, "supprimes": [], "manifest": {"params": pid, "sources": {}}}
    seen: set[str] = set()
    if previous is None and keep_figures and figures_dir and Path(figures_dir).is_dir():
        for sub in Path(figures_dir).iterdir():          # retraitement complet : on repart d'un dossier propre
            if sub.is_dir():
                _drop_figures(figures_dir, sub.name)

    def reusable(source: str, fp: str) -> bool:
        e, d = known.get(source), prev_docs.get(source)
        return bool(e and d and e.get("fingerprint") == fp and (not keep_markdown or "markdown" in d))

    def settle(source: str, fp: str, before: int, reused: bool) -> None:
        seen.add(source)
        if len(docs) > before:
            rep["manifest"]["sources"][source] = {"id": docs[-1]["id"], "fingerprint": fp}

    def classify(source: str, fp: str) -> None:
        if source in known:
            rep["modifies"].append(source)
            _drop_figures(figures_dir, doc_id_for(source))
        else:
            rep["nouveaux"].append(source)

    def carry(source: str) -> None:
        d = prev_docs[source]
        docs.append(d)
        passages.extend(prev_pass.get(d["id"], []))
        rep["inchanges"] += 1

    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.name.startswith(".") or path.resolve() == urls_path.resolve()
                or path.suffix.lower() not in FILE_TYPES):
            continue
        rel = path.relative_to(root).as_posix()
        fp = file_sha256(path)
        before = len(docs)
        if reusable(rel, fp):
            carry(rel)
            settle(rel, fp, before, True)
            continue
        classify(rel, fp)
        coll = FigureCollector(figures_min_px, figures_max_per_doc) if keep_figures else None
        try:
            ex = READERS[path.suffix.lower()](path, figures=coll) if coll else READERS[path.suffix.lower()](path)
        except Exception as e:
            warnings.append(f"{rel} : {type(e).__name__}: {e}")
            seen.add(rel)
            continue
        _finish_figures(ex, coll)
        _add(docs, passages, rel, FILE_TYPES[path.suffix.lower()], ex, words, overlap, warnings, date_resolver,
             theme_tagger, keep_markdown, _file_meta(path, rel, fp) if keep_markdown else None,
             keep_figures, figures_dir)
        settle(rel, fp, before, False)
    if urls_path.exists():
        for line in urls_path.read_text(encoding="utf-8").splitlines():
            url = line.split("#", 1)[0].strip()
            if not url or url in seen:
                continue
            before = len(docs)
            if reusable(url, "url"):
                carry(url)
                settle(url, "url", before, True)
                continue
            classify(url, "url")
            coll = FigureCollector(figures_min_px, figures_max_per_doc) if keep_figures else None
            try:
                ex = fetch(url, figures=coll) if coll else fetch(url)
            except Exception as e:
                warnings.append(f"{url} : {type(e).__name__}: {e}")
                seen.add(url)
                continue
            _finish_figures(ex, coll)
            _add(docs, passages, url, "web", ex, words, overlap, warnings, date_resolver,
                 theme_tagger, keep_markdown, {"url": url}, keep_figures, figures_dir)
            settle(url, "url", before, False)
    for source in known:
        if source not in seen:
            rep["supprimes"].append(source)
            _drop_figures(figures_dir, doc_id_for(source))
    if report is not None:
        report.update(rep)
    return docs, passages, warnings
