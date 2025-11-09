import os
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import typer
from rich import print as rprint
from jinja2 import Environment, FileSystemLoader, select_autoescape

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from PIL import Image, ImageChops, ImageFilter, ImageOps
import urllib.parse as _urlparse

# Optional OpenAI
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# Optional dotenv (.env) loader
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional YAML loader for batch scenarios
try:  # pragma: no cover
    import yaml
except Exception:
    yaml = None  # type: ignore

app = typer.Typer(add_completion=False)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def capture_page(page, url: str, out_dir: Path, name: str, wait_ms: int = 500) -> Tuple[Path, Path]:
    page.goto(url)
    page.wait_for_timeout(wait_ms)
    screenshot_path = out_dir / f"{name}.png"
    dom_path = out_dir / f"{name}.html"
    page.screenshot(path=str(screenshot_path), full_page=True)
    html = page.content()
    dom_path.write_text(html, encoding="utf-8")
    return screenshot_path, dom_path


def visual_diff(pre_img: Path, post_img: Path, out_path: Path) -> None:
    a = Image.open(pre_img).convert("RGB")
    b = Image.open(post_img).convert("RGB")
    # Resize to smallest common size
    w = min(a.width, b.width)
    h = min(a.height, b.height)
    a = a.resize((w, h))
    b = b.resize((w, h))
    # Blur to reduce noise
    a_blur = a.convert("L").filter(ImageFilter.GaussianBlur(radius=2))
    b_blur = b.convert("L").filter(ImageFilter.GaussianBlur(radius=2))
    # Absolute difference
    diff = ImageChops.difference(a_blur, b_blur)
    # Enhance and colorize as heatmap
    diff = ImageOps.autocontrast(diff)
    heat = ImageOps.colorize(diff, black="#000000", white="#ff0000")
    # Blend heatmap over post image
    heat_rgba = heat.convert("RGBA")
    b_rgba = b.convert("RGBA")
    blended = Image.blend(b_rgba, heat_rgba, alpha=0.4)
    blended.convert("RGB").save(out_path)


def dom_diff_summary(pre_html: Path, post_html: Path) -> str:
    pre = BeautifulSoup(pre_html.read_text(encoding="utf-8"), "lxml")
    post = BeautifulSoup(post_html.read_text(encoding="utf-8"), "lxml")
    # Simple summaries
    pre_text = pre.get_text(" ").strip()
    post_text = post.get_text(" ").strip()
    len_delta = len(post_text) - len(pre_text)
    pre_links = len(pre.find_all("a"))
    post_links = len(post.find_all("a"))
    pre_buttons = len(pre.find_all("button"))
    post_buttons = len(post.find_all("button"))

    lines = [
        f"Text length delta: {len_delta:+d}",
        f"Links: {pre_links} -> {post_links}",
        f"Buttons: {pre_buttons} -> {post_buttons}",
    ]

    # Title change
    pre_title = pre.title.text if pre.title else ""
    post_title = post.title.text if post.title else ""
    if pre_title != post_title:
        lines.append(f"Title changed: '{pre_title}' -> '{post_title}'")

    # Added elements by simple tag counts
    for tag in ["h1", "h2", "table", "thead", "tbody", "th", "td", "span", "div"]:
        c1 = len(pre.find_all(tag))
        c2 = len(post.find_all(tag))
        if c1 != c2:
            lines.append(f"<{tag}> count: {c1} -> {c2}")

    return "\n".join(lines)


def extract_dom_context(pre_html: Path, post_html: Path) -> Dict[str, Any]:
    """Collect small, high-signal DOM facts to ground the LLM.
    - Titles, headers, button labels with aria-label, table headers, known app selectors.
    """
    pre = BeautifulSoup(pre_html.read_text(encoding="utf-8"), "lxml")
    post = BeautifulSoup(post_html.read_text(encoding="utf-8"), "lxml")

    def buttons(bs):
        out = []
        for b in bs.find_all("button"):
            label = b.get("aria-label") or ""
            text = (b.get_text(" ") or "").strip()
            out.append({"aria_label": label, "text": text})
        return out

    def headers(bs):
        out = []
        for h in bs.find_all(["h1", "h2", "h3"]):
            out.append({"tag": h.name, "text": (h.get_text(" ") or "").strip()})
        return out

    def table_headers(bs):
        return [ (th.get_text(" ").strip() or "") for th in bs.find_all("th") ]

    def exists_selector(bs, selector):
        try:
            return bool(bs.select(selector))
        except Exception:
            return False

    candidates = [
        "[aria-label=cta]",
        "[aria-label=incidents-chip]",
        "[aria-label=products-table]",
    ]

    return {
        "pre": {
            "title": pre.title.text if pre.title else "",
            "headers": headers(pre),
            "buttons": buttons(pre),
            "table_headers": table_headers(pre),
            "exists": {sel: exists_selector(pre, sel) for sel in candidates},
        },
        "post": {
            "title": post.title.text if post.title else "",
            "headers": headers(post),
            "buttons": buttons(post),
            "table_headers": table_headers(post),
            "exists": {sel: exists_selector(post, sel) for sel in candidates},
        },
        "candidates": candidates,
    }


def render_report(template_dir: Path, out_dir: Path, context: Dict[str, Any]) -> Path:
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=select_autoescape(["html"]))
    tpl = env.get_template("report.html.j2")
    html = tpl.render(**context)
    out_path = out_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def run_llm(provider: str, model: str, meta: Dict[str, Any], dom_summary: str) -> Dict[str, Any]:
    if provider != "openai":
        return {"functional_change": None, "test_ideas": [], "risks": []}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return {"functional_change": None, "test_ideas": [], "risks": []}
    client = OpenAI(api_key=api_key)

    def chat(prompt: str, response_format: Optional[Dict[str, Any]] = None) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a senior QA engineer assisting exploratory testing."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format=response_format if response_format else None,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"LLM error: {e}"

    fc_prompt = (
        "Task: Summarize the specific functional change between two UI variants.\n"
        "Constraints:\n"
        "- Ground strictly in provided facts. Do not invent features.\n"
        "- Reference concrete UI elements (headers, buttons, table columns).\n"
        "Inputs follow as JSON.\n"
        f"Inputs: {json.dumps(meta)}\nDOM summary: {dom_summary}\n"
        "Output: 3-6 sentences, no bullets."
    )
    functional_change = chat(fc_prompt)

    ideas_prompt = (
        "Task: Propose focused exploratory tests tied to observed diffs.\n"
        "Constraints:\n"
        "- 5-8 ideas. Each MUST cite a selector from candidates or an exact header/button text from context.\n"
        "- Steps must be specific to this page.\n"
        "- Expected and oracles must be verifiable (status text, aria attributes, column order).\n"
        "- Reject generic items. If insufficient evidence, output an empty list.\n"
        "Inputs: JSON context is provided.\n"
        "Return strict JSON object: {\n  \"ideas\": [ {\n    \"title\": str, \n    \"selector\": str, \n    \"steps\": [str], \n    \"expected\": str, \n    \"oracles\": [str], \n    \"risk\": str\n  } ]\n}"
    )
    ideas_text = chat(ideas_prompt + "\nContext: " + json.dumps(meta.get("dom_ctx", {})), response_format={"type": "json_object"})

    risks_prompt = (
        "Task: List concrete risks & regression hotspots tied to observed diffs.\n"
        "Constraints:\n"
        "- 4-8 items. Each MUST reference a selector from candidates or an exact column/button/header text from context.\n"
        "- No generic SDLC items.\n"
        "Return strict JSON object: {\n  \"risks\": [str], \n  \"regression_hotspots\": [str]\n}"
    )
    risks_text = chat(risks_prompt + "\nContext: " + json.dumps(meta.get("dom_ctx", {})), response_format={"type": "json_object"})

    # Keep free-form for now; template will display text or lists if parsed.
    ideas: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(ideas_text)
        if isinstance(parsed, dict):
            for key in ("ideas", "tests", "suggestions", "items"):
                if isinstance(parsed.get(key), list):
                    ideas = parsed.get(key, [])
                    break
        elif isinstance(parsed, list):
            # legacy fallback if model ignored wrapper
            ideas = parsed
    except Exception:
        # last resort: line-based fallback
        lines = [ln.strip("- •\t ") for ln in (ideas_text or "").splitlines() if ln.strip()]
        for ln in lines:
            ideas.append({"title": ln, "selector": "", "steps": [], "expected": "", "oracles": [], "risk": ""})

    risks: List[str] = []
    regression_hotspots: List[str] = []
    try:
        parsed_r = json.loads(risks_text)
        if isinstance(parsed_r, dict):
            risks = parsed_r.get("risks", []) or []
            regression_hotspots = parsed_r.get("regression_hotspots", []) or []
        elif isinstance(parsed_r, list):
            risks = parsed_r
    except Exception:
        # fallback: split lines
        risks = [ln.strip("- •\t ") for ln in (risks_text or "").splitlines() if ln.strip()]

    return {
        "functional_change": functional_change,
        "test_ideas": ideas,
        "risks": risks,
        "regression_hotspots": regression_hotspots,
        "raw_ideas_text": ideas_text,
        "raw_risks_text": risks_text,
    }


def run_once(
    pre_url: str,
    post_url: str,
    out: Path,
    provider: str,
    model: str,
    viewport: str,
    headless: bool,
    scenario: str = "",
) -> Path:
    ensure_dir(out)
    artifacts = out / f"run_{int(time.time())}"
    ensure_dir(artifacts)

    w, h = [int(x) for x in viewport.split(",")]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(viewport={"width": w, "height": h})
        page = ctx.new_page()
        rprint(f"[bold]Capturing pre:[/bold] {pre_url}")
        pre_png, pre_html = capture_page(page, pre_url, artifacts, "pre")
        rprint(f"[bold]Capturing post:[/bold] {post_url}")
        post_png, post_html = capture_page(page, post_url, artifacts, "post")
        ctx.close()
        browser.close()

    vis_path = artifacts / "visual_diff.png"
    visual_diff(pre_png, post_png, vis_path)

    dom_summary = dom_diff_summary(pre_html, post_html)
    dom_ctx = extract_dom_context(pre_html, post_html)

    def _path(url: str) -> str:
        try:
            return _urlparse.urlparse(url).path or "/"
        except Exception:
            return ""

    meta = {
        "pre_url": pre_url,
        "post_url": post_url,
        "viewport": viewport,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dom_ctx": dom_ctx,
        "scenario": scenario,
        "path": {"pre": _path(pre_url), "post": _path(post_url)},
    }

    llm_out = run_llm(provider, model, meta, dom_summary)
    # Write raw LLM outputs for debugging
    try:
        (artifacts / "llm_ideas_raw.txt").write_text(llm_out.get("raw_ideas_text", ""), encoding="utf-8")
        (artifacts / "llm_risks_raw.txt").write_text(llm_out.get("raw_risks_text", ""), encoding="utf-8")
    except Exception:
        pass

    # Paths must be relative to the artifacts directory where report.html is saved
    context = {
        "meta": meta,
        "paths": {
            "screenshot_pre": str(pre_png.relative_to(artifacts)),
            "screenshot_post": str(post_png.relative_to(artifacts)),
            "visual_diff": str(vis_path.relative_to(artifacts)),
        },
        "dom": {"summary": dom_summary},
        "llm": llm_out,
    }

    report_path = render_report(Path(__file__).parent / "templates", artifacts, context)
    rprint(f"[green]Report generated:[/green] {report_path}")
    return report_path


@app.command()
def main(
    pre_url: str = typer.Option(..., help="Pre-change URL (variant A)"),
    post_url: str = typer.Option(..., help="Post-change URL (variant B)"),
    out: Path = typer.Option(Path("reports"), help="Output directory"),
    provider: str = typer.Option("openai", help="LLM provider: openai or none"),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI model name"),
    viewport: str = typer.Option("1280,800", help="Viewport WxH, e.g., 1280,800"),
    headless: bool = typer.Option(True, help="Headless browser"),
    scenario: str = typer.Option("", help="Scenario name (e.g., home/products)"),
):
    run_once(pre_url, post_url, out, provider, model, viewport, headless, scenario=scenario)


@app.command()
def batch(
    scenario_file: Path = typer.Option(..., help="YAML file with runs: [{name, pre_url, post_url}]"),
    out: Path = typer.Option(Path("reports"), help="Output directory for all runs"),
    provider: str = typer.Option("openai", help="LLM provider: openai or none"),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI model name"),
    viewport: str = typer.Option("1280,800", help="Viewport WxH, e.g., 1280,800"),
    headless: bool = typer.Option(True, help="Headless browser"),
):
    if yaml is None:
        raise typer.Exit("PyYAML not installed. Run: pip install PyYAML")
    data = yaml.safe_load(scenario_file.read_text(encoding="utf-8"))
    runs = data.get("runs") if isinstance(data, dict) else None
    if not runs:
        raise typer.Exit("Scenario file must contain 'runs: - {name, pre_url, post_url}'")
    index_lines = ["<html><body><h1>Exploratory Runs</h1><ul>"]
    for item in runs:
        name = item.get("name") or "run"
        pre = item.get("pre_url")
        post = item.get("post_url")
        if not pre or not post:
            rprint(f"[yellow]Skipping invalid run entry (missing URLs):[/yellow] {item}")
            continue
        rpt = run_once(pre, post, out, provider, model, viewport, headless, scenario=name)
        rel = Path(rpt).relative_to(out)
        index_lines.append(f"<li><a href='{rel.as_posix()}'>{name}</a></li>")
    index_lines.append("</ul></body></html>")
    (out / "index.html").write_text("\n".join(index_lines), encoding="utf-8")
    rprint(f"[green]Index generated:[/green] {(out / 'index.html')}")


if __name__ == "__main__":
    app()
