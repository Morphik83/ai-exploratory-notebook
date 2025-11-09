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

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# Optional OpenAI
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

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
    a = cv2.imread(str(pre_img))
    b = cv2.imread(str(post_img))
    if a is None or b is None:
        raise RuntimeError("Failed to read screenshots for diff.")
    # Resize to smallest common size
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h))
    b = cv2.resize(b, (w, h))
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(a_gray, b_gray, full=True)
    diff = (1 - diff)  # higher means more different
    diff_norm = (255 * (diff / diff.max())).astype("uint8") if diff.max() > 0 else diff.astype("uint8")
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(b, 0.7, diff_color, 0.6, 0)
    cv2.imwrite(str(out_path), blend)


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

    def chat(prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a senior QA engineer assisting exploratory testing."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"LLM error: {e}"

    fc_prompt = (
        "Summarize the likely functional change between two UI variants based on the metadata and DOM summary.\n"
        f"Metadata: {json.dumps(meta)}\nDOM summary: {dom_summary}\n"
        "Provide a concise 1-2 paragraph summary."
    )
    functional_change = chat(fc_prompt)

    ideas_prompt = (
        "Based on the change, propose 6-10 high-value exploratory test ideas. "
        "For each, include: title; 3-6 steps; expected result; oracles; risk area. "
        "Return as concise bullet points."
    )
    ideas_text = chat(ideas_prompt)

    risks_prompt = (
        "List concrete risks and regression hotspots (selectors/modules) given the change. Return 5-8 bullets."
    )
    risks_text = chat(risks_prompt)

    # Keep free-form for now; template will display text or lists if parsed.
    ideas = []
    if ideas_text:
        ideas = [{"title": line.strip(), "steps": [], "expected": "", "oracles": [], "risk": ""}
                 for line in ideas_text.split("\n") if line.strip()]
    risks = [line.strip() for line in risks_text.split("\n") if line.strip()]

    return {
        "functional_change": functional_change,
        "test_ideas": ideas,
        "risks": risks,
    }


@app.command()
def main(
    pre_url: str = typer.Option(..., help="Pre-change URL (variant A)"),
    post_url: str = typer.Option(..., help="Post-change URL (variant B)"),
    out: Path = typer.Option(Path("reports"), help="Output directory"),
    provider: str = typer.Option("openai", help="LLM provider: openai or none"),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI model name"),
    viewport: str = typer.Option("1280,800", help="Viewport WxH, e.g., 1280,800"),
    headless: bool = typer.Option(True, help="Headless browser"),
):
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

    meta = {
        "pre_url": pre_url,
        "post_url": post_url,
        "viewport": viewport,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    llm_out = run_llm(provider, model, meta, dom_summary)

    context = {
        "meta": meta,
        "paths": {
            "screenshot_pre": str(pre_png.relative_to(out)),
            "screenshot_post": str(post_png.relative_to(out)),
            "visual_diff": str(vis_path.relative_to(out)),
        },
        "dom": {"summary": dom_summary},
        "llm": llm_out,
    }

    report_path = render_report(Path(__file__).parent / "templates", artifacts, context)
    rprint(f"[green]Report generated:[/green] {report_path}")


if __name__ == "__main__":
    app()
