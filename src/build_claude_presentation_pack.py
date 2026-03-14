from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MAX_MARKDOWN_CHARS = 1200
MAX_CODE_LINES = 25
MAX_OUTPUT_CHARS = 700


def read_notebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_source(source: Any) -> str:
    if isinstance(source, list):
        text = "".join(source)
    else:
        text = str(source or "")
    return text.strip()


def trim_markdown(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= MAX_MARKDOWN_CHARS:
        return cleaned
    return cleaned[:MAX_MARKDOWN_CHARS].rstrip() + "\n..."


def trim_code(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= MAX_CODE_LINES:
        return text.rstrip()
    return "\n".join(lines[:MAX_CODE_LINES]).rstrip() + "\n# ..."


def trim_output(text: str) -> str:
    output = text.strip()
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    return output[:MAX_OUTPUT_CHARS].rstrip() + "\n..."


def extract_output_text(outputs: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    image_seen = False
    for output in outputs:
        output_type = output.get("output_type", "")
        if output_type == "stream":
            text = output.get("text", "")
            chunks.append(normalize_source(text))
        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            plain = data.get("text/plain")
            if plain:
                chunks.append(normalize_source(plain))
            if "image/png" in data:
                image_seen = True
        elif output_type == "error":
            traceback = output.get("traceback", [])
            if traceback:
                chunks.append(normalize_source(traceback))

    merged = "\n\n".join(chunk for chunk in chunks if chunk)
    if image_seen:
        merged = (merged + "\n\n" if merged else "") + "[Notebook contains chart/image output.]"
    return trim_output(merged)


def notebook_section(notebook_path: Path) -> str:
    notebook = read_notebook(notebook_path)
    title = notebook_path.stem.replace("_", " ")
    lines: list[str] = [f"## {title}", ""]

    markdown_blocks: list[str] = []
    code_blocks: list[str] = []
    output_blocks: list[str] = []

    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        source_text = normalize_source(cell.get("source", ""))

        if cell_type == "markdown" and source_text:
            markdown_blocks.append(trim_markdown(source_text))

        if cell_type == "code" and source_text:
            code_blocks.append(trim_code(source_text))
            outputs = cell.get("outputs", [])
            if outputs:
                output_text = extract_output_text(outputs)
                if output_text:
                    output_blocks.append(output_text)

    lines.append("### Key Notes")
    lines.append("")
    if markdown_blocks:
        for block in markdown_blocks[:10]:
            lines.append(f"- {block.replace(chr(10), ' ')}")
    else:
        lines.append("- No markdown narrative found.")

    lines.append("")
    lines.append("### Representative Code")
    lines.append("")
    if code_blocks:
        for block in code_blocks[:6]:
            lines.append("```python")
            lines.append(block)
            lines.append("```")
            lines.append("")
    else:
        lines.append("No code cells found.")
        lines.append("")

    lines.append("### Key Output Signals")
    lines.append("")
    if output_blocks:
        for block in output_blocks[:10]:
            lines.append("```")
            lines.append(block)
            lines.append("```")
            lines.append("")
    else:
        lines.append("- No saved outputs detected.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_prompt_template(notebook_names: list[str]) -> str:
    notebooks_text = "\n".join(f"- {name}" for name in notebook_names)
    return f"""# Claude Prompt Template (Copy/Paste)

You are a senior data science communicator. Create a presentation deck from the provided project brief.

Requirements:
- Audience: business + technical stakeholders.
- Tone: clear, concise, executive-friendly.
- Slide count: 10-14 slides.
- Include: objective, data overview, cleaning approach, key EDA findings, bivariate insights, feature engineering decisions, risks/limitations, recommendations, and next steps.
- For each slide provide:
  1) Slide title
  2) 3-5 bullet points
  3) Suggested visual/chart
  4) Speaker notes (short paragraph)

Project notebooks included:
{notebooks_text}

Use the brief below as the source of truth. If information is missing, state assumptions explicitly.
"""


def build_main_document(notebooks: list[Path]) -> str:
    sections = [
        "# Churn Model Project Brief for Claude",
        "",
        "Use this document as source material to generate a presentation.",
        "",
        "## Suggested Storyline",
        "",
        "1. Business problem and objective",
        "2. Data source and dataset structure",
        "3. Data quality and cleaning decisions",
        "4. Univariate insights",
        "5. Bivariate insights",
        "6. Correlation and feature engineering",
        "7. Modeling readiness and risks",
        "8. Recommendations and next steps",
        "",
        "## Notebook Evidence",
        "",
    ]
    for notebook_path in notebooks:
        sections.append(notebook_section(notebook_path))
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebooks_dir = project_root / "notebooks"
    output_dir = project_root / "presentation"
    output_dir.mkdir(exist_ok=True)

    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No notebooks found in ./notebooks")

    main_doc = build_main_document(notebooks)
    prompt_doc = build_prompt_template([path.name for path in notebooks])

    main_path = output_dir / "claude_presentation_brief.md"
    prompt_path = output_dir / "claude_prompt_template.md"

    main_path.write_text(main_doc, encoding="utf-8")
    prompt_path.write_text(prompt_doc, encoding="utf-8")

    print(f"Created: {main_path}")
    print(f"Created: {prompt_path}")


if __name__ == "__main__":
    main()