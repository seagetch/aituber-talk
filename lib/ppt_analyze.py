# -*- coding: utf-8 -*-
"""
Slide Analysis Agent – LangGraph Edition (Fixed Image Serialization)
====================================================================

Implements a multi-pass pipeline (Pass-0 → Pass-3) with dynamic routing via LangGraph.
Corrects image byte serialization by base64-encoding image data.

Requirements
------------
```bash
pip install langgraph python-pptx Pillow openai pydantic tqdm
```

Usage
-----
```bash
python slide_analysis_agent.py deck.pptx --out slides.jsonl --api-key sk-...
``` 
"""
from __future__ import annotations
import argparse
import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import openai
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, ValidationError
from pptx import Presentation
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMU_PER_INCH = 914400
DEFAULT_DPI = 220
# Maximum retries for JSON parsing
MAX_JSON_RETRIES = 3

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
SlideType = Literal[
    "text", "chart", "diagram", "image",
    "mixed", "cover", "thank_you", "appendix",
]
RhetoricalRole = Literal[
    "problem", "cause", "solution", "evidence",
    "benefit", "summary", "transition", "introduction", "other",
]

class VisualElement(BaseModel):
    type: str
    description: str
    extracted_text: List[str] = Field(default_factory=list)

class ChartSeries(BaseModel):
    series_name: str
    data_points: List[List[Any]]

class SlideInsight(BaseModel):
    slide_index: int
    slide_type: SlideType
    rhetorical_role: Optional[RhetoricalRole] = None
    title: Optional[str] = None
    subtitle: Optional[str] = None
    bullet_points: List[str] = Field(default_factory=list)
    paragraphs: List[str] = Field(default_factory=list)
    visual_elements: List[VisualElement] = Field(default_factory=list)
    chart_data: List[ChartSeries] = Field(default_factory=list)
    speaker_notes: Optional[str] = None
    emphasis_cues: List[str] = Field(default_factory=list)
    referred_terms: List[str] = Field(default_factory=list)
    slide_intent: Optional[str] = None
    key_takeaways: List[str] = Field(default_factory=list)
    prev_relation: Optional[str] = None
    next_relation: Optional[str] = None
    narration_draft: Optional[str] = None

    class Config:
        extra = "ignore"

# ---------------------------------------------------------------------------
# OpenAI helper
# ---------------------------------------------------------------------------
def get_openai_client(api_key: Optional[str]) -> None:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key missing. Use --api-key or set OPENAI_API_KEY.")
    openai.api_key = key

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SKELETON_PROMPT = (
    "You are a helpful assistant that extracts skeleton metadata from a slide. "
    "Critically analyze the attached slide image to identify all visual elements including shapes, text regions, charts, diagrams, and icons, then combine with any extracted PDF text. "
    "Return ONLY JSON with keys: "
    "slide_index (int), "
    "title (string|null), "
    "subtitle (string|null), "
    "slide_type (text|chart|diagram|image|mixed|cover|thank_you|appendix), "
    "rhetorical_role (problem|cause|solution|evidence|benefit|summary|transition|introduction|other|null), "
    "visual_elements (array of {type:string, description:string, extracted_text:string[]}), "
    "bullet_points (string[]), "
    "paragraphs (string[]), "
    "emphasis_cues (string[]), "
    "chart_data (array of {series_name:string, data_points:number[][]}), "
    "speaker_notes (string|null), "
    "referred_terms (string[]), "
    "slide_intent (string|null), "
    "key_takeaways (string[]), "
    "prev_relation (string|null), "
    "next_relation (string|null), "
    "narration_draft (string|null). "
    "If a field is not present, return null or an empty array as appropriate."
)

PROMPT_TEXT = (
    "You are analyzing a text-centric slide. Rigorously examine the slide image for visual elements (icons, highlights, layout) and integrate with the slide text."
    "Extract visual_elements, bullet_points, paragraphs, emphasis_cues."
    "Return ONLY JSON with the full SlideInsight schema as in SKELETON_PROMPT."
)

PROMPT_CHART = (
    "You are analyzing a chart slide. Carefully inspect the slide image to detect chart components (axes, bars, lines, legends) and read any chart labels, combining with the slide text."
    "Extract visual_elements and chart_data (series_name, data_points), plus bullet_points, paragraphs, and key_takeaways."
    "Return ONLY JSON with the full SlideInsight schema as in SKELETON_PROMPT."
)

PROMPT_DIAGRAM = (
    "You are analyzing a diagram slide. Thoroughly parse the slide image to map nodes, connections, and annotations, then merge with available text."
    "Extract visual_elements (nodes, arrows, labels), bullet_points, paragraphs, and slide_intent."
    "Return ONLY JSON with the full SlideInsight schema as in SKELETON_PROMPT."
)

PROMPT_IMAGE = (
    "You are analyzing an image-centric slide. Perform full visual analysis on the slide image to identify objects, scenes, text overlays, and inferred messages."
    "Extract visual_elements, slide_intent, and key_takeaways."
    "Return ONLY JSON with the full SlideInsight schema as in SKELETON_PROMPT."
)

PROMPTS_BY_TYPE = {
    "text": PROMPT_TEXT,
    "mixed": PROMPT_TEXT,
    "cover": PROMPT_TEXT,
    "thank_you": PROMPT_TEXT,
    "appendix": PROMPT_TEXT,
    "chart": PROMPT_CHART,
    "diagram": PROMPT_DIAGRAM,
    "image": PROMPT_IMAGE,
}

PROMPTS_BY_TYPE.update({"chart":PROMPT_CHART,"diagram":PROMPT_DIAGRAM,"image":PROMPT_IMAGE})

# ---------------------------------------------------------------------------
# Utils: PDF → images
# ---------------------------------------------------------------------------
def pdf_to_images(pdf_path: Path, dpi: int = DEFAULT_DPI) -> List[Path]:
    outdir = pdf_path.with_suffix("").with_name(pdf_path.stem + "_slides")
    outdir.mkdir(exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    imgs: List[Path] = []
    for i, page in enumerate(tqdm(pages, desc="Pass_0")):
        p = outdir / f"slide_{i:03}.png"
        page.save(p, "PNG")
        imgs.append(p)
    return imgs

# ---------------------------------------------------------------------------
# Utils: PPTX → images
# ---------------------------------------------------------------------------

    # Fallback: blank white images if conversion not available
    prs = Presentation(pptx_path)
    for i, _ in enumerate(tqdm(prs.slides, desc="Pass_0")):
        w_px = int(prs.slide_width / EMU_PER_INCH * dpi)
        h_px = int(prs.slide_height / EMU_PER_INCH * dpi)
        img = Image.new("RGB", (w_px, h_px), "white")
        p = outdir / f"slide_{i:03}.png"
        img.save(p, dpi=(dpi, dpi))
        imgs.append(p)
    return imgs

# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
def low_quality(data,slide_type):
    if slide_type=="chart":
        if not data.get("chart_data"):
            return True
        vis=data.get("visual_elements",[])
        if vis and len(vis[0].get("extracted_text",[]))<2:
            return True
    if slide_type=="text":
        if not data.get("bullet_points") and not data.get("paragraphs"):
            return True
    return False

def gate_quality(state):
    sk,data=state.enriched[-1]
    if low_quality(data,sk.slide_type):print(" Low quality")
    else:print(" Enough quality")
    return "needs_full" if low_quality(data,sk.slide_type) else "mini_ok"

# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------
@dataclass
class PipelineState:
    api_key:str
    pptx:Path
    images:List[Path]=field(default_factory=list)
    idx:int=0
    skeletons:List[SlideInsight]=field(default_factory=list)
    enriched:List[Tuple[SlideInsight,Dict[str,Any]]]=field(default_factory=list)
    final_slides:List[SlideInsight]=field(default_factory=list)
    outline:List[str]=field(default_factory=list)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def node_pass0(state):
    print(" Pass_0")
    # Load images from PDF pages instead of PPTX
    state.images = pdf_to_images(state.pptx)
    return state

def node_skeleton(state):
    print(" Skeleton")
    # Extract PDF text layer
    reader = PdfReader(str(state.pptx))
    try:
        page_text = reader.pages[state.idx].extract_text() or ""
    except Exception:
        page_text = ""
    get_openai_client(state.api_key)
    img_b64 = base64.b64encode(state.images[state.idx].read_bytes()).decode('ascii')
    # Define function schema for extracting skeleton
    fn_skel = {
        "name": "extract_skeleton",
        "description": "Extract slide skeleton metadata",
        "parameters": {
            "type": "object",
            "properties": {
                "slide_index": {"type": "integer"},
                "title": {"type": ["string","null"]},
                "subtitle": {"type": ["string","null"]},
                "slide_type": {"type": "string", "enum": ["text","chart","diagram","image","mixed","cover","thank_you","appendix"]},
                "rhetorical_role": {"type": ["string","null"], "enum": ["problem","cause","solution","evidence","benefit","summary","transition","introduction","other", None]},
                "visual_elements": {"type":"array","items":{"type":"object","properties":{"type":{"type":"string"},"description":{"type":"string"},"extracted_text":{"type":"array","items":{"type":"string"}}},"required":["type","description","extracted_text"]}},
                "bullet_points": {"type":"array","items":{"type":"string"}},
                "paragraphs": {"type":"array","items":{"type":"string"}},
                "emphasis_cues": {"type":"array","items":{"type":"string"}},
                "chart_data": {"type":"array","items":{"type":"object","properties":{"series_name":{"type":"string"},"data_points":{"type":"array","items":{"type":"array","items":{"type":"number"}}}},"required":["series_name","data_points"]}},
                "speaker_notes": {"type":["string","null"]},
                "referred_terms": {"type":"array","items":{"type":"string"}},
                "slide_intent": {"type":["string","null"]},
                "key_takeaways": {"type":"array","items":{"type":"string"}},
                "prev_relation": {"type":["string","null"]},
                "next_relation": {"type":["string","null"]},
                "narration_draft": {"type":["string","null"]}
            },
            "required": ["slide_index","slide_type"]
        }
    }
    # System and user messages
    sys_msg = {"role":"system","content":"Extract skeleton JSON"}
    user_msg = {"role":"user","content":SKELETON_PROMPT + f"PDF Text:{page_text}","image":{"data":img_b64}}
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[sys_msg, user_msg],
        functions=[fn_skel],
        function_call={"name":"extract_skeleton"},
        response_format={"type":"json_object"}
    )
    raw = resp.choices[0].message.function_call.arguments
    data = json.loads(raw)
    # Ensure list fields are not None
    for field in ["bullet_points","paragraphs","visual_elements","chart_data","emphasis_cues","referred_terms","key_takeaways"]:
        if data.get(field) is None:
            data[field] = []
    data.setdefault("slide_index", state.idx)
    try:
        ins = SlideInsight(**data)
    except ValidationError:
        data.setdefault("slide_type","mixed")
        ins = SlideInsight(**data)
    state.skeletons.append(ins)
    return state

def node_enrich_mini(state):
    print(" Enrich mini")
    # Extract PDF text layer
    reader = PdfReader(str(state.pptx))
    try:
        page_text = reader.pages[state.idx].extract_text() or ""
    except Exception:
        page_text = ""
    get_openai_client(state.api_key)
    sk = state.skeletons[-1]
    img_b64 = base64.b64encode(state.images[state.idx].read_bytes()).decode('ascii')
    # Define function schema for mini enrichment
    fn_mini = {
        "name": "enrich_slide",
        "description": "Extract detailed slide content",
        "parameters": {
            "type": "object",
            "properties": {
                "slide_index": {"type": "integer"},
                "visual_elements": {"type":"array","items":{"type":"object"}},
                "bullet_points": {"type":"array","items":{"type":"string"}},
                "paragraphs": {"type":"array","items":{"type":"string"}},
                "emphasis_cues": {"type":"array","items":{"type":"string"}},
                "slide_intent": {"type":["string","null"]},
                "key_takeaways": {"type":"array","items":{"type":"string"}}
            },
            "required": ["slide_index"]
        }
    }
    sys_msg = {"role":"system","content":"Enrich slide with main content"}
    user_msg = {"role":"user","content":PROMPTS_BY_TYPE[sk.slide_type] + f"PDF Text:{page_text}","image":{"data":img_b64}}
    # Call using function-calling
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[sys_msg, user_msg],
        functions=[fn_mini],
        function_call={"name":"enrich_slide"},
        response_format={"type":"json_object"}
    )
    raw = resp.choices[0].message.function_call.arguments
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(" JSON parse error in enrich_mini, using empty dict")
        data = {"slide_index": state.idx}
    data.setdefault("slide_index", state.idx)
    state.enriched.append((sk, data))
    return state

def node_enrich_full(state):
    print(" Enrich full")
    # Extract PDF text layer
    reader = PdfReader(str(state.pptx))
    try:
        page_text = reader.pages[state.idx].extract_text() or ""
    except Exception:
        page_text = ""
    get_openai_client(state.api_key)
    sk = state.skeletons[-1]
    img_b64 = base64.b64encode(state.images[state.idx].read_bytes()).decode('ascii')
    # Define function schema for full enrichment
    fn_full = {
        "name": "enrich_slide_full",
        "description": "Extract complete slide content including visuals and text",
        "parameters": {
            "type": "object",
            "properties": {
                "slide_index": {"type": "integer"},
                "visual_elements": {"type":"array","items":{"type":"object"}},
                "bullet_points": {"type":"array","items":{"type":"string"}},
                "paragraphs": {"type":"array","items":{"type":"string"}},
                "emphasis_cues": {"type":"array","items":{"type":"string"}},
                "chart_data": {"type":"array","items":{"type":"object"}},
                "speaker_notes": {"type":["string","null"]},
                "referred_terms": {"type":"array","items":{"type":"string"}},
                "slide_intent": {"type":["string","null"]},
                "key_takeaways": {"type":"array","items":{"type":"string"}}
            },
            "required": ["slide_index"]
        }
    }
    sys_msg = {"role":"system","content":"Fully enrich slide with all details"}
    user_msg = {"role":"user","content":PROMPTS_BY_TYPE[sk.slide_type] + f"PDF Text:{page_text}","image":{"data":img_b64}}
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[sys_msg, user_msg],
        functions=[fn_full],
        function_call={"name":"enrich_slide_full"},
        response_format={"type":"json_object"}
    )
    raw = resp.choices[0].message.function_call.arguments
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(" JSON parse error in enrich_full, using empty dict")
        data = {"slide_index": state.idx}
    data.setdefault("slide_index", state.idx)
    state.enriched[-1] = (sk, data)
    return state

def node_merge(state):
    print(" merge")
    sk,data=state.enriched[-1]
    merged=sk.dict();merged.update(data)
    try:
        slide=SlideInsight(**merged)
        print(f"  slide={slide}")
    except ValidationError as e:
        print(f" [err]{e}")
        merged.setdefault("slide_type","mixed")
        try:
            slide=SlideInsight(**merged)
        except Exception as e:
            print(f" [err]{e}")
            slide=sk
    state.final_slides.append(slide)
    return state

def node_advance(state):
    state.idx+=1;print(f"Page: {state.idx}")
    return state

def cond_more(state):return "iterate" if state.idx<len(state.images) else "done"

def node_refine(state):
    print(" Refine")
    get_openai_client(state.api_key)
    # Define full JSON schema for SlideInsight to enforce all fields
    schema_slide = {
        "type": "object",
        "properties": {
            "slide_index": {"type": "integer"},
            "slide_type": {"type": "string", "enum": ["text","chart","diagram","image","mixed","cover","thank_you","appendix"]},
            "rhetorical_role": {"type": ["string","null"], "enum": ["problem","cause","solution","evidence","benefit","summary","transition","other", "null"]},
            "title": {"type": ["string","null"]},
            "subtitle": {"type": ["string","null"]},
            "bullet_points": {"type": "array", "items": {"type": "string"}},
            "paragraphs": {"type": "array", "items": {"type": "string"}},
            "visual_elements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                        "extracted_text": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["type","description","extracted_text"]
                }
            },
            "chart_data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "series_name": {"type": "string"},
                        "data_points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
                    },
                    "required": ["series_name","data_points"]
                }
            },
            "speaker_notes": {"type": ["string","null"]},
            "emphasis_cues": {"type": "array", "items": {"type": "string"}},
            "referred_terms": {"type": "array", "items": {"type": "string"}},
            "slide_intent": {"type": ["string","null"]},
            "key_takeaways": {"type": "array", "items": {"type": "string"}},
            "prev_relation": {"type": ["string","null"]},
            "next_relation": {"type": ["string","null"]},
            "narration_draft": {"type": ["string","null"]}
        },
        "required": ["slide_index","slide_type","bullet_points","paragraphs","visual_elements","chart_data","emphasis_cues","referred_terms","key_takeaways"]
    }
    fn_def = {
        "name": "refine_slides",
        "description": "Refine slides and enforce full SlideInsight schema",
        "parameters": {
            "type": "object",
            "properties": {
                "slides": {"type": "array", "items": schema_slide},
                "deck_outline": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["slides"]
        }
    }
    sys_msg = {"role": "system", "content": "Refine slides and return ONLY valid JSON with keys 'slides' (array of slide objects) and 'deck_outline' (array of strings)."}
    user_msg = {"role": "user", "content": "Please refine and return JSON."}
    out = None
    for i in range(MAX_JSON_RETRIES):
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[sys_msg, user_msg],
            functions=[fn_def],
            function_call={"name": "refine_slides"},
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.function_call.arguments
        try:
            out = json.loads(raw)
            break
        except json.JSONDecodeError:
            print(f" JSON parse error in refine, retry {i+1}/{MAX_JSON_RETRIES}")
    if out is None:
        print(" JSON parse failed in refine after retries, using fallback outline and slides")
        out = {"slides": [s.dict() for s in state.final_slides], "deck_outline": state.outline}
    state.final_slides = [SlideInsight(**s) for s in out["slides"]]
    state.outline = out.get("deck_outline", [])
    return state
# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    g=StateGraph(PipelineState)
    g.set_entry_point("pass0")
    nodes=[("pass0",node_pass0),("skeleton",node_skeleton),("enrich_mini",node_enrich_mini),("enrich_full",node_enrich_full),("merge",node_merge),("advance",node_advance),("refine",node_refine)]
    for name,fn in nodes: g.add_node(name,fn)
    edges=[("pass0","skeleton"),("skeleton","enrich_mini"),("enrich_mini","needs_full","enrich_full"),("enrich_mini","mini_ok","merge"),("enrich_full","merge"),("merge","advance"),("advance","iterate","skeleton"),("advance","done","refine"),("refine","END",None)]
    g.add_edge("pass0","skeleton");
    g.add_edge("skeleton","enrich_mini");
    g.add_conditional_edges("enrich_mini",gate_quality,{"needs_full":"enrich_full","mini_ok":"merge"});
    g.add_edge("enrich_full","merge");
    g.add_edge("merge","advance");
    g.add_conditional_edges("advance",cond_more,{"iterate":"skeleton","done":"refine"});
    g.add_edge("refine",END)
    # Compile graph with increased recursion limit to avoid GraphRecursionError
    return g.compile()
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser=argparse.ArgumentParser(description="Slide Analysis Agent – LangGraph Edition")
    parser.add_argument("pptx", type=Path, help="Path to PDF export of PowerPoint (.pdf)")
    parser.add_argument("--out",type=Path,default=Path("slides.jsonl"),help="Output JSONL path")
    parser.add_argument("--api-key",type=str,help="OpenAI API key (optional)")
    args=parser.parse_args()
    state=PipelineState(api_key=args.api_key or os.getenv("OPENAI_API_KEY",""),pptx=args.pptx)
    graph=build_graph()
    first_run = graph.invoke(state, config={"recursion_limit": 1000})
    final_state = first_run
    print("\nDeck Outline:")
    for i,sec in enumerate(state.outline,1):print(f"  {i}. {sec}")
    print(f"\nWriting slides to {args.out}")
    with open(args.out,"w",encoding="utf-8")as f:
        for s in state.final_slides:f.write(json.dumps(s.dict(),ensure_ascii=False)+"\n")

if __name__=="__main__":main()
