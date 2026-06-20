<div align="center">
  
<a href="https://vectify.ai/pageindex" target="_blank">
  <img src="https://github.com/user-attachments/assets/46201e72-675b-43bc-bfbd-081cc6b65a1d" alt="PageIndex Banner" />
</a>

<br/>
<br/>

<p align="center">
  <a href="https://trendshift.io/repositories/14736" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14736" alt="VectifyAI%2FPageIndex | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# PageIndex: Vectorless, Reasoning-based RAG

<p align="center"><b>Reasoning-based RAG&nbsp; ◦ &nbsp;No Vector DB, No Chunking&nbsp; ◦ &nbsp;Context-Aware Retrieval&nbsp; ◦ &nbsp;Human-like</b></p>

<h4 align="center">
  <a href="https://vectify.ai">🌐 Website</a>&nbsp; • &nbsp;
  <a href="https://chat.pageindex.ai">🖥️ Chat Platform</a>&nbsp; • &nbsp;
  <a href="https://pageindex.ai/developer">🔌 MCP & API</a>&nbsp; • &nbsp;
  <a href="https://docs.pageindex.ai">📖 Docs</a>&nbsp; • &nbsp;
  <a href="https://discord.com/invite/VuXuf29EUj">💬 Discord</a>&nbsp; • &nbsp;
  <a href="https://ii2abc2jejf.typeform.com/to/tK3AXl8T">✉️ Contact</a>&nbsp;
</h4>
  
</div>


---

# 🐙 GitIndex — Vectorless Repo Intelligence (built on PageIndex)

> **This repository = PageIndex (base engine) + GitIndex (new feature built on top).**
>
> [**PageIndex**](#-introduction-to-pageindex) — documented in full below — is the base code: it turns a **document** into a reasoning-friendly **tree** and lets an LLM agent navigate it, with **no vectors and no chunking**. **GitIndex** is a feature built on that *same* architecture, but pointed at **GitHub** instead of PDFs. It indexes **entire repositories (code), issues & PRs, and whole orgs (100+ repos)** into the same kind of tree, then lets an agent reason over it to answer questions.

GitIndex reuses PageIndex's retrieval contract verbatim — a tree of `{title, node_id, summary, …}` nodes plus the agent tools `get_document`, `get_document_structure`, and `get_node_content`. Only the **tree-building** and **content-addressing** layers are GitHub-specific; the reasoning/retrieval layer is unchanged from PageIndex.

### What GitIndex indexes

| Source | Tree shape | Example question |
|---|---|---|
| **Repo (code)** | `dir → file → symbol` (functions / classes / methods) | *"Where is the entry point? How does auth work?"* |
| **Issues & PRs** | `tracker → state → item` (title + body + comments) | *"What open bugs mention retries?"* |
| **Org / user** | `corpus → language → repo` (lazy, scales to 100+ repos) | *"Which repos touch the payment service?"* |

### 🤔 RAG vs. GitIndex — what's the difference?

Traditional code RAG **splits files into fixed chunks, embeds them, and runs similarity search** — discarding the one thing code has in abundance: **structure**. GitIndex keeps the structure and **reasons** over it.

| | Vector RAG (typical) | **GitIndex (vectorless)** |
|---|---|---|
| Retrieval unit | Fixed-size text chunks | Real code units: dir / file / **symbol** |
| Index | Embeddings in a vector DB | **Hierarchical tree** of cheap summaries |
| Retrieval method | Top-k cosine similarity | **Agent reasoning** + tree search |
| Notion of "relevance" | Similarity (an approximation) | Traceable node selection, with reasons |
| Updates | Re-embed changed chunks | **Merkle diff** on git blob OIDs |
| Cross-file questions | Hard (chunks are isolated) | Native (the agent walks the tree) |
| Cost at scale | Store + query every vector | Summaries **de-duplicated by content hash** |

> **similarity ≠ relevance.** For code, what's *relevant* depends on call graphs, file layout, and intent — which needs **reasoning over structure**, not nearest-neighbor vectors. This is the PageIndex thesis, applied to repositories.

### 🏗️ Architecture

```mermaid
flowchart TD
  A[GitHub GraphQL v4] -->|batched tree walk + blob OIDs| B[Tree builder]
  B --> C{Source type}
  C -->|repo| C1[dir → file → symbol<br/>ast / tree-sitter]
  C -->|issues / PRs| C2[tracker → state → item]
  C -->|org| C3[corpus → language → repo]
  C1 --> D[Cheap Merkle summarizer]
  C2 --> D
  C3 --> D
  D -->|content-addressed cache| E[(PageIndex tree:<br/>title · node_id · summary · text)]
  E --> F[Agent tools:<br/>get_document / get_document_structure / get_node_content]
  F --> G[LLM agent via LiteLLM<br/>OpenAI · Anthropic · GitHub Copilot]
  G -->|reasons, then fetches only the needed nodes| E
```

**The cheap-summarization trick (think like a DSA programmer).** Summarizing every file across 100+ repos with an LLM would be slow and expensive. GitIndex treats git as the **Merkle tree it already is**: every file's blob **OID is a content hash**, so summaries are **content-addressed and de-duplicated** across all repos. Four tiers keep cost near-zero:

1. **Tier 0 — extractive (no LLM):** signatures, docstrings, README headers, issue titles.
2. **Tier 1 — memoized:** an OID → summary cache, so identical files are summarized once, ever.
3. **Tier 2 — batched LLM:** leftover leaves are bin-packed into a few cheap calls.
4. **Tier 3 — bottom-up reduction:** parent summaries are composed from their children.

**Incremental refresh** is just a Merkle diff: only files whose **blob OID changed** are re-fetched, re-parsed, and re-summarized.

### ⚙️ Setup

```bash
# 1) Install PageIndex deps + the agent runtime
pip3 install -r requirements.txt
pip3 install "openai-agents[litellm]"

# 2) A read-only GitHub token for the GraphQL API
export GITHUB_TOKEN=ghp_xxx          # or GH_TOKEN

# 3) Choose your LLM in pageindex/config.yaml (everything is routed via LiteLLM):
#    model: "gpt-4o-2024-11-20"                # OpenAI         → needs OPENAI_API_KEY
#    model: "github_copilot/claude-opus-4.8"   # GitHub Copilot → device login, no API key
#    model: "anthropic/claude-sonnet-4-6"      # Anthropic      → needs ANTHROPIC_API_KEY
```

`GITHUB_TOKEN` powers fetching (GraphQL); the configured `model` powers summaries and the agent. With a **GitHub Copilot** model, LiteLLM performs a one-time `github.com/login/device` and caches the token — no separate API key required.

### 🚀 Usage

**Python API**

```python
from pageindex import PageIndexClient

client = PageIndexClient(workspace="examples/workspace")
doc_id = client.index_github("owner/name")                       # auto: repo or org
# client.index_github("owner/name", mode="tracker", include="both")  # issues + PRs
# client.index_github("some-org", mode="org")                        # 100+ repos

client.refresh(doc_id)                                # incremental Merkle update
print(client.get_document_structure(doc_id))          # the tree (titles + summaries)
print(client.get_node_content(doc_id, "0007,0012"))   # code/text for specific nodes
```

**Agent CLI demo**

```bash
MODE=repo REPO=owner/name QUESTION="What does this repo do?" \
  python3 examples/github_vectorless_rag_demo.py
```

**Chat UI (ChatGPT-style)**

```bash
GITHUB_TOKEN=$GITHUB_TOKEN python3 webapp/server.py    # → http://127.0.0.1:8000
```

Type `owner/name`, click **Index**, then chat. Answers stream token-by-token with the agent's tool calls shown as chips, and responses come **only** from the indexed tree (traceable).

### 🧩 Where the GitIndex code lives

| Module | Role |
|---|---|
| `pageindex/sources/github_graphql.py` | Batched GitHub GraphQL client (tree walk, blobs, issues/PRs, org repos) |
| `pageindex/sources/code_parser.py` | Symbol extraction (Python `ast`; optional tree-sitter for other languages) |
| `pageindex/summarize.py` | Cheap Merkle summarizer + content-addressed summary cache |
| `pageindex/sources/github_repo.py`, `github_issues.py`, `github_org.py` | The three tree builders (repo / tracker / org) |
| `pageindex/retrieve.py`, `pageindex/client.py` | `get_node_content`, `index_github*()`, `refresh()` |
| `examples/github_vectorless_rag_demo.py` | Agent CLI demo |
| `webapp/` | Starlette backend + ChatGPT-style chat UI |

> The rest of this README documents **PageIndex**, the base engine GitIndex is built on.

---


<details open>
<summary><h2>📢 Updates</h2></summary>

- 🔥 [**Agentic Vectorless RAG**](https://github.com/VectifyAI/PageIndex/blob/main/examples/agentic_vectorless_rag_demo.py) — A simple agentic, vectorless RAG [example](#agentic-vectorless-rag-an-example) with *self-hosted PageIndex*, using OpenAI Agents SDK.
- [**Scale PageIndex to Millions of Documents**](https://pageindex.ai/blog/pageindex-filesystem) — *PageIndex File System* is a file-level tree layer that lets PageIndex reason over an entire corpus, not just a single document, enabling massive-scale document search.
- [PageIndex Chat](https://chat.pageindex.ai) — Human-like document analysis agent [platform](https://chat.pageindex.ai) for professional long documents. Also available via [MCP](https://pageindex.ai/developer) or [API](https://pageindex.ai/developer).
- [PageIndex Framework](https://pageindex.ai/blog/pageindex-intro) — Deep dive into PageIndex: an *agentic, in-context tree index* that enables LLMs to perform *reasoning-based, context-aware retrieval* over long documents.

 <!-- **🧪 Cookbooks:**
- [Vectorless RAG](https://docs.pageindex.ai/cookbook/vectorless-rag-pageindex): A minimal, hands-on example of reasoning-based RAG using PageIndex. No vectors, no chunking, and human-like retrieval.
- [Vision-based Vectorless RAG](https://docs.pageindex.ai/cookbook/vision-rag-pageindex): OCR-free, vision-only RAG with PageIndex's reasoning-native retrieval workflow that works directly over PDF page images. -->

</details>

---

# 📑 Introduction to PageIndex

Are you frustrated with vector database retrieval accuracy for long professional documents? Traditional vector-based RAG relies on semantic *similarity* rather than true *relevance*. But **similarity ≠ relevance** — what we truly need in retrieval is **relevance**, and that requires **reasoning**. When working with professional documents that demand domain expertise and multi-step reasoning, similarity search often falls short — missing what's relevant but not similar, and returning what's similar yet not relevant.

Inspired by AlphaGo, we propose **[PageIndex](https://vectify.ai/pageindex)** — a **vectorless**, **reasoning-based RAG** system that builds a **hierarchical tree index** from long documents and uses LLMs to **reason** *over that index* for **agentic, context-aware retrieval**. The retrieval is *traceable* and *explainable*, with no vector DBs or chunking.
PageIndex simulates how *human experts* navigate and extract knowledge from complex documents through *tree search*, enabling LLMs to *think* and *reason* their way to the most relevant document sections. It performs retrieval in two steps:

1. Generate a “Table-of-Contents” **tree structure index** of documents
2. Perform reasoning-based retrieval through **tree search**

<div align="center">
  <a href="https://pageindex.ai/blog/pageindex-intro" target="_blank" title="The PageIndex Framework">
    <img src="https://docs.pageindex.ai/images/cookbook/vectorless-rag.png" width="70%">
  </a>
</div>

### 🎯 Core Features

Compared to traditional vector-based RAG, **PageIndex** features:
- **No Vector DB**: Uses document structure and LLM reasoning for retrieval, instead of vector similarity search.
- **No Chunking**: Documents are organized into natural sections, not artificial chunks.
- **Better Traceability & Explainability**: Retrieval is reasoning-driven and grounded in explicit page and section references, making every result traceable and interpretable — no more “vibe retrieval” with opaque, approximate vector search.
- **Context-Aware Retrieval**: Retrieval depends on your full context (e.g., conversation history and domain knowledge), and easily incorporates new context.
- **Human-like Retrieval**: Simulates how human experts navigate and extract knowledge from complex documents.

PageIndex powers a reasoning-based RAG system that achieved **state-of-the-art** [98.7% accuracy](https://github.com/VectifyAI/Mafin2.5-FinanceBench) on FinanceBench, vastly outperforming vector-based RAG solutions on professional document analysis ([blog post](https://vectify.ai/blog/Mafin2.5)).

### 📍 Explore PageIndex

To learn more, please see a detailed introduction to the [PageIndex framework](https://pageindex.ai/blog/pageindex-intro). Check out [our GitHub](https://docs.pageindex.ai/open-source) for open-source code, and the [cookbooks](https://docs.pageindex.ai/cookbook), [tutorials](https://docs.pageindex.ai/tutorials), and [blog](https://pageindex.ai/blog) for more usage guides and examples.

The PageIndex service is available as a ChatGPT-style [chat platform](https://chat.pageindex.ai), or can be integrated via [MCP](https://pageindex.ai/developer) or [API](https://pageindex.ai/developer), with [enterprise](https://pageindex.ai/enterprise) deployment available.

### 🛠️ Deployment Options
- **Self-host** — run locally with this open-source repo (using standard PDF parsing).
- **Cloud Service** — production-grade pipeline with enhanced OCR, tree building, and retrieval for best results. Try instantly on our [Chat Platform](https://chat.pageindex.ai/), or integrate via [MCP](https://pageindex.ai/developer) or [API](https://pageindex.ai/developer).
- **Enterprise** — dedicated or private deployment (VPC, on-prem). [Contact us](https://ii2abc2jejf.typeform.com/to/gVv7qkaN) or [book a demo](https://calendly.com/pageindex/meet) to learn more.

### 🧪 Quick Hands-on

- 🔥 [**Agentic Vectorless RAG**](examples/agentic_vectorless_rag_demo.py) (**latest**) — a simple but complete **agentic vectorless RAG** [example](#agentic-vectorless-rag-an-example) with *self-hosted* PageIndex, using OpenAI Agents SDK.
- Try the [Vectorless RAG](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb) notebook — a *minimal*, hands-on example of reasoning-based RAG using PageIndex.
- Check out [Vision-based Vectorless RAG](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/vision_RAG_pageindex.ipynb) — no OCR; a minimal, vision-based & reasoning-native RAG pipeline that works directly over page images.
  
<div align="center">
  <a href="https://github.com/VectifyAI/PageIndex/blob/main/examples/agentic_vectorless_rag_demo.py" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/View_on_GitHub-Agentic_Vectorless_RAG-blue?style=for-the-badge&logo=github" alt="View on GitHub: Agentic Vectorless RAG" />
  </a>
  <br/>
  <a href="https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/Open_In_Colab-Vectorless_RAG-orange?style=for-the-badge&logo=googlecolab" alt="Open in Colab: Vectorless RAG" />
  </a>
  &nbsp;&nbsp;
  <a href="https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/vision_RAG_pageindex.ipynb" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/Open_In_Colab-Vision_RAG-orange?style=for-the-badge&logo=googlecolab" alt="Open in Colab: Vision RAG" />
  </a>
</div>

---

# 🌲 PageIndex Tree Structure

PageIndex can transform lengthy PDF documents into a semantic **tree structure**, similar to a _“table of contents”_ but optimized for use with Large Language Models (LLMs). It's ideal for: financial reports, regulatory filings, academic textbooks, legal or technical manuals, and any document that exceeds LLM context limits.

Below is an example PageIndex tree structure. Also see more example [documents](https://github.com/VectifyAI/PageIndex/tree/main/examples/documents) and generated [tree structures](https://github.com/VectifyAI/PageIndex/tree/main/examples/documents/results).

```jsonc
...
{
  "title": "Financial Stability",
  "node_id": "0006",
  "start_index": 21,
  "end_index": 22,
  "summary": "The Federal Reserve ...",
  "nodes": [
    {
      "title": "Monitoring Financial Vulnerabilities",
      "node_id": "0007",
      "start_index": 22,
      "end_index": 28,
      "summary": "The Federal Reserve's monitoring ..."
    },
    {
      "title": "Domestic and International Cooperation and Coordination",
      "node_id": "0008",
      "start_index": 28,
      "end_index": 31,
      "summary": "In 2023, the Federal Reserve collaborated ..."
    }
  ]
}
...
```

You can generate the PageIndex tree structure with this open-source repo; or use our [API](https://pageindex.ai/developer) for higher-quality results powered by our enhanced OCR and tree building pipeline.

---

# ⚙️ Package Usage

> **Note:** This package uses standard PDF parsing. For use cases with complex PDFs, our [cloud service](https://pageindex.ai/developer) (via MCP and API) offers enhanced OCR, tree building, and retrieval.

You can follow these steps to generate a PageIndex tree from a PDF document.

### 1. Install dependencies

```bash
pip3 install --upgrade -r requirements.txt
```

### 2. Set your LLM API key

Create a `.env` file in the root directory with your LLM API key. Multi-LLM is supported via [LiteLLM](https://docs.litellm.ai/docs/providers):

```bash
OPENAI_API_KEY=your_openai_key_here
```

### 3. Generate PageIndex structure for your PDF

```bash
python3 run_pageindex.py --pdf_path /path/to/your/document.pdf
```

<details>
<summary>Optional parameters</summary>
<br>
You can customize the processing with additional optional arguments:

```
--model                 LLM model to use (default: gpt-4o-2024-11-20)
--toc-check-pages       Pages to check for table of contents (default: 20)
--max-pages-per-node    Max pages per node (default: 10)
--max-tokens-per-node   Max tokens per node (default: 20000)
--if-add-node-id        Add node ID (yes/no, default: yes)
--if-add-node-summary   Add node summary (yes/no, default: yes)
--if-add-doc-description Add doc description (yes/no, default: yes)
```
</details>

<details>
<summary>Markdown support</summary>
<br>
We also provide markdown support for PageIndex. You can use the `--md_path` flag to generate a tree structure for a markdown file.

```bash
python3 run_pageindex.py --md_path /path/to/your/document.md
```

> Note: in this mode, we use "#" to determine node headings and their levels. For example, "##" is level 2, "###" is level 3, etc. Make sure your markdown file is formatted correctly. If your Markdown file was converted from a PDF or HTML, we don't recommend using this mode, since most existing conversion tools cannot preserve the original hierarchy. Instead, use our [PageIndex OCR](https://pageindex.ai/blog/ocr), which is designed to preserve it, to convert the PDF to a markdown file and then use this mode.
</details>

## Agentic Vectorless RAG: An Example

For a simple, end-to-end **agentic vectorless RAG** example using **self-hosted PageIndex** (with OpenAI Agents SDK), see [`examples/agentic_vectorless_rag_demo.py`](examples/agentic_vectorless_rag_demo.py).

```bash
# Install optional dependency
pip3 install openai-agents

# Run the demo
python3 examples/agentic_vectorless_rag_demo.py
```

<!--
# ☁️ Improved Tree Generation with PageIndex OCR

This repo is designed for generating PageIndex tree structure for simple PDFs, but many real-world use cases involve complex PDFs that are hard to parse by classic Python tools. However, extracting high-quality text from PDF documents remains a non-trivial challenge. Most OCR tools only extract page-level content, losing the broader document context and hierarchy.

To address this, we introduced PageIndex OCR — the first long-context OCR model designed to preserve the global structure of documents. PageIndex OCR significantly outperforms other leading OCR tools, such as those from Mistral and Contextual AI, in recognizing true hierarchy and semantic relationships across document pages.

- Experience next-level OCR quality with PageIndex OCR at our [Dashboard](https://dash.pageindex.ai/).
- Integrate PageIndex OCR seamlessly into your stack via our [API](https://docs.pageindex.ai/quickstart).

<p align="center">
  <img src="https://github.com/user-attachments/assets/eb35d8ae-865c-4e60-a33b-ebbd00c41732" width="80%">
</p>
-->

---

# 📈 Case Study: PageIndex Leads Finance QA Benchmark

[Mafin 2.5](https://vectify.ai/mafin) is a reasoning-based RAG system for financial document analysis, powered by **PageIndex**. It achieved a state-of-the-art [**98.7% accuracy**](https://vectify.ai/blog/Mafin2.5) on the [FinanceBench](https://arxiv.org/abs/2311.11944) benchmark, significantly outperforming traditional vector-based RAG systems.

PageIndex's hierarchical indexing and reasoning-driven retrieval enable precise navigation and extraction of relevant context from complex financial reports, such as SEC filings and earnings disclosures.

Explore the full [benchmark results](https://github.com/VectifyAI/Mafin2.5-FinanceBench) and our [blog post](https://vectify.ai/blog/Mafin2.5) for detailed comparisons and performance metrics.

<div align="center">
  <a href="https://github.com/VectifyAI/Mafin2.5-FinanceBench">
    <img src="https://github.com/user-attachments/assets/571aa074-d803-43c7-80c4-a04254b782a3" width="70%">
  </a>
</div>

---

# 🧭 Resources

* 📝 [Blog](https://pageindex.ai/blog): technical articles, research insights, and product updates.
* 🔧 [Developer](https://pageindex.ai/developer): MCP setup, API docs, and integration guides.
* 🧪 [Cookbooks](https://docs.pageindex.ai/cookbook): hands-on, runnable examples and advanced use cases.
* 📖 [Tutorials](https://docs.pageindex.ai/tutorials): practical guides and strategies, including *Document Search* and *Tree Search*.

---

# ⭐ Support Us

Leave us a star 🌟 if you like our project. Thank you!  

<p>
  <img src="https://github.com/user-attachments/assets/eae4ff38-48ae-4a7c-b19f-eab81201d794" width="80%">
</p>

Please cite this work as:
```
Mingtian Zhang, Yu Tang and PageIndex Team,
"PageIndex: Next-Generation Vectorless, Reasoning-based RAG",
PageIndex Blog, Sep 2025.
```

<details>
<summary>Or use the BibTeX citation.</summary>

```bibtex
@article{zhang2025pageindex,
  author = {Mingtian Zhang and Yu Tang and PageIndex Team},
  title = {PageIndex: Next-Generation Vectorless, Reasoning-based RAG},
  journal = {PageIndex Blog},
  year = {2025},
  month = {September},
  note = {https://pageindex.ai/blog/pageindex-intro},
}
```
</details>


### 🌐 Open-Source Ecosystem

PageIndex anchors a growing open-source [ecosystem](https://docs.pageindex.ai/open-source) of **long-context AI infra** — [OpenKB](https://github.com/VectifyAI/OpenKB) is an LLM knowledge base that compiles documents into an interlinked wiki. [ChatIndex](https://github.com/VectifyAI/ChatIndex) provides tree indexing and retrieval for long conversational histories and memory. [ConDB](https://github.com/VectifyAI/ConDB) is a KV-cache native context database for tree-based retrieval at scale. [PageIndex MCP](https://github.com/VectifyAI/pageindex-mcp) is PageIndex's MCP server.

### Connect with Us

<div align="center">

[![Website](https://img.shields.io/badge/Website-2D72CF?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDEgMSAxMWgyLjV2MTJoNnYtN2g1djdoNlYxMUgyM3oiLz48L3N2Zz4%3D)](https://pageindex.ai)&nbsp;
[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/PageIndexAI)&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTIwLjQ1IDIwLjQ1aC0zLjU1di01LjU3YzAtMS4zMy0uMDMtMy4wNC0xLjg1LTMuMDQtMS44NSAwLTIuMTQgMS40NS0yLjE0IDIuOTR2NS42N0g5LjM1VjloMy40MXYxLjU2aC4wNWMuNDgtLjkgMS42NC0xLjg1IDMuMzctMS44NSAzLjYgMCA0LjI3IDIuMzcgNC4yNyA1LjQ2djYuMjh6TTUuMzQgNy40M2EyLjA2IDIuMDYgMCAxIDEgMC00LjEzIDIuMDYgMi4wNiAwIDAgMSAwIDQuMTN6TTcuMTIgMjAuNDVIMy41NlY5aDMuNTZ2MTEuNDV6TTIyLjIyIDBIMS43N0MuNzkgMCAwIC43NyAwIDEuNzN2MjAuNTRDMCAyMy4yMy43OSAyNCAxLjc3IDI0aDIwLjQ1QzIzLjIgMjQgMjQgMjMuMjMgMjQgMjIuMjdWMS43M0MyNCAuNzcgMjMuMiAwIDIyLjIyIDB6Ii8%2BPC9zdmc%2B)](https://www.linkedin.com/company/vectify-ai/)&nbsp;
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/VuXuf29EUj)&nbsp;
[![Book a Demo](https://img.shields.io/badge/Book_a_Demo-6E7E96?style=for-the-badge&logo=googlecalendar&logoColor=white)](https://calendly.com/pageindex/meet)&nbsp;
[![Contact Us](https://img.shields.io/badge/Contact_Us-3B82F6?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgNCAyMCAxNiI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTIwIDRINGMtMS4xIDAtMiAuOS0yIDJ2MTJjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0wIDQtOCA1LTgtNVY2bDggNSA4LTV6Ii8%2BPC9zdmc%2B)](https://ii2abc2jejf.typeform.com/to/tK3AXl8T)

</div>

---

© 2026 [Vectify AI](https://vectify.ai)
