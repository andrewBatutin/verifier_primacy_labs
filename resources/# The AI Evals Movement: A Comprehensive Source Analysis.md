# The AI Evals Movement: A Comprehensive Source Analysis

A tight-knit group of practitioners is reshaping how the industry approaches AI evaluation. **Hamel Husain, Shreya Shankar, and John Berryman**—connected through Parlance Labs and the Maven AI Evals course—have collectively trained over 3,000 professionals and established what's becoming the dominant methodology for evaluating LLM applications. Their core thesis: most AI products fail not from model limitations, but from inadequate evaluation systems.

---

## The central hub: Parlance Labs and the Maven course

The **"AI Evals for Engineers & PMs"** course on Maven represents the commercial epicenter of this movement. It's the **#1 highest-grossing course on Maven**, having trained professionals from **500+ companies** including teams at OpenAI, Anthropic, Google, Microsoft, and Amazon.

**Course structure and pricing:**
- 4 weeks, 3-4 hours weekly commitment
- 2 lessons per week with 12+ hours of live office hours
- Full price ~$2,000-2,500 (discounts bring it to ~$1,300)
- Includes $1,000 in Modal Labs compute credits
- Lifetime access to materials and 1,000+ student Discord community

**The complete curriculum covers:**

| Week | Focus | Key Topics |
|------|-------|------------|
| 1 | Fundamentals & Lifecycle | Three Gulfs Model, Analyse→Measure→Improve cycle, error analysis to "theoretical saturation" |
| 2 | Implementing Evaluations | Code-based evals, LLM-as-judge, binary pass/fail vs. scoring rubrics |
| 3 | Architecture-Specific Eval | RAG evaluation, agentic systems, production monitoring |
| 4 | Efficiency & Optimization | Human review interfaces, cost optimization, sampling strategies |

The course explicitly rejects generic benchmarks in favor of **product-specific evaluations**, teaching students to build custom annotation tools in Jupyter notebooks and use qualitative research methods like "open coding" and "axial coding" borrowed from social science.

---

## Hamel Husain: The practical evangelist

Hamel serves as the movement's public voice. With **42.3K Twitter followers** and a newsletter exceeding **25,000 subscribers**, his content has shaped industry thinking on evals.

### Professional credentials that matter

His authority stems from foundational work at GitHub, where he **led the team that created CodeSearchNet**—early LLM research on semantic code search that OpenAI used for code understanding, contributing to GitHub Copilot's development. He previously worked at Airbnb on ML forecasting and is now founder of Parlance Labs, an "engineer-only" consulting firm with a **minimum engagement of $89,500**.

### The three pillars of his philosophy

**1. Error analysis first, always.** Hamel insists teams spend **60-80% of development time** on error analysis and evaluation. His process: manually review 20-50 outputs, categorize failures through "open coding," and reach "theoretical saturation" (typically 100+ examples) before writing any automated evals.

**2. Binary judgments over scales.** "If your evaluations consist of metrics that LLMs score on a 1-5 scale, you're doing it wrong." Pass/fail forces clarity and actionability; critiques capture nuance while maintaining simplicity.

**3. Generic metrics are useless.** BERTScore, ROUGE, cosine similarity—these "lead people astray." The only meaningful evals are domain-specific ones built with your principal domain expert.

### Key blog posts documenting the methodology

| Post | Date | Core Contribution |
|------|------|-------------------|
| "Your AI Product Needs Evals" | Mar 2024 | Introduced the three-level evaluation hierarchy (Unit Tests → Human/Model Eval → A/B Testing) |
| "Using LLM-as-a-Judge: A Complete Guide" | Oct 2024 | Codified "Critique Shadowing"—a 7-step methodology for building aligned LLM judges |
| "LLM Evals: Everything You Need to Know" | Jan 2026 | Comprehensive FAQ synthesizing lessons from 700+ students |

### The "Three Gulfs" model

A central framework taught in the course:
- **Gulf of Specification**: Communicating developer intent to the LLM pipeline
- **Gulf of Comprehension**: Cannot manually review all inputs/outputs at scale
- **Gulf of Generalization**: LLMs don't generalize perfectly across diverse inputs

### Tools and open-source contributions

- **nbdev** (5.2K GitHub stars): Literate programming for Python using Jupyter
- **CodeSearchNet** (2.4K stars): Foundational dataset for ML on code
- **Judgy**: Course-released Python package for confidence intervals on LLM-as-Judge metrics

### Notable quotes that define his stance

> "Unsuccessful products almost always share a common root cause: a failure to create robust evaluation systems."

> "Remove ALL friction from looking at data. Notebooks are the single most effective tool for evals."

> "Be wary of optimizing for high eval pass rates. If you're passing 100% of your evals, you're likely not challenging your system enough. A 70% pass rate might indicate a more meaningful evaluation."

---

## Shreya Shankar: The academic backbone

Shreya brings scholarly rigor to the movement. A **final-year PhD student at UC Berkeley** (advised by Aditya Parameswaran), she bridges academic research with practical industry impact.

### Research that's reshaping tooling

Her work has been deployed at scale: **SPADE** assertions are now integrated into **LangSmith** (LangChain's platform), used across **2,000+ pipelines** in finance, medicine, and IT.

**Key papers and their contributions:**

| Paper | Venue | Impact |
|-------|-------|--------|
| "Who Validates the Validators?" | UIST 2024 | Introduced **EvalGen**—a mixed-initiative system for aligning LLM-generated evaluation functions with human preferences |
| "SPADE: Synthesizing Data Quality Assertions" | VLDB 2024 | Automatically generates assertions from prompt version histories; deployed in LangSmith |
| "DocETL: Agentic Query Rewriting and Evaluation" | VLDB 2025 | **3,400+ GitHub stars**; used by California public defenders for criminal trial document analysis |
| "Steering Semantic Data Processing with DocWrangler" | UIST 2025 | **Best Paper Honorable Mention**; IDE for DocETL pipeline refinement |

### Her defining philosophical position

Shreya's September 2025 post **"In Defense of AI Evals, for Everyone"** (~240K impressions) directly countered rising anti-eval sentiment in the community:

> "When people say they 'don't do evals,' they are usually lying to themselves. Every successful product evaluates quality somewhere in the lifecycle."

She argues that anti-eval rhetoric is **"particularly harmful for new builders"** who lack intuition for error analysis. Her position: evals live on a spectrum—not one-size-fits-all—but systematic measurement is non-negotiable for serious products.

### The data flywheel concept

Her July 2024 post introduced a critical framework: production LLM products need **ongoing feedback loops** to continually improve through labeling, correlating against human judgment, and feeding back improvements. This isn't a one-time eval setup—it's a continuous cycle.

### Social media and teaching presence

With **50.3K Twitter followers**, her most viral post captured the movement's central tension:

> "Evals are arguably the hardest part of LLMOps. LLMs mess up, so we check them w/ other LLMs, but this feels icky. Who validates the validators??"

She created **evals.info** as a community resource and offers the 166-page course reader free to universities teaching AI engineering.

---

## Arcturus Labs and John Berryman: The Copilot connection

John Berryman brings unique credibility: he **helped build GitHub Copilot**, working on both code completions and the chat product on GitHub's Model Improvements team.

### The company's consulting model

Arcturus Labs offers services ranging from kickstart guidance to embedded expert engagements:
- **LLM Application Audit**: Architecture, prompting strategy, and evaluation framework assessment
- **3-Day Team Training**: Intensive workshops on prompt engineering, RAG, and AI agents
- **Office Hours**: Weekly expert guidance at $X/hour

### His three-part eval taxonomy from Copilot

From the Maven Lightning Lesson "How Evals Made GitHub Copilot Happen" (880 students):

| Type | Method | Use Case |
|------|--------|----------|
| **Algorithmic** | Confusion matrices | Tool call verification |
| **Verifiable** | Harnesslib concrete checks | Code execution testing |
| **Subjective** | LLM-as-Judge | Quality assessment requiring judgment |

### Unique frameworks from the blog

**1. "Fire Yourself First"** (Jan 2025): Inspired by E-Myth methodology—incrementally automate by identifying tasks you do manually, documenting them, then systematically automating each one.

**2. "Roaming RAG"** (Nov 2024): An alternative to vector databases where the LLM navigates documentation using tools—particularly suited for well-organized docs like llms.txt.

**3. "AI Empathy" approach** (Oct 2025): Treat LLMs as "AI Interns" on their first day. Empathizing with their perspective improves context engineering.

### Key published resources

- **"Prompt Engineering for LLMs"** (O'Reilly, 2024): Co-authored with Albert Ziegler, foreword by Hamel Husain
- **"Relevant Search"** (Manning): The art of building search applications
- **"Context Engineering for LLMs"** (announced 2025): Second book with Ziegler

### The Copilot insight that shapes his eval philosophy

> "Prompt crafting is really all about creating a 'pseudo-document' that will lead the model to a completion that benefits the customer."

Their primary success metric was **"completion"**—when users accept and keep suggestions. This user-centric measurement philosophy carries through to his consulting work.

---

## The interconnected network

These four sources aren't independent—they form a tightly integrated ecosystem.

### Direct collaboration map

```
                    Parlance Labs
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   Hamel Husain    Shreya Shankar    John Berryman
   (Founder)       (Research Lead)   (Team Member)
        │                │                │
        └───────┬────────┴────────────────┘
                │
        Maven AI Evals Course
        (Co-instructors)
```

**John Berryman** is listed on the Parlance Labs team page and taught "Prompt Engineering" in their open LLM course. He co-presents the "How Evals Made GitHub Copilot Happen" lightning lesson with Hamel and Shawn Simister.

**Shreya Shankar** serves as Research Lead at Parlance Labs. Her academic research directly informs course content, and she co-authors key methodology posts with Hamel.

### Industry validation through testimonials

**Harrison Chase (LangChain CEO):** "Hamel is one of most knowledgeable people about LLM evals. We've even made many improvements to LangSmith because of his work."

**Simon Willison:** "Hamel's content is fantastic, but it's a bit absurd that he's single-handedly having to make up for a lack of good materials about this topic across the rest of our industry!"

**Eugene Yan (Amazon):** "When I have questions about the intersection of data and production AI systems, Shreya & Hamel are the first people I call."

---

## Shared philosophical tenets across all sources

Despite different backgrounds (industry consultant, academic researcher, Copilot builder), they converge on core principles:

### What they agree on

1. **Error analysis precedes automation.** All three emphasize manually reviewing outputs before building automated evals. The magic number: 20-50 outputs minimum, often 100+.

2. **Domain experts are non-negotiable.** "Many developers attempt to act as the domain expert themselves, or find a convenient proxy. This is a recipe for disaster." (Hamel)

3. **Binary judgments beat scales.** Pass/fail with written critiques captures more signal than 1-5 ratings that lack clear anchors.

4. **Generic benchmarks are misleading.** Product-specific evals trump off-the-shelf metrics every time.

5. **The human must remain in the loop.** John: "The most important thing is that a human is actually looking at the data and making the human judgment."

6. **Evals are iterative, not one-time.** Shreya's data flywheel concept frames this as continuous improvement, not a checkbox.

### Where they add distinct value

| Source | Unique Contribution |
|--------|---------------------|
| Hamel | The "Three Gulfs" model; practical consulting experience across 30+ companies |
| Shreya | Academic rigor (EvalGen, SPADE); large-scale deployment in LangSmith |
| John | Copilot-specific lessons; search/retrieval expertise; artifact-aware UX patterns |

---

## Tools and open-source projects released

| Tool | Creator | GitHub Stars | Purpose |
|------|---------|--------------|---------|
| **DocETL** | Shreya | 3,400+ | Agentic LLM-powered data processing with built-in evaluation |
| **nbdev** | Hamel (with fast.ai) | 5,200+ | Literate programming enabling tests as first-class citizens |
| **CodeSearchNet** | Hamel (at GitHub) | 2,400+ | Benchmark dataset for code understanding |
| **Judgy** | Course repo | 75+ | Confidence intervals for LLM-as-Judge metrics |
| **artifact-aware-assistant** | John | 12 | Demo for artifact-aware AI assistant UX |
| **gpt3-sandbox** | Shreya | 2,900+ | Early GPT-3 web demo creator |

---

## Content inventory across formats

### YouTube and conference talks

- **Hamel**: "Made this video to explain evals" (Dec 2025, 59K views); "Build Your Own Eval Tools with Notebooks" (Jun 2025)
- **Shreya**: Stanford MLSys Seminar (Apr 2024) on evaluating LLM pipeline output quality
- **John**: Maven Lightning Lesson on GitHub Copilot evals (880 students); Nashville AI Tinkerers meetups

### Podcast appearances

| Guest | Podcast | Key Topic |
|-------|---------|-----------|
| Hamel + Shreya | Lenny's Podcast | "Why AI evals are the hottest new skill for product builders" |
| Hamel | Vanishing Gradients | "A Field Guide to Rapidly Improving AI Products" |
| Shreya | TWIML AI | "AI Agents for Data Analysis" with DocETL focus |
| John | Scaling Tech | "Prompt Engineering for LLMs: Best Practices" |
| Hamel | TWIML AI | "Building Real-World LLM Products with Fine-Tuning" |

### Books and long-form publications

- **"Prompt Engineering for LLMs"** (O'Reilly, 2024) - John Berryman & Albert Ziegler
- **"Relevant Search"** (Manning) - John Berryman & Doug Turnbull
- **"What We Learned from a Year of Building with LLMs"** (O'Reilly report) - Hamel, Shreya, Eugene Yan, Bryan Bischof, Jason Liu, Charles Frye
- **AI Evals O'Reilly Book** (Spring 2026) - Hamel & Shreya (forthcoming)

---

## Key frameworks and mental models for thesis development

### The Analyse→Measure→Improve cycle

The central iterative process:
1. **Analyse**: Collect examples, categorize failure modes through open coding, reach theoretical saturation
2. **Measure**: Translate qualitative insights into quantitative metrics (binary judgments preferred)
3. **Improve**: Refine prompts, models, architecture based on data-driven insights
4. **Return to Analyse** with production data

### Three levels of evaluation maturity

| Level | Method | When to Use |
|-------|--------|-------------|
| Level 1 | Unit tests/assertions | Run on every code change |
| Level 2 | Human review + LLM-as-judge | Periodic deep analysis |
| Level 3 | A/B testing | Mature products measuring real user outcomes |

### The "Critique Shadowing" methodology for LLM judges

Hamel's 7-step process:
1. Find the Principal Domain Expert (one person whose judgment defines success)
2. Create a diverse dataset covering features, scenarios, personas
3. Collect binary pass/fail judgments with written critiques
4. Fix obvious errors in the system before automating evaluation
5. Build LLM judge iteratively, aligning with human expert
6. Perform error analysis on judge disagreements
7. Create specialized judges only if the general judge fails on specific categories

---

## Contrarian positions they've taken

**Against eval-driven development:** "Unlike traditional software where failure modes are predictable, LLMs have infinite surface area for potential failures. Write evals for errors you discover, not errors you imagine."

**Against the "Claude doesn't use evals" narrative:** Hamel called this "a crazy controversy"—rigorous evals remain essential regardless of what frontier labs claim publicly.

**Against metric overload:** "I hate off-the-shelf metrics that come with many evaluation frameworks. They tend to lead people astray."

**For building custom tools:** Rather than adopting evaluation platforms wholesale, they advocate building lightweight custom annotation tools in Jupyter notebooks—removing all friction from data review.

---

## Synthesis for thesis development

The collective output from these four sources supports several compact theses:

1. **The root cause thesis:** Most AI product failures trace to inadequate evaluation systems, not model limitations.

2. **The error analysis thesis:** Systematic manual review of outputs (20-100+ examples) must precede any automated evaluation.

3. **The domain expert thesis:** Evaluation quality is bounded by access to a "benevolent dictator" domain expert who defines success.

4. **The anti-generic thesis:** Off-the-shelf metrics (BERTScore, ROUGE, generic LLM-as-judge) harm more than help for production applications.

5. **The binary judgment thesis:** Pass/fail with critiques produces more actionable signal than 1-5 scales.

6. **The continuous improvement thesis:** Evals aren't a one-time setup but a data flywheel requiring ongoing investment.

7. **The tooling thesis:** Custom lightweight tools (Jupyter notebooks, simple annotation interfaces) outperform complex evaluation platforms.

8. **The "new PRD" thesis:** Well-crafted evaluation prompts effectively become living product requirements that continuously test the AI system.

This ecosystem—combining Hamel's practical consulting experience, Shreya's academic research rigor, and John's Copilot-building background—has established what may become the dominant paradigm for production AI evaluation methodology.