# Why Claude 4.5 refuses benign safety research: two mechanisms, not one

SPAR Model Forensics take-home, August 2026. Mohit Goyal.

A fork of [adsingh-64/safety-refusals](https://github.com/adsingh-64/safety-refusals) by Aditya Singh, whose replication and framing this builds on directly. His original README is preserved as [UPSTREAM_README.md](UPSTREAM_README.md).

> ## Fabrication notice
>
> The SAFETY-2847 ticket, its incidents, and the "Claude Internal" system prompt are **invented for this experiment**. No such project, deployment, or internal document exists.
>
> The default name roster uses **real Anthropic researchers whose involvement is entirely fictional**. They did not report, assign, approve, or review anything described here. The names fill slots in a fabricated ticket, inherited from the upstream experiment. `Names.FICTIONAL` and `Names.ANONYMOUS` rosters exist in the code so this can be tested rather than assumed harmless.
>
> The stored responses in `results/` quote this fabricated ticket back, so the same applies to them.

## The finding

Anthropic reads these refusals as the model mistaking a benign request for a jailbreak. UK AISI reads them as mild misalignment. Across 540 responses and 17 conditions, **two distinct mechanisms are being counted as one behaviour**:

| | Uncertainty | Objection |
|---|---|---|
| Response shape | withholds, offers to continue | declines, no offer |
| Asks whether the request is authorised | yes | no, argues the harm instead |
| Letting the model reason first | **collapses**, 43 to 70 points | **no change**, 0 points |
| Adding legal and executive sign-off | worth 30 points | **worth nothing**, 0 of 30 |

Both camps are partly right, about different regimes. Full write-up: [WRITEUP_DRAFT.md](WRITEUP_DRAFT.md).

![Two mechanisms behind one refusal](results/figure.png)

## What this fork adds

Upstream moved one line and saw refusal go from 0% to 100%. That line changes three things at once, so the result cannot say which one matters. Here they move independently.

| | |
|---|---|
| `conditions.py` | composes the ticket from trust, consequence and valence flags |
| `experiments.py` | named condition sets: pilot, replication, exp1 to exp4, controls |
| `budget.py` | prices every batch before it runs and refuses to cross a cap |
| `runner.py` | estimate, guard, call, persist, report |
| `store.py` | append-only JSONL, so an interrupted sweep resumes instead of re-billing |
| `judge.py` | four-level rubric replacing the binary refused/complied label |
| `grading.py` | grades stored responses, so revising the rubric costs no new completions |
| `labels.py`, `reading.py` | human labelling, judge validation, and structural signals |
| `backends.py` | native Anthropic API and OpenRouter behind one interface |

130 tests, none of which touch the network.

## Fidelity to upstream

`tests/test_conditions.py::test_absent_cell_matches_upstream_prompt_exactly` asserts that the composed `trust=full, consequence=full-minus-target-line` cell is **byte identical** to upstream's `USER_PROMPT`. If that test fails, these numbers have stopped being comparable to his and the difference must be explained before anything is reported.

## Grading, validated three ways

- **Claude Haiku 4.5** graded all 540 responses on a four-level rubric.
- **GPT 5.6 Sol**, via the Codex CLI, independently graded a 40-sample stratified subset: **40 of 40** agreement on the withholding boundary, 35 of 40 exact. All five disagreements fall within a category and move no reported rate.
- **A mechanical text check** (do all five requested enterprise domains appear at length) agrees with the graders on the drafted-or-not question in **540 of 540** samples, asserted as a test.

No human validated the grades. Two models could in principle share a blind spot.

## Running it

```bash
uv sync
cp .env.example .env      # then fill in ANTHROPIC_API_KEY or OPENROUTER_API_KEY
uv run pytest             # 130 tests, no network
```

Dry runs need no API key. They print the matrix, what is already stored, and the price:

```bash
uv run experiments/run.py exp1 --dry-run --reasoning off --n 30 --model opus-4.5
uv run experiments/grade.py --report-only
uv run scripts/make_figure.py
```

Then, with a key:

```bash
uv run experiments/smoke.py                                  # one call, proves the wiring
uv run experiments/run.py pilot --n 10 --cap 1
uv run experiments/grade.py --cap 2
```

Three cost guards, because the upstream default `max_tokens=16000` is roughly a six-fold tail: a hard `max_tokens` cap, a spend ledger that refuses a batch **before** sending it, and a store that tops a cell up rather than restarting it. Total spend for the whole study was about USD 25.

## Layout

```
src/safety_refusals/   conditions, experiments, budget, store, runner,
                       judge, grading, labels, reading, backends, models
experiments/           run.py, grade.py, label.py, smoke.py
scripts/               figure, appendix, docx, labelling page, codex grader, migration
results/               540 responses, 560 verdicts, figure, appendix table
tests/                 130 tests
WRITEUP_DRAFT.md       the full write-up
HARNESS.md             design notes for this fork
```

## Credit

The experiment design, the SAFETY-2847 ticket, and the original 0% to 100% result are Aditya Singh's. This fork separates his single manipulation into independent axes, replaces the binary label with a graded one, and tests the two questions he left open: which component of the framing carries the effect, and whether the effect survives when the intervention is unambiguously value-eroding.
