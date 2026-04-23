# Benchmark audit — configuration & realism

Triple-pass audit of the elephantmemory harness after the mem0/letta/zep adapter
fixes landed in [PR #4](https://github.com/prachitbhike/elephantmemory/pull/4).
The blocking adapter bugs are resolved; what remains is a set of configuration
and methodology choices that still bias the comparison or weaken its
real-world signal for builders.

## A. Adapter-configuration issues still in the codebase

1. **mem0 uses Claude as its internal LLM**
   ([mem0_adapter.py:55-58](elephantmemory/adapters/mem0_adapter.py)). mem0's
   extraction prompts were authored and tuned against OpenAI; running them on
   Claude is supported but is not the documented "happy path." A builder
   evaluating mem0 will almost certainly pick OpenAI for the LLM. We're
   handicapping mem0 in a non-obvious way. Either A/B both providers or default
   to OpenAI and footnote the choice.

2. **Letta agents are initialized with very thin memory blocks**
   ([letta_adapter.py:102-105](elephantmemory/adapters/letta_adapter.py)). The
   `human` block is `"User id: {x}. Details unknown — learn from
   conversation."` and the `persona` is one sentence. Letta's value prop is
   *agent-curated* memory; how well that curation works depends heavily on
   whether the persona block instructs the agent to reflect on conversations
   and write to core/archival memory. A real builder writes a 5–10 line
   persona that explicitly tells the agent how to manage memory. Without that,
   we're benchmarking default Letta, which is weaker than tuned Letta.

3. **"zep" is graphiti-core OSS, not the Zep product.** The README labels the
   column "zep". Most readers will assume the commercial Zep memory layer
   (which has a smarter retrieval/summarization layer on top of graphiti).
   Either rename the column to `graphiti` or add a prominent footnote.

4. **`claude_memory` has a real home-field advantage** that's not just the
   model. Judge is Claude Sonnet 4.6, assistant model is Claude Sonnet 4.6,
   memory tool is Claude-native with prompts authored by Anthropic. The
   "same family avoids home-field bias" rationale in the README is backwards
   — using the same family for assistant + judge maximizes alignment with
   whichever framework runs on Claude. Control for it by re-judging a sample
   of probes through a non-Claude judge (GPT-4o or Gemini) and reporting the
   delta.

5. **Latency is "end-to-end query," not "memory retrieval."** For
   pgvector/mem0 it's `embed + vector search + Claude answer call`. For
   claude_memory and letta it's the entire agent tool-loop. Defensible, but
   should be labeled clearly so a builder isn't misled into comparing raw
   retrieval speeds.

6. **Cost reporting only tracks `write_cost_usd`.** Query costs are stored
   per-outcome but never aggregated in [report.py](elephantmemory/report.py).
   For frameworks that do extraction at write (mem0, pgvector, zep), write
   cost dominates. For agent frameworks (letta, claude_memory), query cost
   dominates. The current report makes the latter look free.

## B. Scenario realism issues — these matter most for builder usefulness

7. **"Concurrency" scenarios are not actually concurrent.**
   [runner.py:46-71](elephantmemory/runner.py) processes events strictly
   sequentially in timestamp order. The `concurrency_address` scenario has
   two writes 30 seconds apart, run back-to-back — that's a temporal-supersede
   test. True concurrency would mean two clients writing to the same user
   simultaneously and verifying the memory layer converges. For builders
   running multi-instance servers, real concurrency (race conditions,
   write-write conflicts, eventual consistency) is the question. Either rename
   the category to `rapid_supersede` or build a real concurrent driver.

8. **Sessions are 1–3 turns, very clean, focused entirely on the fact under
   test.** Real conversations are 10–30 turns mixed with weather chatter,
   half-formed asks, and corrections. When the relevant fact is one sentence
   buried in a 20-turn transcript, extraction quality tanks for *every*
   framework — and the spread between them changes. We're benchmarking
   artificially easy inputs.

9. **Probes are phrased almost identically to the original statement.**
   `"Heading out to walk Pepper, my golden retriever"` → probe `"What's my
   dog's name?"`. Real users phrase questions inconsistently — "what's my pup
   called", "do I have any pets". Vector retrieval looks much better when the
   query lexically matches storage. Need paraphrased probes.

10. **Almost no decoy data.** Each scenario has 1–4 sessions total. Real users
    accumulate hundreds of sessions; retrieval needs to surface the right one
    out of 100s of competing facts. With a corpus this small, even random
    retrieval often hits.

11. **No long-tail / volume scenarios.** Nothing tests "150 sessions in, can
    we still retrieve a fact from session 3?" — the actual production
    question.

12. **Forget tests use a single deletion-request shape** (`"any health info"`).
    Real GDPR-shaped requests are usually narrower ("delete reference to my
    employer") and sometimes broader ("erase everything"). Need both extremes.

13. **No mid-session correction scenarios** (`"ignore that, I meant…"`).
    Common in real assistant traffic, surprisingly hard for memory layers.

14. **Brooke barely exists.** Cross-user isolation is one of the most
    important developer concerns and we test it with a user who has
    effectively no history. The realistic case is *both* users having rich
    histories that overlap topically.

## C. Prioritized fix plan

| Priority | Fix | Effort |
|---|---|---|
| P0 | Run mem0 with OpenAI as its LLM (its documented default) and report both | small |
| P0 | Beef up Letta's persona block with explicit memory-curation instructions | small |
| P0 | Aggregate per-query cost in [report.py](elephantmemory/report.py); split write/query/total | small |
| P0 | Rename "zep" → "graphiti" in the README + report headers | trivial |
| P1 | Add a paraphrase variant for every recall / temporal probe | medium |
| P1 | Extend 5–10 scenarios with 10–20 noise turns of unrelated chatter per session | medium |
| P1 | Cross-validate a 20-probe sample with GPT-4o as judge; report disagreement rate | small |
| P2 | Build a real concurrent-write driver (threadpool) for the concurrency category — or rename it | medium |
| P2 | Add a "long history" scenario set: 50+ sessions per user, probe a fact from session 3 | medium |
| P2 | Give Brooke a parallel rich history that overlaps with Atlas's topics | medium |

P0 is ~1 hour total and makes the next run materially more credible. P1 is
the bigger lift but has the highest payoff for "useful to a builder."
