# tot_minimal.py
# -----------------------------------------------------------------------------
# Tree of Thoughts (ToT) in PURE PYTHON. No LangChain, no LangGraph, no CrewAI.
#
# The whole point of this file: ToT is an ALGORITHM, not a framework feature.
#   generate candidate thoughts -> evaluate them -> keep the best (prune)
#   -> expand the survivors -> repeat. That loop IS Tree of Thoughts.
#
# Compare with chain-of-thought (CoT), which is a SINGLE linear path.
# ToT explores a TREE of paths and searches it (here: beam search).
# -----------------------------------------------------------------------------

from dataclasses import dataclass


# --- 1. The ONLY model dependency. Swap in any provider you like. -------------
def call_model(prompt: str) -> str:
    """Replace this with your real model call (Anthropic, OpenAI, local, ...)."""
    raise NotImplementedError("Wire up your model of choice here.")


# --- 2. A node in the reasoning tree -----------------------------------------
@dataclass
class Thought:
    steps: list[str]          # the partial reasoning path so far
    score: float = 0.0        # how promising it looks (set by evaluate)

    @property
    def text(self) -> str:
        return "\n".join(self.steps)


# --- 3. The two model-powered operations -------------------------------------
def generate(task: str, thought: Thought, k: int) -> list[Thought]:
    """Branch: propose k distinct next steps from the current path."""
    prompt = (
        f"Task: {task}\n"
        f"Reasoning so far:\n{thought.text or '(none yet)'}\n"
        f"Propose {k} distinct next steps, one per line."
    )
    lines = [ln.strip() for ln in call_model(prompt).splitlines() if ln.strip()]
    return [Thought(thought.steps + [ln]) for ln in lines]


def evaluate(task: str, thought: Thought) -> float:
    """Score: rate how likely this partial path leads to a correct answer."""
    prompt = (
        f"Task: {task}\n"
        f"Candidate path:\n{thought.text}\n"
        f"Rate from 0 to 1 how likely this leads to a correct solution. "
        f"Reply with only a number."
    )
    try:
        return float(call_model(prompt).strip())
    except ValueError:
        return 0.0


# --- 4. The search loop -- THIS is Tree of Thoughts --------------------------
def tree_of_thoughts(task: str, k: int = 3, beam: int = 2, depth: int = 3) -> Thought:
    frontier = [Thought(steps=[])]
    for _ in range(depth):
        candidates: list[Thought] = []
        for t in frontier:
            candidates.extend(generate(task, t, k))      # 1. branch
        for c in candidates:
            c.score = evaluate(task, c)                   # 2. evaluate
        candidates.sort(key=lambda c: c.score, reverse=True)
        frontier = candidates[:beam]                      # 3. prune (beam search)
    return frontier[0]


if __name__ == "__main__":
    best = tree_of_thoughts("Use 4, 9, 10, 13 and + - * / to make 24.")
    print(best.text, "->", best.score)
