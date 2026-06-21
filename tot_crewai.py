# tot_crewai.py
# -----------------------------------------------------------------------------
# The SAME Tree of Thoughts, sketched in CrewAI.
#
# CrewAI is a PYTHON library (pip install crewai) -- no GUI, no website.
# Its nouns: Agent (an LLM persona), Task (work for an agent), Crew (a team).
#
# Important: CrewAI has NO built-in "Tree of Thoughts." It orchestrates AGENTS,
# not search. So we map ToT's two model jobs onto two agents...
#     Proposer  -> generates candidate next steps
#     Evaluator -> scores how promising each is
# ...and the search loop itself is STILL just plain Python (the for-loop below).
# That's the whole point: CrewAI gives you the personas, not the algorithm.
#
#   pip install crewai
# -----------------------------------------------------------------------------

from crewai import Agent, Task, Crew

# Reuse the same node type as the other two files -- ToT doesn't change.
from tot_minimal import Thought


# --- 1. Two agents = the two places a model is actually needed ----------------
proposer = Agent(
    role="Reasoning Proposer",
    goal="Propose several distinct next steps for a partially-solved problem",
    backstory="You expand promising lines of reasoning, one step at a time.",
)

evaluator = Agent(
    role="Reasoning Critic",
    goal="Judge how likely a partial reasoning path leads to a correct answer",
    backstory="You are a strict grader who replies with a single number 0-1.",
)


# --- 2. One-shot helpers that run a tiny crew and return the text -------------
def generate(task: str, thought: Thought, k: int) -> list[Thought]:
    t = Task(
        description=(
            f"Task: {task}\nReasoning so far:\n{thought.text or '(none yet)'}\n"
            f"Propose {k} distinct next steps, one per line."
        ),
        expected_output=f"{k} lines, one next step each.",
        agent=proposer,
    )
    out = Crew(agents=[proposer], tasks=[t]).kickoff()
    lines = [ln.strip() for ln in str(out).splitlines() if ln.strip()]
    return [Thought(thought.steps + [ln]) for ln in lines]


def evaluate(task: str, thought: Thought) -> float:
    t = Task(
        description=(
            f"Task: {task}\nCandidate path:\n{thought.text}\n"
            f"Rate 0-1 how likely this leads to a correct solution. Reply with only a number."
        ),
        expected_output="A single number between 0 and 1.",
        agent=evaluator,
    )
    out = Crew(agents=[evaluator], tasks=[t]).kickoff()
    try:
        return float(str(out).strip())
    except ValueError:
        return 0.0


# --- 3. The search loop -- IDENTICAL to tot_minimal.py -----------------------
# Note: nothing here is CrewAI. The framework only powered generate/evaluate.
def tree_of_thoughts(task: str, k: int = 3, beam: int = 2, depth: int = 3) -> Thought:
    frontier = [Thought(steps=[])]
    for _ in range(depth):
        cands: list[Thought] = []
        for th in frontier:
            cands += generate(task, th, k)        # branch  (Proposer agent)
        for c in cands:
            c.score = evaluate(task, c)           # score   (Evaluator agent)
        cands.sort(key=lambda c: c.score, reverse=True)
        frontier = cands[:beam]                   # prune   (plain Python)
    return frontier[0]


if __name__ == "__main__":
    best = tree_of_thoughts("Use 4, 9, 10, 13 and + - * / to make 24.")
    print(best.text, "->", best.score)

# -----------------------------------------------------------------------------
# Takeaway: even in CrewAI, the ToT search is your own loop. CrewAI is worth it
# when ToT is ONE PART of a bigger multi-agent system (roles, delegation,
# tools) -- not when ToT is all you need. For ToT alone it's overkill.
# (For real branching/looping/state inside CrewAI, see its newer "Flows" API.)
# -----------------------------------------------------------------------------
