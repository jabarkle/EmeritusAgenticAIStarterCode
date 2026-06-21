# tot_langgraph.py
# -----------------------------------------------------------------------------
# The SAME Tree of Thoughts as tot_minimal.py, expressed in LangGraph.
#
# Notice what the framework ADDS:
#   - a typed shared state (ToTState)
#   - named nodes wired into a graph
#   - a conditional edge that loops until we hit max depth
#   - (for free, not shown) checkpointing, streaming, human-in-the-loop, tracing
#
# The generate/evaluate/prune LOGIC is identical -- we even import it unchanged.
# For a simple ToT this is extra machinery. For a PRODUCTION system that needs
# persistence, resumability, or observability, the machinery earns its keep.
#
#   pip install langgraph
# -----------------------------------------------------------------------------

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

# Reuse the exact same primitives -- ToT doesn't change, only the plumbing does.
from tot_minimal import Thought, generate, evaluate


class ToTState(TypedDict):
    task: str
    frontier: list[Thought]
    depth_left: int
    k: int
    beam: int


def expand(state: ToTState) -> ToTState:
    """One layer of the tree: branch -> evaluate -> prune (beam search)."""
    candidates: list[Thought] = []
    for t in state["frontier"]:
        candidates.extend(generate(state["task"], t, state["k"]))   # branch
    for c in candidates:
        c.score = evaluate(state["task"], c)                        # evaluate
    candidates.sort(key=lambda c: c.score, reverse=True)
    return {
        **state,
        "frontier": candidates[: state["beam"]],                    # prune
        "depth_left": state["depth_left"] - 1,
    }


def keep_going(state: ToTState) -> str:
    """Conditional edge: loop back to expand, or stop."""
    return "expand" if state["depth_left"] > 0 else END


graph = StateGraph(ToTState)
graph.add_node("expand", expand)
graph.add_edge(START, "expand")
graph.add_conditional_edges("expand", keep_going, {"expand": "expand", END: END})
app = graph.compile()


if __name__ == "__main__":
    out = app.invoke(
        {
            "task": "Use 4, 9, 10, 13 and + - * / to make 24.",
            "frontier": [Thought(steps=[])],
            "depth_left": 3,
            "k": 3,
            "beam": 2,
        }
    )
    best = out["frontier"][0]
    print(best.text, "->", best.score)
