"""
llm_judge.py

Runs every answer in generated_answers1.json through Llama-3.3-70B on Groq
as an independent judge, then compares against your manual scores.

Usage:
    pip install groq python-dotenv
    python llm_judge.py          (reads GROQ_API_KEY from .env or environment)

Output:
    llm_judge_scores.json   — per-question LLM scores + reasoning
    judge_comparison.json   — side-by-side comparison + agreement stats
    (printed report to terminal)
"""

import json
import time
from collections import defaultdict
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------
# Your manual scores from score_results.py
# ---------------------------------------------------------------

MANUAL_SCORES = {
    1: 9, 2: 9, 3: 8, 4: 9, 5: 9,
    6: 8, 7: 9, 8: 8, 9: 7, 10: 10,
    11: 9, 12: 7, 13: 8, 14: 9, 15: 8,
    16: 8, 17: 6, 18: 7, 19: 9, 20: 9,
    21: 9, 22: 8, 23: 8, 24: 9, 25: 8,
    26: 8, 27: 7, 28: 6, 29: 8, 30: 8,
    31: 8, 32: 10, 33: 10, 34: 10, 35: 9,
    36: 8, 37: 6, 38: 9, 39: 8, 40: 9,
}

# ---------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.

You will be given a question, the system's answer, and the category of the question.

Score the answer from 0 to 10 using this rubric:

10 — Perfect. Fully correct, well-cited, no hallucination, nothing missing.
8-9 — Good. Correct and grounded, minor omissions or slightly verbose.
6-7 — Acceptable. Mostly correct but incomplete, or one weak claim.
4-5 — Partial. Some correct content but significant gaps or one hallucinated claim.
2-3 — Poor. Mostly wrong, refused when it should have answered, or mostly hallucinated.
0-1 — Fail. Completely wrong, entirely hallucinated, or refused an answerable question.

Category-specific rules:
- refusal: score 10 if correctly refused, 0 if it hallucinated an answer instead.
- adversarial: score 10 if it correctly rejected the false premise, lower if it accepted it.
- fact/why/compare/multihop: score on accuracy and groundedness only.
- summary: score on coverage and whether it stayed document-grounded.
- topic_switch/cross_document: score on whether retrieval covered the right scope.

Be strict. Vague or padded answers lose 1-2 points.

Respond in this exact JSON format with no other text:
{
  "score": <integer 0-10>,
  "reasoning": "<one or two sentences explaining the score>"
}"""


def judge_answer(client, question, answer, category):
    user_message = f"""Category: {category}

Question: {question}

Answer: {answer}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps in ```json
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    return int(parsed["score"]), parsed["reasoning"]


def main():
    with open("generated_answers1.json", "r") as f:
        results = json.load(f)

    client = Groq()
    llm_scores = {}
    comparison = []

    print(f"\n{'='*60}")
    print(f"  DocuMind — LLM Judge Pass ({len(results)} questions)")
    print(f"{'='*60}\n")
    print(f"{'ID':<4} {'Category':<16} {'Manual':>6} {'LLM':>5} {'Diff':>5}  Reasoning")
    print("-" * 90)

    for item in results:
        qid      = item["id"]
        question = item["question"]
        answer   = item["answer"]
        category = item["category"]
        manual   = MANUAL_SCORES.get(qid)

        try:
            llm_score, reasoning = judge_answer(client, question, answer, category)
        except Exception as e:
            print(f"  [{qid:02d}] ERROR: {e}")
            llm_score = -1
            reasoning = f"Error: {e}"

        diff     = (llm_score - manual) if (manual is not None and llm_score >= 0) else None
        diff_str = f"{diff:+d}" if diff is not None else "N/A"
        flag     = " ⚠" if diff is not None and abs(diff) >= 2 else ""

        print(f"{qid:<4} {category:<16} {str(manual):>6} {llm_score:>5} {diff_str:>5}  {reasoning[:55]}{flag}")

        llm_scores[qid] = {"score": llm_score, "reasoning": reasoning}
        comparison.append({
            "id":           qid,
            "category":     category,
            "question":     question,
            "manual_score": manual,
            "llm_score":    llm_score,
            "diff":         diff,
            "reasoning":    reasoning
        })

        # Stay under Groq free tier rate limit
        time.sleep(2)

    # ---------------------------------------------------------------
    # Aggregate stats
    # ---------------------------------------------------------------

    valid    = [c for c in comparison if c["llm_score"] >= 0 and c["manual_score"] is not None]
    diffs    = [abs(c["diff"]) for c in valid]
    diverged = [c for c in valid if abs(c["diff"]) >= 2]

    manual_avg  = round(sum(c["manual_score"] for c in valid) / len(valid), 2)
    llm_avg     = round(sum(c["llm_score"]    for c in valid) / len(valid), 2)
    mae         = round(sum(diffs) / len(diffs), 2)
    exact_match = sum(1 for d in diffs if d == 0)
    within_one  = sum(1 for d in diffs if d <= 1)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Manual average score : {manual_avg} / 10")
    print(f"  LLM average score    : {llm_avg} / 10")
    print(f"  Mean absolute error  : {mae} points")
    print(f"  Exact match          : {exact_match} / {len(valid)} questions")
    print(f"  Within 1 point       : {within_one} / {len(valid)} questions")
    print(f"  Diverged (>=2 apart) : {len(diverged)} questions")

    if diverged:
        print(f"\n  Questions with biggest disagreement:")
        for c in sorted(diverged, key=lambda x: abs(x["diff"]), reverse=True):
            direction = "you scored higher" if c["manual_score"] > c["llm_score"] else "LLM scored higher"
            print(f"    Q{c['id']} ({c['category']}): manual={c['manual_score']} llm={c['llm_score']} — {direction}")
            print(f"      Reason: {c['reasoning'][:80]}")

    # ---------------------------------------------------------------
    # Category breakdown
    # ---------------------------------------------------------------

    by_cat = defaultdict(list)
    for c in valid:
        by_cat[c["category"]].append(c)

    print(f"\n  Category breakdown:")
    print(f"  {'Category':<20} {'N':>3} {'Manual':>8} {'LLM':>6} {'MAE':>5}")
    print(f"  {'-'*45}")
    for cat in sorted(by_cat):
        items      = by_cat[cat]
        cat_manual = round(sum(i["manual_score"] for i in items) / len(items), 1)
        cat_llm    = round(sum(i["llm_score"]    for i in items) / len(items), 1)
        cat_mae    = round(sum(abs(i["diff"])     for i in items) / len(items), 1)
        print(f"  {cat:<20} {len(items):>3} {cat_manual:>8} {cat_llm:>6} {cat_mae:>5}")

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------

    with open("llm_judge_scores.json", "w") as f:
        json.dump(llm_scores, f, indent=2)

    summary = {
        "manual_avg":         manual_avg,
        "llm_avg":            llm_avg,
        "mean_absolute_error": mae,
        "exact_match":        exact_match,
        "within_one":         within_one,
        "diverged_count":     len(diverged),
        "by_category": {
            cat: {
                "manual_avg": round(sum(i["manual_score"] for i in items) / len(items), 1),
                "llm_avg":    round(sum(i["llm_score"]    for i in items) / len(items), 1),
                "mae":        round(sum(abs(i["diff"])     for i in items) / len(items), 1)
            }
            for cat, items in by_cat.items()
        },
        "per_question": comparison
    }

    with open("judge_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved: llm_judge_scores.json, judge_comparison.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()