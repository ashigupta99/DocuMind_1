import json
import sys
import time
sys.path.append("..")

from rag import *

# ------------ load documents ------------

uploaded_files = [
    open("lstm.pdf", "rb"),
    open("image_analysis_notes.pdf", "rb")
]

documents, warnings = load_documents(uploaded_files)

if warnings:
    print(f"[WARN] {len(warnings)} warning(s) during loading:")
    for w in warnings:
        print(f"  - {w}")

chunks = chunk_documents(documents)
index, chunks = build_index(chunks)

bm25_index = build_bm25_index(chunks)

print(f"[INFO] Indexed {len(chunks)} chunks from {len(documents)} pages.\n")

# ---------- load evaluation questions ----------

with open("evaluation_suite.json", "r") as f:
    suite = json.load(f)

results = []
failed_ids = []

for item in suite:
    q_id = item["id"]
    q = item["question"]
    category = item["category"]

    print(f"[{q_id:02d}/{len(suite)}] ({category}) {q[:70]}...")

    try:
        response = generate_answer(
            q,
            index, bm25_index,
            chunks,
            history=None
        )
        answer = response["answer"]
        sources = response["sources"]

        # Flag rate limit hits so you can spot them in results
        if "rate limit" in answer.lower():
            print(f"  ⚠ Rate limited — will retry after 30s")
            time.sleep(30)
            response = generate_answer(q, index, bm25_index, chunks, history=None)
            answer = response["answer"]
            sources = response["sources"]

    except Exception as e:
        print(f"  ✗ Exception: {e}")
        answer = f"ERROR: {e}"
        sources = []
        failed_ids.append(q_id)

    results.append({
        "id": q_id,
        "category": category,
        "question": q,
        "answer": answer,
        "sources": sources
    })

    # Delay between calls to stay under Groq rate limit.
    # 2s is enough for most runs; increase to 5s if you still hit limits.
    time.sleep(2)

# ---------- save results ----------

with open("generated_answers1.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"\n✓ Done. {len(results)} answers saved to generated_answers.json.")

if failed_ids:
    print(f"✗ Failed (exception) on IDs: {failed_ids}")

# ---------- quick category summary ----------

rate_limited = [r for r in results if "rate limit" in r["answer"].lower()]
refused = [r for r in results if "couldn't find" in r["answer"].lower()]
answered = [r for r in results if r not in rate_limited and r not in refused]

print(f"\n--- Quick summary ---")
print(f"  Answered:     {len(answered)}")
print(f"  Refused:      {len(refused)}")
print(f"  Rate limited: {len(rate_limited)}")
print(f"  Errors:       {len(failed_ids)}")


# import json
# import time

# from rag import *

# # Rebuild index
# uploaded_files = [
#     open("lstm.pdf", "rb"),
#     open("image_analysis_notes.pdf", "rb")
# ]

# documents, warnings = load_documents(uploaded_files)

# chunks = chunk_documents(documents)

# index, chunks = build_index(chunks)

# bm25_index = build_bm25_index(chunks)

# # Load old answers
# with open("generated_answers1.json", "r") as f:
#     results = json.load(f)

# for item in results:

#     if "rate limit" in item["answer"].lower():

#         print(f"Retrying ID {item['id']}...")

#         time.sleep(5)

#         response = generate_answer(
#             item["question"],
#             index, bm_index,
#             chunks,
#             history=None
#         )

#         item["answer"] = response["answer"]
#         item["sources"] = response["sources"]

# # Save updated file
# with open("generated_answers1.json", "w") as f:
#     json.dump(results, f, indent=4)

# print("Finished updating rate-limited answers.")