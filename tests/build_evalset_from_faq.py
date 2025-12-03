import yaml, json, os, random

FAQ = "data/faq_de.yaml"
OUT = "tests/eval_set.jsonl"

def main(n=20, seed=42):
    random.seed(seed)
    data = yaml.safe_load(open(FAQ, "r", encoding="utf-8")) or []
    rows = []
    for item in (data if isinstance(data, list) else []):
        q = (item.get("user_input") or item.get("question") or "").strip()
        a = (item.get("reference")  or item.get("answer")   or "").strip()
        if q and a:
            rows.append({"user_input": q, "reference": a})
    random.shuffle(rows)
    rows = rows[:n]
    os.makedirs("tests", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(rows)} items → {OUT}")

if __name__ == "__main__":
    main()