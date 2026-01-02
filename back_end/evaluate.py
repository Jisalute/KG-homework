# evaluate.py
import json
import re
import csv
from collections import defaultdict
from handler import query_handler
from typing import List

with open("test_cases.json", "r", encoding="utf-8") as f:
    TEST_CASES = json.load(f)

def normalize_answer(text: str) -> set:
    if not text:
        return set()
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return set(text.split())

def answer_f1(pred: str, gold: List[str]) -> float:
    pred_tokens = normalize_answer(pred)
    gold_tokens = normalize_answer(" ".join(gold))
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

def classify_error(question: str, golden: List[str], system_pred: List[str], matched: bool) -> str:
    if not matched:
        return "pattern_mismatch"
    if not system_pred:
        return "kg_missing"  # KG 中无此三元组
    if set(system_pred) == set(golden):
        return "correct"
    return "wrong_retrieval"

def get_relation_type(question: str) -> str:
    if "作词" in question:
        return "作词"
    elif "唱" in question or "歌手" in question:
        return "歌手"
    else:
        return "其他"

def evaluate():
    total = len(TEST_CASES)
    f1_total = 0.0
    hits_at_1 = 0
    hdr_numerator = 0
    hdr_denominator = 0

    error_stats = defaultdict(int)
    relation_stats = {"歌手": {"p":0, "r":0, "f1":0, "count":0}, "作词": {"p":0, "r":0, "f1":0, "count":0}}

    # 存储每条结果用于写入 CSV
    results_rows = []

    for test_case in TEST_CASES:
        question = test_case["question"]
        golden = test_case["golden_answer"]
        llm_ans = test_case["llm_answer"]

        # 调用你的系统
        print(f"\n[评估中] 调用 query_handler 处理问题: {question}")
        res = query_handler(question)
        print(f"[评估中] 返回结果: {res}")
        system_ans = res["data"] if res["state"] == 0 else []
        final_str = ", ".join(system_ans)

        # 判断是否匹配成功（模拟 handler 内部逻辑）
        matched = any([
            re.search(r"歌曲(.+)的作词人是", question),
            re.search(r"(.+)是谁唱的", question),
            re.search(r"谁唱的(.+)", question),
            re.search(r"谁作词的(.+)", question),
            re.search(r"(.+)是哪个专辑的", question),  # 新增专辑 pattern
        ])

        # Answer F1
        f1 = answer_f1(final_str, golden)
        f1_total += f1

        # Hits@1
        if system_ans and set(system_ans) & set(golden):
            hits_at_1 += 1

        # HDR
        llm_correct = set(normalize_answer(llm_ans)) >= set([g.lower() for g in golden])
        if not llm_correct:
            hdr_denominator += 1
            if set(system_ans) >= set(golden):
                hdr_numerator += 1

        # 错误分类
        err_type = classify_error(question, golden, system_ans, matched)
        error_stats[err_type] += 1

        # 关系类型
        rel = get_relation_type(question)
        if rel in relation_stats:
            relation_stats[rel]["count"] += 1
            relation_stats[rel]["f1"] += f1

        # 记录本条结果
        results_rows.append({
            "question": question,
            "golden_answer": "; ".join(golden) if golden else "",
            "llm_answer": llm_ans,
            "system_answer": "; ".join(system_ans),
            "f1_score": round(f1, 4),
            "error_type": err_type,
            "relation_type": rel
        })

    # 计算最终指标
    avg_f1 = f1_total / total * 100
    hits_at_1_rate = hits_at_1 / total * 100
    hdr = hdr_numerator / hdr_denominator * 100 if hdr_denominator > 0 else 0.0

    # 输出到控制台
    print("\n📊 官方评估指标 (Academic Standard):")
    print(f"   • Answer F1 Score : {avg_f1:6.2f}%")
    print(f"   • KG Hits@1       : {hits_at_1_rate:6.2f}%")
    print(f"   • Hallucination Correction Rate (HDR): {hdr:6.2f}%\n")

    print("🔍 错误分析:")
    for err, count in error_stats.items():
        print(f"   • {err:20s}: {count} ({count/total*100:5.1f}%)")

    print("\n📈 按关系类型表现:")
    for rel, stat in relation_stats.items():
        if stat["count"] > 0:
            avg_rel_f1 = stat["f1"] / stat["count"] * 100
            print(f"   • {rel:4s} F1: {avg_rel_f1:6.2f}% ({stat['count']} samples)")

    # === 写入 CSV 文件 ===
    output_file = "evaluation_results.csv"
    with open(output_file, "w", encoding="utf-8-sig", newline="") as csvfile:
        fieldnames = [
            "question",
            "golden_answer",
            "llm_answer",
            "system_answer",
            "f1_score",
            "error_type",
            "relation_type"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

        # 写入汇总行（可选）
        writer.writerow({
            "question": "=== SUMMARY ===",
            "golden_answer": "",
            "llm_answer": "",
            "system_answer": "",
            "f1_score": round(avg_f1 / 100, 4),
            "error_type": f"F1={avg_f1:.2f}%, Hits@1={hits_at_1_rate:.2f}%, HDR={hdr:.2f}%",
            "relation_type": ""
        })

    print(f"\n✅ 评估结果已保存至: {output_file}")

if __name__ == "__main__":
    evaluate()