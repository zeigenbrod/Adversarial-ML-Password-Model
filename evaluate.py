import json
import argparse
from collections import Counter
from zxcvbn import zxcvbn as zxcvbn_check

# helper functions
def score_passwords(passwords):
    """run a list of plaintext passwords through zxcvbn. returns the list of result dicts."""
    results = []
    for pwd in passwords:
        pwd = pwd.strip()
        if not pwd or len(pwd) < 2:
            continue
        try:
            r = zxcvbn_check(pwd)
            results.append({
                "password": pwd,
                "zxcvbn_score": r["score"],
                "guesses_log10": r.get("guesses_log10", 0),
                "feedback": r["feedback"].get("warning", ""),
                "is_adversarial_candidate": r["score"] >= 3,
            })
        except Exception:
            continue
    return results


def compute_stats(results):
    """compute summary statistics from a list of scored password dicts."""
    total = len(results)
    if total == 0:
        return {}

    scores = Counter(r["zxcvbn_score"] for r in results)
    adversarial = sum(1 for r in results if r["is_adversarial_candidate"])
    avg_log10 = sum(r["guesses_log10"] for r in results) / total
    avg_score = sum(r["zxcvbn_score"] for r in results) / total

    # score breakdown as percentages
    score_pct = {str(k): round(v / total * 100, 1) for k, v in sorted(scores.items())}

    return {
        "total": total,
        "score_distribution": dict(sorted(scores.items())),
        "score_distribution_pct": score_pct,
        "adversarial_count": adversarial,
        "adversarial_rate_pct": round(adversarial / total * 100, 1),
        "avg_guesses_log10": round(avg_log10, 3),
        "avg_zxcvbn_score": round(avg_score, 3),
    }


def pattern_breakdown(results):
    """attempt to classify passwords by pattern type and report adversarial rate per pattern. works on both the baseline and Tiny LLaMA output."""
    import re
    patterns = {
        "leet+num+special": [],
        "leet+num": [],
        "cap+num": [],
        "leet_only": [],
        "word+special": [],
        "word+num": [],
        "word_only": [],
        "other": [],
    }

    leet_chars = set("@31$70")

    for r in results:
        p = r["password"]
        has_upper = any(c.isupper() for c in p)
        has_digit = any(c.isdigit() for c in p)
        has_special = bool(re.search(r'[!@#$&]', p))
        has_leet = any(c in leet_chars for c in p)
        is_alpha = p.isalpha()

        if has_leet and has_digit and has_special:
            patterns["leet+num+special"].append(r)
        elif has_leet and has_digit:
            patterns["leet+num"].append(r)
        elif has_upper and has_digit and not has_leet:
            patterns["cap+num"].append(r)
        elif has_leet and not has_digit:
            patterns["leet_only"].append(r)
        elif has_special and not has_digit:
            patterns["word+special"].append(r)
        elif has_digit and not has_leet and not has_upper:
            patterns["word+num"].append(r)
        elif is_alpha:
            patterns["word_only"].append(r)
        else:
            patterns["other"].append(r)

    breakdown = {}
    for name, entries in patterns.items():
        if not entries:
            continue
        adv = sum(1 for e in entries if e["is_adversarial_candidate"])
        breakdown[name] = {
            "count": len(entries),
            "adversarial_count": adv,
            "adversarial_rate_pct": round(adv / len(entries) * 100, 1),
            "avg_score": round(sum(e["zxcvbn_score"] for e in entries) / len(entries), 2),
        }
    return breakdown


def top_adversarial_examples(results, n=10):
    """return top N adversarial passwords sorted by guesses_log10 descending."""
    candidates = [r for r in results if r["is_adversarial_candidate"]]
    candidates.sort(key=lambda x: x["guesses_log10"], reverse=True)
    return candidates[:n]


def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_stats(label, stats):
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  Total passwords:        {stats['total']}")
    print(f"  Adversarial rate:       {stats['adversarial_rate_pct']}%  ({stats['adversarial_count']} passwords scored 3–4)")
    print(f"  Avg zxcvbn score:       {stats['avg_zxcvbn_score']} / 4")
    print(f"  Avg guesses (log10):    {stats['avg_guesses_log10']}")
    print()
    print("  score distribution:")
    for score, count in stats["score_distribution"].items():
        pct = stats["score_distribution_pct"][str(score)]
        bar = "█" * int(pct / 2)
        print(f"    Score {score}: {bar:30s} {count:5d}  ({pct}%)")


def print_pattern_table(breakdown):
    print(f"\n  {'Pattern':<22} {'Count':>6}  {'Adv. Count':>10}  {'Adv. Rate':>10}  {'Avg Score':>10}")
    print(f"  {'─'*22} {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")
    sorted_patterns = sorted(breakdown.items(), key=lambda x: x[1]["adversarial_rate_pct"], reverse=True)
    for name, b in sorted_patterns:
        print(f"  {name:<22} {b['count']:>6}  {b['adversarial_count']:>10}  {b['adversarial_rate_pct']:>9.1f}%  {b['avg_score']:>10.2f}")


# main
def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial password generation.")
    parser.add_argument("--baseline", default="labeled_passwords.json",
                        help="Path to baseline labeled_passwords.json")
    parser.add_argument("--generated", default=None,
                        help="Path to Tiny LLaMA generated passwords (.txt, one per line)")
    args = parser.parse_args()

    report = {}

    # baseline evaluation
    print_section("Baseline Evaluation (labeled_passwords.json)")

    try:
        with open(args.baseline) as f:
            baseline_data = json.load(f)
        print(f"\n  Loaded {len(baseline_data)} passwords from {args.baseline}")
    except FileNotFoundError:
        print(f"\n  ERROR: Could not find {args.baseline}")
        print("  Make sure you run this script from the project directory.")
        return

    baseline_stats = compute_stats(baseline_data)
    print_stats("Baseline Dataset", baseline_stats)
    report["baseline"] = baseline_stats

    baseline_patterns = pattern_breakdown(baseline_data)
    print_section("Baseline Pattern Breakdown")
    print_pattern_table(baseline_patterns)
    report["baseline_patterns"] = baseline_patterns

    top_baseline = top_adversarial_examples(baseline_data)
    print_section("Baseline Top 10 Adversarial Canidates")
    print(f"\n  {'Password':<20} {'Score':>6}  {'log10 Guesses':>14}  {'Feedback'}")
    print(f"  {'─'*20} {'─'*6}  {'─'*14}  {'─'*30}")
    for r in top_baseline:
        fb = r.get("zxcvbn_feedback", r.get("feedback", ""))[:35]
        print(f"  {r['password']:<20} {r['zxcvbn_score']:>6}  {r['guesses_log10']:>14.3f}  {fb}")
    report["baseline_top_adversarial"] = top_baseline

    # Tiny LLaMA evaluation
    if args.generated:
        print_section("Tiny LLaMA Generated Password Evaluation")

        try:
            with open(args.generated) as f:
                raw_passwords = [line.strip() for line in f if line.strip()]
            print(f"\n  Loaded {len(raw_passwords)} generated passwords from {args.generated}")
        except FileNotFoundError:
            print(f"\n  Error: Could not find {args.generated}")
            raw_passwords = []

        if raw_passwords:
            generated_results = score_passwords(raw_passwords)
            generated_stats = compute_stats(generated_results)
            print_stats("Tiny LLaMA Generated Passwords", generated_stats)
            report["generated"] = generated_stats

            generated_patterns = pattern_breakdown(generated_results)
            print_section("Tiny LLaMA Pattern Breakdown")
            print_pattern_table(generated_patterns)
            report["generated_patterns"] = generated_patterns

            top_generated = top_adversarial_examples(generated_results)
            print_section("Tiny LLaMA Top 10 Adversarial Canidates")
            print(f"\n  {'Password':<20} {'Score':>6}  {'log10 Guesses':>14}  {'Feedback'}")
            print(f"  {'─'*20} {'─'*6}  {'─'*14}  {'─'*30}")
            for r in top_generated:
                print(f"  {r['password']:<20} {r['zxcvbn_score']:>6}  {r['guesses_log10']:>14.3f}  {r.get('feedback','')[:35]}")
            report["generated_top_adversarial"] = top_generated

            # head-to-head comparison
            print_section("head-to-head Comparison: Baseline vs Tiny LLaMA")
            b = baseline_stats
            g = generated_stats
            delta_fool = round(g["adversarial_rate_pct"] - b["adversarial_rate_pct"], 1)
            delta_score = round(g["avg_zxcvbn_score"] - b["avg_zxcvbn_score"], 3)
            delta_log10 = round(g["avg_guesses_log10"] - b["avg_guesses_log10"], 3)

            print(f"\n  {'Metric':<30} {'Baseline':>10}  {'Tiny LLaMA':>10}  {'Delta':>10}")
            print(f"  {'─'*30} {'─'*10}  {'─'*10}  {'─'*10}")
            print(f"  {'Adversarial rate (%)':<30} {b['adversarial_rate_pct']:>10}  {g['adversarial_rate_pct']:>10}  {delta_fool:>+10.1f}")
            print(f"  {'Avg zxcvbn score':<30} {b['avg_zxcvbn_score']:>10}  {g['avg_zxcvbn_score']:>10}  {delta_score:>+10.3f}")
            print(f"  {'Avg guesses (log10)':<30} {b['avg_guesses_log10']:>10}  {g['avg_guesses_log10']:>10}  {delta_log10:>+10.3f}")

            verdict = "IMPROVED" if delta_fool > 0 else "NO IMPROVEMENT" if delta_fool == 0 else "DECLINED"
            print(f"\n  Verdict: Tiny LLaMA fool rate {verdict} vs baseline ({delta_fool:+.1f}%)")
            report["comparison"] = {
                "delta_adversarial_rate_pct": delta_fool,
                "delta_avg_score": delta_score,
                "delta_avg_guesses_log10": delta_log10,
                "verdict": verdict,
            }

    # save report
    print_section("Report Saved")
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Results saved to evaluation_report.json")
    print()


if __name__ == "__main__":
    main()
