import argparse

from sae.autointerp import run_autointerp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dashboard", required=True, help="features.json from build_dashboard")
    p.add_argument("--output", default="dashboards/autointerp.json")
    p.add_argument("--num-features", type=int, default=None)
    p.add_argument("--model", default="claude-sonnet-4-5")
    args = p.parse_args()

    results = run_autointerp(
        dashboard_path=args.dashboard,
        output_path=args.output,
        num_features=args.num_features,
        model=args.model,
    )
    scored = [r for r in results if r.balanced_accuracy == r.balanced_accuracy]
    if scored:
        mean = sum(r.balanced_accuracy for r in scored) / len(scored)
        print(f"Scored {len(scored)} features; mean balanced accuracy = {mean:.3f}")


if __name__ == "__main__":
    main()
