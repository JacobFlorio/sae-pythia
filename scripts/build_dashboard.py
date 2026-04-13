import argparse

from sae.dashboard import collect_max_activating, load_sae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output", default="dashboards/features.json")
    p.add_argument("--top-n", type=int, default=16)
    p.add_argument("--num-docs", type=int, default=2000)
    args = p.parse_args()

    sae = load_sae(args.checkpoint)
    collect_max_activating(
        sae=sae,
        model_name=args.model,
        layer=args.layer,
        output_path=args.output,
        top_n=args.top_n,
        num_docs=args.num_docs,
    )


if __name__ == "__main__":
    main()
