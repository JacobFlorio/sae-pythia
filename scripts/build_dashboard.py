import argparse

from datasets import load_dataset

from sae.dashboard import collect_max_activating, load_sae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output", default="dashboards/features.json")
    p.add_argument("--top-n", type=int, default=16)
    p.add_argument("--num-docs", type=int, default=2000)
    p.add_argument("--dataset", default="monology/pile-uncopyrighted")
    args = p.parse_args()

    # Initialize the streaming dataset BEFORE loading the SAE/model to GPU.
    # Loading datasets after CUDA context initialization causes a fork/thread
    # deadlock in the HuggingFace streaming pipeline on some systems.
    print("Initializing dataset stream...")
    dataset = load_dataset(args.dataset, split="train", streaming=True)

    sae = load_sae(args.checkpoint)
    collect_max_activating(
        sae=sae,
        model_name=args.model,
        layer=args.layer,
        output_path=args.output,
        top_n=args.top_n,
        num_docs=args.num_docs,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
