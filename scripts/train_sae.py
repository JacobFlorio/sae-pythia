import argparse

from sae.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--d-sae", type=int, default=16_384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--tokens", type=int, default=50_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()

    train(
        model_name=args.model,
        layer=args.layer,
        d_sae=args.d_sae,
        k=args.k,
        total_tokens=args.tokens,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
