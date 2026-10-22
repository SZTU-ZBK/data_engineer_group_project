"""Print recommendations from a saved model."""

import argparse

from src.recommend import recommend_for_user


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--model", choices=["itemcf", "lightfm"], default="itemcf")
    parser.add_argument("--stage", choices=["validation", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    result = recommend_for_user(args.user_id, stage=args.stage, k=args.top_k, algorithm=args.model)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
