"""
main.py - CLI entrypoint for the Markdown RAG PoC.

Usage:
    # Build index
    uv run markdown-rag index <markdown_directory> [--db <db_path>]

    # Ask a single question
    uv run markdown-rag ask "<question>" [--db <db_path>] [--no-expand] [--expansions N]

    # Interactive chat mode
    uv run markdown-rag chat [--db <db_path>] [--no-expand] [--expansions N]
"""

import argparse
import os
import sys


def check_env() -> bool:
    """Check that required API key environment variables are set."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] 環境変数が設定されていません: OPENAI_API_KEY")
        return False
    return True


def cmd_index(args: argparse.Namespace) -> None:
    """Run the index build command."""
    from indexer import build_index

    build_index(args.directory, args.db)


def cmd_ask(args: argparse.Namespace) -> None:
    """Run a single question and print the answer."""
    from query import answer_question

    print(f"\n質問: {args.question}\n")
    answer, sources = answer_question(
        args.question,
        args.db,
        top_k=args.top_k,
        expand=not args.no_expand,
        n_expansions=args.expansions,
    )

    print("=== 回答 ===")
    print(answer)
    print("\n=== 参照ソース ===")
    if sources:
        for s in sources:
            print(f"  - {s}")
    else:
        print("  (なし)")


def cmd_chat(args: argparse.Namespace) -> None:
    """Run interactive chat mode."""
    from query import answer_question

    print("=== Markdown RAG チャットモード ===")
    print("終了するには 'exit' または 'quit' を入力してください。\n")

    while True:
        try:
            question = input("質問: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("終了します。")
            break

        print()
        answer, sources = answer_question(
            question,
            args.db,
            top_k=args.top_k,
            expand=not args.no_expand,
            n_expansions=args.expansions,
        )

        print("=== 回答 ===")
        print(answer)
        print("\n=== 参照ソース ===")
        if sources:
            for s in sources:
                print(f"  - {s}")
        else:
            print("  (なし)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Markdown RAG PoC - Search Markdown files and answer questions"
    )
    parser.add_argument("--db", default="./chroma_db", help="ChromaDB storage path (default: ./chroma_db)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # index subcommand
    index_parser = subparsers.add_parser("index", help="Index Markdown files into ChromaDB")
    index_parser.add_argument("directory", help="Path to the Markdown directory")

    # ask subcommand
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Question text")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve (default: 5)")
    ask_parser.add_argument("--no-expand", action="store_true", help="Disable multi-query expansion")
    ask_parser.add_argument("--expansions", type=int, default=3, help="Number of expanded queries to generate (default: 3)")

    # chat subcommand
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat mode")
    chat_parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve (default: 5)")
    chat_parser.add_argument("--no-expand", action="store_true", help="Disable multi-query expansion")
    chat_parser.add_argument("--expansions", type=int, default=3, help="Number of expanded queries to generate (default: 3)")

    args = parser.parse_args()

    if not check_env():
        sys.exit(1)

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()
