"""
main.py - Markdown RAG PoC の CLIエントリーポイント

使い方:
    # インデックス作成
    python main.py index <markdown_directory> [--db <db_path>]

    # 質問（一回）
    python main.py ask "<question>" [--db <db_path>]

    # 対話モード
    python main.py chat [--db <db_path>]
"""

import argparse
import os
import sys


def check_env() -> bool:
    """必要な環境変数が設定されているか確認する"""
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        print(f"[ERROR] 環境変数が設定されていません: {', '.join(missing)}")
        return False
    return True


def cmd_index(args: argparse.Namespace) -> None:
    """インデックス作成コマンド"""
    from indexer import build_index

    build_index(args.directory, args.db)


def cmd_ask(args: argparse.Namespace) -> None:
    """一回質問コマンド"""
    from query import answer_question

    print(f"\n質問: {args.question}\n")
    answer, sources = answer_question(args.question, args.db, top_k=args.top_k)

    print("=== 回答 ===")
    print(answer)
    print("\n=== 参照ソース ===")
    if sources:
        for s in sources:
            print(f"  - {s}")
    else:
        print("  (なし)")


def cmd_chat(args: argparse.Namespace) -> None:
    """対話モード"""
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
        answer, sources = answer_question(question, args.db, top_k=args.top_k)

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
        description="Markdown RAG PoC - MarkdownファイルをRAGで検索・回答"
    )
    parser.add_argument("--db", default="./chroma_db", help="ChromaDB の保存パス (default: ./chroma_db)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # index コマンド
    index_parser = subparsers.add_parser("index", help="Markdownファイルをインデックス化する")
    index_parser.add_argument("directory", help="対象の Markdown ディレクトリパス")

    # ask コマンド
    ask_parser = subparsers.add_parser("ask", help="一回だけ質問する")
    ask_parser.add_argument("question", help="質問文")
    ask_parser.add_argument("--top-k", type=int, default=5, help="検索結果の上位件数 (default: 5)")

    # chat コマンド
    chat_parser = subparsers.add_parser("chat", help="対話モードで質問する")
    chat_parser.add_argument("--top-k", type=int, default=5, help="検索結果の上位件数 (default: 5)")

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
