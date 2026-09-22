"""
Convenience entry point.

    python main.py api        -> run the FastAPI backend
    python main.py ui         -> run the Streamlit UI
    python main.py ingest FILE.pdf   -> ingest a single PDF from the CLI
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "api":
        from app.config import settings

        subprocess.run(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", settings.api_host, "--port", str(settings.api_port), "--reload"]
        )
    elif command == "ui":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])
    elif command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest FILE.pdf")
            sys.exit(1)
        from pathlib import Path

        from app.rag.ingestion import ingest_pdf

        file_path = sys.argv[2]
        result = ingest_pdf(file_path, Path(file_path).name)
        print(result.document.model_dump_json(indent=2))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
