"""Execute the teaching notebook with this Python interpreter, without a user kernel install."""

import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook = nbformat.read(ROOT / "examples/quickstart.ipynb", as_version=4)
    with tempfile.TemporaryDirectory(prefix="rstats-kernel-") as temp:
        kernel = Path(temp) / "rstats"
        kernel.mkdir()
        import json

        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Rstats validation",
                    "language": "python",
                }
            )
        )
        manager = KernelSpecManager(kernel_dirs=[temp])
        from jupyter_client import KernelManager

        km = KernelManager(kernel_name="rstats", kernel_spec_manager=manager, transport="ipc")
        client = NotebookClient(
            notebook, km=km, timeout=120, resources={"metadata": {"path": str(ROOT)}}
        )
        try:
            client.execute()
        finally:
            if client.kc is not None:
                client.kc.stop_channels()
            if km.has_kernel:
                km.shutdown_kernel(now=True)
    nbformat.write(notebook, ROOT / "examples/executed.ipynb")
    images = [
        output
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    if len(images) < 3:
        raise RuntimeError("The teaching notebook must render its three plots inline.")
    print("Executed all teaching notebook cells successfully.")


if __name__ == "__main__":
    main()
