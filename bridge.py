"""One command to seed, load, and launch the VisualMemory demo."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="VisualMemory demo launcher")
    parser.add_argument("--seed", action="store_true", help="Run seed.py first")
    parser.add_argument("--no-launch", action="store_true", help="Validate only, don't open App")
    args = parser.parse_args()

    # ── 1. Optional seed ──────────────────────────────────────────────────────
    if args.seed:
        print("[bridge] Running seed.py ...")
        try:
            from seed import seed
            seed()
        except ImportError:
            print("[bridge] ERROR: seed.py not found — skipping seed step", file=sys.stderr)

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    print("[bridge] Loading FiftyOne zoo dataset 'quickstart' ...")
    import fiftyone as fo
    import fiftyone.zoo as foz

    dataset = foz.load_zoo_dataset("quickstart")
    print(f"[bridge] Dataset: {dataset.name!r}, {len(dataset)} samples")

    # ── 3. Verify plugin ──────────────────────────────────────────────────────
    import fiftyone.plugins as fop

    plugins = fop.list_plugins()
    plugin_names = [p.name for p in plugins]
    print(f"[bridge] Plugins loaded: {plugin_names}")

    if "visual-memory-plugin" not in plugin_names:
        print(
            "[bridge] WARNING: visual-memory-plugin not found.\n"
            "  Symlink it into your plugins dir:\n"
            "    PLUGINS_DIR=$(python -c \"import fiftyone as fo; print(fo.config.plugins_dir)\")\n"
            "    ln -s $(pwd)/visual-memory-plugin $PLUGINS_DIR/visual-memory-plugin",
            file=sys.stderr,
        )

    # ── 4. Launch ─────────────────────────────────────────────────────────────
    if args.no_launch:
        print("[bridge] --no-launch set, skipping App. Validation complete.")
        return

    print("[bridge] Launching FiftyOne App ...")
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
