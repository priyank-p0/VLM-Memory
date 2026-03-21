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
            from visual_memory.seed import seed
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

    # ── 3b. Diagnose backend initialization ─────────────────────────────────
    print("[bridge] Diagnosing plugin backend ...")
    try:
        import importlib.util, os
        plugin_init = os.path.join(os.path.dirname(__file__), "visual-memory-plugin", "__init__.py")
        spec = importlib.util.spec_from_file_location("visual_memory_plugin", plugin_init)
        plugin_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin_mod)

        _db = getattr(plugin_mod, "_db", None)
        _encoder = getattr(plugin_mod, "_encoder", None)
        _vlm = getattr(plugin_mod, "_vlm", None)
        _DB_PATH = getattr(plugin_mod, "_DB_PATH", "unknown")
        use_mocks = getattr(plugin_mod, "USE_MOCKS", "unknown")

        print(f"[bridge]   USE_MOCKS = {use_mocks}")
        print(f"[bridge]   DB_PATH   = {_DB_PATH}")
        print(f"[bridge]   _db type  = {type(_db).__name__}")
        print(f"[bridge]   _enc type = {type(_encoder).__name__}")
        print(f"[bridge]   _vlm type = {type(_vlm).__name__}")

        if os.path.exists(_DB_PATH):
            print(f"[bridge]   DB file exists: YES ({os.path.getsize(_DB_PATH)} bytes)")
        else:
            print(f"[bridge]   DB file exists: NO — memories won't load!")

        if _db is not None:
            count = _db.count(dataset.name)
            print(f"[bridge]   _db.count('{dataset.name}') = {count}")
            if count == 0:
                total = _db.count()
                print(f"[bridge]   _db.count() (all datasets) = {total}")
                if total > 0:
                    print(f"[bridge]   WARNING: Memories exist but none match dataset_name='{dataset.name}'")
        # Close the DB connection so FiftyOne's plugin load won't hit DuckDB lock
        if _db is not None and hasattr(_db, "close"):
            _db.close()
            print("[bridge]   DB connection closed (freeing lock for FiftyOne)")
    except Exception as e:
        import traceback
        print(f"[bridge]   ERROR importing plugin internals: {e}")
        traceback.print_exc()

    # ── 4. Launch ─────────────────────────────────────────────────────────────
    if args.no_launch:
        print("[bridge] --no-launch set, skipping App. Validation complete.")
        return

    print("[bridge] Launching FiftyOne App ...")
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
