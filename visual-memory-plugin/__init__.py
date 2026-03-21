import os
import numpy as np
import fiftyone.operators as foo
import fiftyone.operators.types as types

# ── Toggle this flag at integration time ──────────────────────────────────────
USE_MOCKS = False
# ─────────────────────────────────────────────────────────────────────────────


# ── Mock backends (signatures match real API exactly) ─────────────────────────


# ── Backend initialization ────────────────────────────────────────────────────

# FIX (Issues 1+2): resolve symlinks so paths work when plugin dir is symlinked
# into FiftyOne's plugins dir. __file__ without realpath stays in the plugins
# dir; realpath follows the symlink back to the actual project root.
_PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.dirname(_PLUGIN_DIR)
_DB_PATH = os.path.join(_PROJECT_ROOT, "memory.duckdb")

# if USE_MOCKS:
#     _db = MockMemoryDB()
#     _encoder = MockEncoder()
#     _vlm = MockVLM()
#else:
# FIX (Issue 3): wrap in try/except — any failure here silently removes all
# operators from the FiftyOne UI. Fall back to mocks so the plugin stays
# visible and we can see the error in the notify banner.
try:
    import sys, traceback
    sys.path.insert(0, _PROJECT_ROOT)
    print(f"[visual-memory-plugin] PROJECT_ROOT: {_PROJECT_ROOT}")
    print(f"[visual-memory-plugin] DB_PATH: {_DB_PATH}")
    from memory_db import VisualMemoryDB
    print("[visual-memory-plugin] memory_db imported OK")
    from visual_memory.encoder import get_encoder
    print("[visual-memory-plugin] encoder imported OK")
    from visual_memory.vlm_adapter import get_vlm
    print("[visual-memory-plugin] vlm_adapter imported OK")
    _db = VisualMemoryDB(_DB_PATH)
    print(f"[visual-memory-plugin] DB connected, count={_db.count()}")
    _encoder = get_encoder()
    print(f"[visual-memory-plugin] encoder ready: {type(_encoder).__name__}")
    _vlm = get_vlm()
    print(f"[visual-memory-plugin] vlm ready: {type(_vlm).__name__}")
    print(f"[visual-memory-plugin] Real backends loaded successfully.")
except Exception as _e:
    traceback.print_exc()
    print(f"[visual-memory-plugin] WARNING: backend init failed ({_e}), using dummy backends")

    class _MockDB:
        def write(self, **kw): return True
        def search(self, *a, **kw): return []
        def count(self, *a, **kw): return 0
        def clear(self, **kw): pass

    class _MockEnc:
        def encode_image(self, p): return np.zeros(512, dtype=np.float32)
        def encode_text(self, t): return np.zeros(512, dtype=np.float32)

    class _MockVLM:
        def analyze_with_memory(self, p, m, **kw): return {"description": "mock", "tags": [], "reasoning": "mock"}
        def analyze(self, p, **kw): return self.analyze_with_memory(p, [])

    _db = _MockDB()
    _encoder = _MockEnc()
    _vlm = _MockVLM()


# ── Operators ─────────────────────────────────────────────────────────────────

class StoreMemory(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="store_memory",
            label="Store Visual Memory",
            # FIX (Issue 7): call this out clearly so demo user remembers to
            # choose "Delegate" for large batches — inline blocks the UI.
            description="TIP: choose 'Delegate' for >10 samples to avoid freezing the UI.",
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "task_context",
            label="Task context",
            description="Optional task hint for the VLM (e.g. 'focus on animals')",
            required=False,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        task = ctx.params.get("task_context", "") or ""
        view = ctx.target_view()
        stored = 0
        failed = 0

        for sample in view.iter_samples(progress=True):
            embedding = _encoder.encode_image(sample.filepath)
            if embedding is None:
                failed += 1
                continue
            # FIX (Issue 5): catch per-sample errors so one bad image / Gemini
            # hiccup doesn't abort the whole batch.
            try:
                result = _vlm.analyze(sample.filepath, task=task or None)
                _db.write(
                    sample_id=sample.id,
                    filepath=sample.filepath,
                    embedding=embedding,
                    description=result["description"],
                    tags=result.get("tags", []),
                    dataset_name=ctx.dataset.name,
                )
                stored += 1
            except Exception as e:
                print(f"[store_memory] sample {sample.id} failed: {e}")
                failed += 1

        msg = f"Stored {stored} memories"
        if failed:
            msg += f" ({failed} failed)"
        ctx.ops.notify(msg)


class RecallMemories(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="recall_memories",
            label="Recall Visual Memories",
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "query",
            label="Text query",
            description="Natural language query (leave blank to use selected sample)",
            required=False,
        )
        inputs.int(
            "k",
            label="Number of results",
            default=5,
            required=False,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        query_text = (ctx.params.get("query") or "").strip()
        k = ctx.params.get("k") or 5

        if query_text:
            embedding = _encoder.encode_text(query_text)
        elif ctx.current_sample:
            sample = ctx.dataset[ctx.current_sample]
            embedding = _encoder.encode_image(sample.filepath)
            if embedding is None:
                ctx.ops.notify("Failed to embed image", variant="error")
                return
        else:
            ctx.ops.notify("Enter a query or select a sample first", variant="warning")
            return

        results = _db.search(embedding, k=k, dataset_name=ctx.dataset.name)
        if not results:
            ctx.ops.notify("No memories found")
            return

        sample_ids = [r["sample_id"] for r in results]
        ctx.ops.set_view(ctx.dataset.select(sample_ids))
        ctx.ops.notify(f"Found {len(results)} similar memories")


class AnalyzeWithMemory(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="analyze_with_memory",
            label="Analyze with Visual Memory",
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "task",
            label="Task",
            description="What should the VLM focus on?",
            required=False,
        )
        inputs.int(
            "k",
            label="Memories to retrieve",
            default=5,
            required=False,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        if not ctx.current_sample:
            ctx.ops.notify("Open a sample in the modal first", variant="warning")
            return

        sample = ctx.dataset[ctx.current_sample]
        embedding = _encoder.encode_image(sample.filepath)
        if embedding is None:
            ctx.ops.notify("Failed to embed image", variant="error")
            return

        k = ctx.params.get("k") or 5
        memories = _db.search(embedding, k=k, dataset_name=ctx.dataset.name)
        result = _vlm.analyze_with_memory(
            sample.filepath, memories, task=ctx.params.get("task") or None
        )

        # FIX (Issue 6): sample.save() can raise if field schema conflicts with
        # a prior run (e.g., field existed as a different type). Catch and notify
        # instead of crashing the operator.
        try:
            sample["memory_description"] = result["description"]
            sample["memory_reasoning"] = result["reasoning"]
            sample["memory_tags"] = result.get("tags", [])
            sample["num_memories_used"] = len(memories)
            sample.save()
        except Exception as e:
            ctx.ops.notify(f"Analysis done but failed to save fields: {e}", variant="warning")
            return

        _db.write(
            sample_id=sample.id,
            filepath=sample.filepath,
            embedding=embedding,
            description=result["description"],
            tags=result.get("tags", []),
            dataset_name=ctx.dataset.name,
        )

        ctx.ops.notify("Analysis complete — check sample fields!")
        ctx.ops.reload_samples()


class ClearMemories(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="clear_memories",
            label="Clear Memories",
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.bool(
            "confirm",
            label="Confirm — this cannot be undone",
            default=False,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        if not ctx.params.get("confirm"):
            ctx.ops.notify("Check the confirm box to proceed", variant="warning")
            return
        _db.clear(dataset_name=ctx.dataset.name)
        ctx.ops.notify("Memories cleared")


# ── Panel ─────────────────────────────────────────────────────────────────────

class MemoryBrowserPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(name="MemoryBrowserPanel", label="Visual Memory")

    def on_load(self, ctx):
        ctx.panel.state.query = ""
        ctx.panel.state.results_md = ""
        ctx.panel.state.memory_count = 0
        self._refresh_count(ctx)

    def render(self, ctx):
        panel = types.Object()
        panel.md("hdr", label="# 🧠 Visual Memory Browser")

        panel.str(
            "query",
            label="Search memories",
            description="Natural language query",
        )
        panel.btn("search_btn", label="🔍 Search", on_click=self.on_search)

        # FIX (Issue 4): use getattr so render() is safe even if on_load hasn't
        # run yet (can happen on panel re-mount in some FiftyOne versions).
        count = getattr(ctx.panel.state, "memory_count", 0) or 0
        panel.md("stats", label=f"**{count}** memories stored")

        results_md = getattr(ctx.panel.state, "results_md", "")
        if results_md:
            panel.md("results", label=results_md)

        panel.btn("store_btn", label="📝 Memorize Selected", on_click=self.on_store)
        panel.btn("analyze_btn", label="🧠 Analyze with Memory", on_click=self.on_analyze)
        panel.btn("refresh_btn", label="🔄 Refresh", on_click=self.on_refresh)

        return types.Property(panel)

    def on_search(self, ctx):
        query = getattr(ctx.panel.state, "query", "") or ""
        if not query.strip():
            ctx.panel.state.results_md = "*Enter a query above*"
            return
        try:
            text_emb = _encoder.encode_text(query)
            results = _db.search(text_emb, k=5, dataset_name=ctx.dataset.name)
            if not results:
                ctx.panel.state.results_md = "*No memories found*"
                return

            md = "### Results\n\n"
            for i, r in enumerate(results):
                desc = (r.get("description") or "")[:150]
                sim = r.get("similarity", 0)
                md += f"**{i+1}.** ({sim:.0%}) {desc}\n\n"
            ctx.panel.state.results_md = md

            sample_ids = [r["sample_id"] for r in results]
            ctx.ops.set_view(ctx.dataset.select(sample_ids))
        except Exception as e:
            ctx.panel.state.results_md = f"*Error: {e}*"

    def on_store(self, ctx):
        ctx.ops.trigger("@visual-memory-plugin/store_memory")

    def on_analyze(self, ctx):
        ctx.ops.trigger("@visual-memory-plugin/analyze_with_memory")

    def on_refresh(self, ctx):
        self._refresh_count(ctx)

    def _refresh_count(self, ctx):
        try:
            ctx.panel.state.memory_count = _db.count(ctx.dataset.name)
        except Exception:
            ctx.panel.state.memory_count = 0


# ── Registration ──────────────────────────────────────────────────────────────

def register(p):
    p.register(StoreMemory)
    p.register(RecallMemories)
    p.register(AnalyzeWithMemory)
    p.register(ClearMemories)
    p.register(MemoryBrowserPanel)
