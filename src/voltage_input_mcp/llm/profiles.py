"""Model profiles, sized against real VRAM budgets.

Model choice here is not a matter of taste -- it is a packing problem. Two models must be
resident simultaneously, and on a 6 GB laptop GPU that rules out most of the obvious
answers. The profiles below state their memory cost explicitly so `voltage.doctor` can
tell the orchestrator which ones will actually run before anything is downloaded.

Why these models
----------------
**Vision: Qwen2.5-VL-3B-Instruct.** GUI grounding is a specific skill, not a side effect
of general captioning. Qwen2.5-VL is explicitly trained to emit bounding boxes for
on-screen elements and to work in a normalised coordinate space, which is exactly the one
question this layer is ever asked. A general captioner of the same size will describe a
screenshot beautifully and put the boxes in the wrong place. The 3B is chosen over the 7B
purely because the 7B plus an actuator does not fit in 6 GB.

**Actuator: Qwen3-1.7B.** The actuator's job is heavily constrained by a GBNF grammar and
a state brief -- it picks among a small set of legal continuations rather than reasoning
freely. That is close to the easiest thing a small instruct model can be asked to do, and
1.7B clears it while leaving room for the vision model. Its decode speed is what sets the
floor on cycle time, so smaller genuinely is better here up to the point of competence.

All repo and filename values below were verified against the HuggingFace API; the Qwen3
Q4_K_M quants come from `unsloth` because the official `Qwen/Qwen3-1.7B-GGUF` repo
publishes only Q8_0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ModelSpec", "Profile", "PROFILES", "EXPERIMENTAL", "DEFAULT_PROFILE",
    "recommend", "get_profile",
]

Role = Literal["vision", "actuator", "draft"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model plus the server tuning appropriate to its role.

    The two roles are bottlenecked on different things and must not be tuned identically:

    **Both are decode-bound.** This contradicts the obvious guess and cost a round of
    mis-tuning to discover. A 896x504 screenshot is 576 visual tokens (Qwen2.5-VL uses
    14x14 patches with a 2x2 merge, one token per 28x28 block), which *sounds* like the
    dominant cost -- but measured prefill is ~28 ms and flat from 448x252 to 896x504,
    while decode runs at ~22 ms/token. Image size is nearly free; output length is
    everything.

    The vision micro-batch below is therefore sized to take the whole image in one pass
    and then left alone: it is correct, but it is not where the time goes. The knobs that
    matter are `Perception.max_elements` (~21 tokens, ~500 ms per reported element) and
    the actuator's `note` length.

    There is a third, easily missed cost: **GBNF grammar evaluation runs on the CPU, once
    per sampled token.** It sits directly on the actuator's critical path, which is why
    the actuator is given real CPU threads despite being fully GPU-offloaded, and why
    keeping the grammar small (restricting `allow_keys`) is a latency optimisation and
    not only a safety one.
    """

    role: Role
    label: str
    hf_repo: str
    hf_file: str
    mmproj_repo: str | None = None
    mmproj_file: str | None = None
    ollama_tag: str | None = None
    params_b: float = 0.0
    quant: str = "Q4_K_M"
    weights_mb: int = 0
    n_ctx: int = 4096
    port: int = 8080
    gpu_layers: int = 99
    # Prefill batching. `ubatch` is the real knob: it is how many tokens go through the
    # GPU in one pass.
    batch_size: int = 512
    ubatch_size: int = 512
    # CPU threads. Matters for sampling and grammar evaluation even at full GPU offload.
    threads: int = 4
    extra_args: tuple[str, ...] = ()

    @property
    def is_vision(self) -> bool:
        return self.role == "vision"

    @property
    def kv_mb(self) -> int:
        """Approximate KV-cache cost at `n_ctx` with q8_0 K/V.

        A heuristic, not a derivation -- exact cost needs the layer count and GQA head
        configuration from the GGUF. It is deliberately a slight overestimate so the
        profile-fits check errs toward refusing a config rather than thrashing.
        """
        return int(self.params_b * self.n_ctx * 0.012) or 48

    @property
    def compute_buffer_mb(self) -> int:
        """Rough CUDA compute buffer, which scales with the micro-batch."""
        return max(48, int(self.ubatch_size * self.params_b * 0.06))

    @property
    def mmproj_mb(self) -> int:
        return 700 if self.mmproj_file else 0

    @property
    def total_mb(self) -> int:
        return self.weights_mb + self.mmproj_mb + self.kv_mb + self.compute_buffer_mb

    def llama_server_args(
        self,
        *,
        host: str = "127.0.0.1",
        cpu_only: bool = False,
        model_dir: str | None = None,
        slot_cache_dir: str | None = None,
    ) -> list[str]:
        """Render the tuned llama-server command line for this model."""
        args = ["llama-server", "--host", host, "--port", str(self.port)]

        if model_dir:
            args += ["--model", f"{model_dir}/{self.hf_file}"]
            if self.mmproj_file:
                args += ["--mmproj", f"{model_dir}/{self.mmproj_file}"]
        else:
            args += ["-hf", f"{self.hf_repo}:{self.quant}"]

        args += [
            "-c", str(self.n_ctx),
            "--parallel", "1",
            "-ngl", "0" if cpu_only else str(self.gpu_layers),
            "--batch-size", str(self.batch_size),
            "--ubatch-size", str(self.ubatch_size),
            "--threads", str(self.threads),
            # q8_0 K/V roughly halves cache memory at no measurable quality cost for
            # prompts this short, and on a 6 GB card that headroom is the whole game.
            # Requires a build with GGML_CUDA_FA_ALL_QUANTS=ON to stay on the fast path.
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "-fa", "on",
            # Reuse a cached prefix even when a chunk in the middle changed. Prompts here
            # are static-prefix + small dynamic tail by construction, so this hits on
            # essentially every cycle.
            "--cache-reuse", "512",
            # Load fully resident. mmap leaves first-touch page faults to happen during
            # the first few cycles, which shows up as mysterious early-run jitter.
            "--no-mmap",
            # Expose /metrics so bench.sh can read real prefill/decode timings.
            "--metrics",
        ]

        # Persist the prompt cache across restarts. The static prefix is identical every
        # run, so this removes the cold-start penalty on the first cycle after a restart.
        if slot_cache_dir and not cpu_only:
            args += ["--slot-save-path", slot_cache_dir]

        args += list(self.extra_args)
        return args

    @property
    def env(self) -> dict[str, str]:
        """Environment the server must run with."""
        return {
            # If this is 1, a VRAM overflow silently spills into host memory over PCIe
            # instead of failing. The server starts, works, and is ~10x slower, with no
            # error anywhere. Pin it off so overflow is loud.
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "0",
        }


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    description: str
    vision: ModelSpec
    actuator: ModelSpec
    min_vram_mb: int
    notes: str = ""
    warning: str = ""
    actuator_on_cpu: bool = False
    expected_cycle_ms: tuple[int, int] = (0, 0)
    draft: ModelSpec | None = None

    @property
    def vram_mb(self) -> int:
        total = self.vision.total_mb
        if not self.actuator_on_cpu:
            total += self.actuator.total_mb
        if self.draft is not None:
            total += self.draft.total_mb
        # CUDA context and cuBLAS workspaces, per process. Roughly 250-300 MB each on an
        # Ampere card, and there are two servers.
        return total + 550

    def fits(self, available_mb: int) -> bool:
        return self.vram_mb <= available_mb

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "vision": self.vision.label,
            "actuator": self.actuator.label + (" (CPU)" if self.actuator_on_cpu else ""),
            "estimated_vram_mb": self.vram_mb,
            "min_vram_mb": self.min_vram_mb,
            "expected_cycle_ms": list(self.expected_cycle_ms),
            "notes": self.notes,
            "warning": self.warning,
            "experimental": self.name in EXPERIMENTAL,
        }


# -- model definitions --------------------------------------------------------------------

QWEN_VL_3B = ModelSpec(
    role="vision",
    label="Qwen2.5-VL-3B-Instruct",
    hf_repo="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
    hf_file="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
    mmproj_repo="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
    mmproj_file="mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
    ollama_tag="qwen2.5vl:3b",
    params_b=3.0,
    weights_mb=1900,
    # 2048 is ample: 576 image tokens + a ~120 token instruction + a ~60 token reply.
    # Every token of context costs KV cache, and KV is what we are short of.
    n_ctx=2048,
    port=8080,
    # 768 covers all 576 image tokens plus the instruction in a single GPU pass, so the
    # prompt is prefilled without splitting. Larger buys nothing here; smaller splits the
    # image across passes and loses throughput.
    batch_size=1024,
    ubatch_size=768,
    # Fully GPU-offloaded and produces few tokens, so it barely touches the CPU.
    threads=4,
)

QWEN_VL_7B = ModelSpec(
    role="vision",
    label="Qwen2.5-VL-7B-Instruct",
    hf_repo="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
    hf_file="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    mmproj_repo="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
    mmproj_file="mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
    ollama_tag="qwen2.5vl:7b",
    params_b=7.0,
    weights_mb=4700,
    n_ctx=2048,
    port=8080,
    batch_size=1024,
    ubatch_size=768,
    threads=4,
)

QWEN3_1_7B = ModelSpec(
    role="actuator",
    label="Qwen3-1.7B",
    hf_repo="unsloth/Qwen3-1.7B-GGUF",
    hf_file="Qwen3-1.7B-Q4_K_M.gguf",
    ollama_tag="qwen3:1.7b",
    params_b=1.7,
    weights_mb=1100,
    # The actuator prompt is a static system block plus a short per-cycle tail -- around
    # 500 tokens in practice. 2048 leaves generous headroom and keeps KV small.
    n_ctx=2048,
    port=8081,
    # Decode-bound: micro-batch size does not matter, and a small one keeps the compute
    # buffer (and therefore VRAM) minimal.
    batch_size=512,
    ubatch_size=256,
    # Deliberately higher than the vision model despite full GPU offload. GBNF grammar
    # evaluation runs on the CPU once per sampled token and sits on the critical path;
    # starving it here directly raises per-token latency.
    threads=8,
)

QWEN3_4B = ModelSpec(
    role="actuator",
    label="Qwen3-4B-Instruct-2507",
    hf_repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
    hf_file="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    ollama_tag="qwen3:4b",
    params_b=4.0,
    weights_mb=2500,
    n_ctx=2048,
    port=8081,
    batch_size=512,
    ubatch_size=256,
    threads=8,
)

QWEN3_0_6B_DRAFT = ModelSpec(
    role="draft",
    label="Qwen3-0.6B (draft)",
    hf_repo="unsloth/Qwen3-0.6B-GGUF",
    hf_file="Qwen3-0.6B-Q4_K_M.gguf",
    params_b=0.6,
    weights_mb=400,
    n_ctx=4096,
    port=8082,
)


# -- profiles ------------------------------------------------------------------------------

PROFILES: dict[str, Profile] = {
    "lean": Profile(
        name="lean",
        description="Qwen2.5-VL-3B + Qwen3-1.7B, both on GPU. The 6 GB target.",
        vision=QWEN_VL_3B,
        actuator=QWEN3_1_7B,
        min_vram_mb=5200,
        expected_cycle_ms=(280, 700),
        notes=(
            "Fits a 6 GB card alongside a running desktop. Vision is gated to on-change "
            "by default, so most cycles cost only the actuator's ~90 ms."
        ),
    ),
    "balanced": Profile(
        name="balanced",
        description="Qwen2.5-VL-3B + Qwen3-4B. Better instruction following, needs 8 GB.",
        vision=QWEN_VL_3B,
        actuator=QWEN3_4B,
        min_vram_mb=7000,
        expected_cycle_ms=(380, 950),
        notes=(
            "Worth it when playbook states are complex enough that the 1.7B keeps "
            "picking the wrong legal action rather than emitting an illegal one."
        ),
    ),
    "split": Profile(
        name="split",
        description="Qwen2.5-VL-3B on GPU, Qwen3-1.7B on CPU. For very tight VRAM.",
        vision=QWEN_VL_3B,
        actuator=QWEN3_1_7B,
        actuator_on_cpu=True,
        min_vram_mb=3600,
        expected_cycle_ms=(400, 1100),
        notes=(
            "A 1.7B at Q4 decodes ~25-40 tok/s on 12 CPU cores, and grammar-constrained "
            "bursts are only 20-40 tokens, so the CPU actuator costs roughly +150 ms per "
            "cycle while freeing 1.4 GB of VRAM for the vision model."
        ),
    ),
    "quality": Profile(
        name="quality",
        description="Qwen2.5-VL-7B + Qwen3-4B. Best grounding, needs 12 GB.",
        vision=QWEN_VL_7B,
        actuator=QWEN3_4B,
        min_vram_mb=11000,
        expected_cycle_ms=(600, 1400),
        notes="Only worth it for dense or unusual UI where the 3B mislocates elements.",
    ),
    "ollama": Profile(
        name="ollama",
        description="Same models through Ollama. Zero build, no GBNF, ~2x slower.",
        vision=QWEN_VL_3B,
        actuator=QWEN3_1_7B,
        min_vram_mb=5200,
        expected_cycle_ms=(500, 1400),
        notes=(
            "Start here if you do not want to build llama.cpp. Bursts are constrained by "
            "a JSON schema rather than a grammar, so malformed bursts are possible and "
            "cost a cycle when they happen."
        ),
    ),
}

DEFAULT_PROFILE = "lean"


def get_profile(name: str) -> Profile:
    """Look up a profile, custom ones included."""
    from .custom import all_profiles

    available = all_profiles()
    try:
        return available[name]
    except KeyError:
        raise KeyError(
            f"unknown profile {name!r}; available: {', '.join(available)}"
        ) from None


# The desktop is already using the GPU. A compositor with a browser open holds
# 400-800 MB on a typical KDE/GNOME session, and this is a *laptop* GPU running the
# actual display. Recommending a profile that fits total VRAM rather than free VRAM
# produces a config that loads, then thrashes into system RAM the moment anything else
# draws a window -- which looks like the models being mysteriously slow.
DESKTOP_RESERVE_MB = 900


def usable_vram_mb(total_mb: int, reserve_mb: int = DESKTOP_RESERVE_MB) -> int:
    """VRAM actually available to the models, after the desktop's share."""
    return max(0, total_mb - reserve_mb)


def recommend(
    vram_mb: int, *, prefer_ollama: bool = False, reserve_mb: int = DESKTOP_RESERVE_MB
) -> Profile:
    """Largest profile that fits alongside a running desktop."""
    if prefer_ollama:
        return PROFILES["ollama"]
    from .custom import all_profiles

    budget = usable_vram_mb(vram_mb, reserve_mb)
    # Experimental profiles are never recommended: each trades something away that the
    # user has to consent to knowingly. `hyper` would win on VRAM every time and quietly
    # hand someone a vision model that cannot ground.
    ordered = sorted(
        (
            p for p in all_profiles().values()
            if p.name != "ollama" and p.name not in EXPERIMENTAL
        ),
        key=lambda p: -p.vram_mb,
    )
    for profile in ordered:
        if profile.fits(budget):
            return profile
    return PROFILES["split"]


def detect_vram_mb() -> int | None:
    """Total VRAM on the primary CUDA device, via nvidia-smi. None if unavailable."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, timeout=4.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.decode().strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


# ==========================================================================================
# Experimental profiles
# ==========================================================================================
#
# These trade one thing hard against another and are not defaults for good reasons. Each
# carries a `warning` that says what actually breaks, grounded in the measurements in
# `voltage bench` rather than in vibes.
#
# The measured facts that make these predictable:
#   * decode dominates, at roughly 22 ms/token on a 6 GB Ampere card
#   * decode speed scales with active parameters, so a 0.6B is ~3x faster than a 1.7B
#   * one reported vision element is ~21 tokens, so ~500 ms
#   * prefill is ~28 ms and flat -- image size is nearly free
#
# So the small end genuinely does raise the loop rate. What it costs is grounding, and
# grounding failure is not graceful: the click lands somewhere else.

SMOLVLM_500M = ModelSpec(
    role="vision",
    label="SmolVLM-Instruct (~500M)",
    hf_repo="ggml-org/SmolVLM-Instruct-GGUF",
    hf_file="SmolVLM-Instruct-Q4_K_M.gguf",
    mmproj_repo="ggml-org/SmolVLM-Instruct-GGUF",
    mmproj_file="mmproj-SmolVLM-Instruct-Q8_0.gguf",
    params_b=0.5, weights_mb=420, n_ctx=2048, port=8080,
    batch_size=1024, ubatch_size=768, threads=4,
)

QWEN3_0_6B = ModelSpec(
    role="actuator",
    label="Qwen3-0.6B",
    hf_repo="unsloth/Qwen3-0.6B-GGUF",
    hf_file="Qwen3-0.6B-Q4_K_M.gguf",
    ollama_tag="qwen3:0.6b",
    params_b=0.6, weights_mb=400, n_ctx=2048, port=8081,
    batch_size=512, ubatch_size=256, threads=8,
)

QWEN_VL_32B = ModelSpec(
    role="vision",
    label="Qwen2.5-VL-32B-Instruct",
    hf_repo="ggml-org/Qwen2.5-VL-32B-Instruct-GGUF",
    hf_file="Qwen2.5-VL-32B-Instruct-Q4_K_M.gguf",
    mmproj_repo="ggml-org/Qwen2.5-VL-32B-Instruct-GGUF",
    mmproj_file="mmproj-Qwen2.5-VL-32B-Instruct-Q8_0.gguf",
    params_b=32.0, weights_mb=19500, n_ctx=4096, port=8080,
    batch_size=2048, ubatch_size=1024, threads=6,
)

QWEN3_14B = ModelSpec(
    role="actuator",
    label="Qwen3-14B",
    hf_repo="unsloth/Qwen3-14B-GGUF",
    hf_file="Qwen3-14B-Q4_K_M.gguf",
    ollama_tag="qwen3:14b",
    params_b=14.0, weights_mb=9000, n_ctx=4096, port=8081,
    batch_size=512, ubatch_size=256, threads=8,
)

QWEN3_30B_A3B = ModelSpec(
    role="actuator",
    label="Qwen3-30B-A3B (MoE)",
    hf_repo="unsloth/Qwen3-30B-A3B-GGUF",
    hf_file="Qwen3-30B-A3B-Q4_K_M.gguf",
    ollama_tag="qwen3:30b-a3b",
    params_b=3.0,  # ACTIVE params -- decode speed tracks this, not the 30B total
    weights_mb=18500,
    n_ctx=4096, port=8081, batch_size=512, ubatch_size=256, threads=8,
)


EXPERIMENTAL: dict[str, Profile] = {
    "hyper": Profile(
        name="hyper",
        description="SmolVLM-500M + Qwen3-0.6B. Maximum loop rate, worst grounding.",
        vision=SMOLVLM_500M,
        actuator=QWEN3_0_6B,
        min_vram_mb=2200,
        expected_cycle_ms=(60, 200),
        notes=(
            "Roughly 3-4x the loop rate of `lean`: ~0.9 GB of weights and decode at a "
            "fraction of the cost. Suited to reflex-heavy playbooks where probes do the "
            "real work and the models mostly pick between a few pre-planned bursts."
        ),
        warning=(
            "SmolVLM-500M is NOT a grounding model. It will return boxes, and they will "
            "often be wrong -- and a wrong box is a click in the wrong place, not a "
            "graceful degradation. Only use this where `watch` is empty or where every "
            "click is fenced by click_allow_regions and require_target_element. Run "
            "`voltage compare` against a fixture before trusting it with dry_run off."
        ),
    ),
    "fast": Profile(
        name="fast",
        description="Qwen2.5-VL-3B + Qwen3-0.6B. Keeps real grounding, faster decisions.",
        vision=QWEN_VL_3B,
        actuator=QWEN3_0_6B,
        min_vram_mb=4000,
        expected_cycle_ms=(120, 500),
        notes=(
            "The honest speed profile. Vision is unchanged from `lean`, so grounding is "
            "unchanged; only the actuator shrinks. Under a GBNF grammar the actuator is "
            "choosing among a handful of legal continuations, which a 0.6B can often do."
        ),
        warning=(
            "A 0.6B follows a `brief` less reliably. It stays *legal* -- the grammar "
            "guarantees that -- but is likelier to pick a legal-but-wrong action, so "
            "expect more wasted cycles on multi-step states. Keep briefs to one "
            "instruction and prefer more states over more complex ones."
        ),
    ),
    "beefy": Profile(
        name="beefy",
        description="Qwen2.5-VL-32B + Qwen3-14B. Best grounding. Needs ~32 GB.",
        vision=QWEN_VL_32B,
        actuator=QWEN3_14B,
        min_vram_mb=31000,
        expected_cycle_ms=(900, 2500),
        notes=(
            "For dense, unusual or low-contrast UI where the 3B mislocates elements. "
            "Worth it when grounding is the failure mode and latency is not."
        ),
        warning=(
            "Slow, and slowness compounds here: decode scales with parameters, and a "
            "reported element already costs ~21 tokens. Expect 1-2.5 s per perceived "
            "cycle. Set perception.mode to 'on_change' and max_elements to 2-3, or the "
            "loop will crawl. Unusable for games."
        ),
    ),
    "beefy_moe": Profile(
        name="beefy_moe",
        description="Qwen2.5-VL-32B + Qwen3-30B-A3B. Big capacity at 3B decode speed.",
        vision=QWEN_VL_32B,
        actuator=QWEN3_30B_A3B,
        min_vram_mb=40000,
        expected_cycle_ms=(700, 2000),
        notes=(
            "The interesting one for this workload. Qwen3-30B-A3B is a mixture of "
            "experts with ~3B active parameters, so it decodes at roughly 3B speed while "
            "reasoning with 30B capacity -- and decode is exactly what bottlenecks this "
            "loop. A much better actuator than a dense 14B at similar or better latency."
        ),
        warning=(
            "The 30B total still has to be resident: ~18.5 GB for the actuator alone, "
            "~40 GB with the 32B vision model. Only the active experts are fast, not the "
            "memory. Needs a 48 GB card or two GPUs."
        ),
    ),
    "cpu_only": Profile(
        name="cpu_only",
        description="Qwen2.5-VL-3B + Qwen3-0.6B, both on CPU. No GPU required.",
        vision=QWEN_VL_3B,
        actuator=QWEN3_0_6B,
        actuator_on_cpu=True,
        min_vram_mb=0,
        expected_cycle_ms=(2000, 8000),
        notes="Proves the pipeline works without a GPU. Useful for authoring and "
              "dry-run testing playbooks on a laptop.",
        warning=(
            "Several seconds per cycle. Fine for validating a playbook's logic in "
            "dry_run, useless for driving anything in real time."
        ),
    ),
}
