"""
hardware/base_hardware.py

Hardware specification loader.

Each platform has its own dedicated YAML file in configs/hardware/:
  configs/hardware/cerebras_wse2.yaml
  configs/hardware/sambanova_sn30.yaml
  configs/hardware/graphcore_bow_ipu.yaml

To add a new platform: create configs/hardware/new_platform.yaml
No Python code changes needed.
"""

from dataclasses import dataclass
from typing import Optional
import yaml
import os


@dataclass
class HardwareSpec:
    """
    Unified hardware specification container.
    All bandwidth in GB/s. Memory in GB. FLOPS in TFLOPS.
    """
    name: str
    execution_model: str                     # 'wafer_scale' | 'dataflow' | 'bsp'

    # Compute
    peak_flops_fp16: float                   # TFLOPS
    peak_flops_bf16: float
    peak_flops_fp32: float

    # On-chip memory
    total_onchip_sram_gb: float
    onchip_bandwidth_gbps: float             # GB/s

    # Off-chip memory
    offchip_memory_gb: float
    offchip_bandwidth_gbps: float = 0.0

    # Multi-tier memory
    hbm_gb: float = 0.0
    hbm_bandwidth_gbps: float = 0.0
    ddr_gb: float = 0.0
    ddr_bandwidth_gbps: float = 0.0

    # Interconnect
    inter_chip_bandwidth_gbps: float = 0.0
    inter_chip_latency_us: float = 0.0

    # Execution model parameters
    pipeline_depth: int = 1
    ridge_point_flops_per_byte: float = 1.0
    supports_weight_streaming: bool = False
    supports_operator_fusion: bool = False

    # BSP-specific (Graphcore)
    exchange_fabric_bandwidth_gbps: float = 0.0
    threads_per_tile: int = 1
    num_tiles: int = 1
    onchip_sram_per_tile_kb: float = 0.0

    # Dataflow-specific (SambaNova)
    num_pcus: int = 0
    num_pmus: int = 0
    pcu_simd_lanes: int = 1
    functional_units_per_pcu: int = 1

    def peak_flops(self, dtype: str = "fp16") -> float:
        """Return peak FLOPS in TFLOPS for given data type."""
        mapping = {
            "fp16": self.peak_flops_fp16,
            "bf16": self.peak_flops_bf16,
            "fp32": self.peak_flops_fp32,
        }
        return mapping.get(dtype.lower(), self.peak_flops_fp16)

    def peak_flops_ops(self, dtype: str = "fp16") -> float:
        """Return peak FLOPS as raw operations/second."""
        return self.peak_flops(dtype) * 1e12

    def onchip_bandwidth_bytes(self) -> float:
        """Return on-chip bandwidth in bytes/sec."""
        return self.onchip_bandwidth_gbps * 1e9

    def effective_offchip_bandwidth(self) -> float:
        """
        Return effective off-chip bandwidth in bytes/sec.

        Priority order:
          1. HBM  — if present (SambaNova SN40L)
          2. DDR  — if present (SambaNova SN30, Graphcore)
          3. Generic offchip — fallback (Cerebras MemoryX)
        """
        if self.hbm_bandwidth_gbps > 0:
            return self.hbm_bandwidth_gbps * 1e9
        if self.ddr_bandwidth_gbps > 0:
            return self.ddr_bandwidth_gbps * 1e9
        return self.offchip_bandwidth_gbps * 1e9

    def memory_tiers(self) -> list:
        """
        Return ordered list of (name, capacity_gb, bandwidth_gbps).
        First entry is always on-chip (fastest).
        """
        tiers = [("onchip_sram",
                  self.total_onchip_sram_gb,
                  self.onchip_bandwidth_gbps)]
        if self.hbm_gb > 0:
            tiers.append(("hbm", self.hbm_gb, self.hbm_bandwidth_gbps))
        if self.ddr_gb > 0:
            tiers.append(("ddr", self.ddr_gb, self.ddr_bandwidth_gbps))
        elif self.offchip_memory_gb > 0:
            bw = (self.offchip_bandwidth_gbps
                  if self.offchip_bandwidth_gbps > 0
                  else self.effective_offchip_bandwidth() / 1e9)
            tiers.append(("offchip_dram", self.offchip_memory_gb, bw))
        return tiers

    def __str__(self) -> str:
        return (
            f"HardwareSpec({self.name})\n"
            f"  Execution model : {self.execution_model}\n"
            f"  Peak FLOPS FP16 : {self.peak_flops_fp16} TFLOPS\n"
            f"  On-chip SRAM    : {self.total_onchip_sram_gb:.3f} GB\n"
            f"  On-chip BW      : {self.onchip_bandwidth_gbps:.1f} GB/s\n"
            f"  Off-chip mem    : {self.offchip_memory_gb:.1f} GB\n"
            f"  DDR BW          : {self.ddr_bandwidth_gbps:.1f} GB/s\n"
            f"  Inter-chip BW   : {self.inter_chip_bandwidth_gbps:.1f} GB/s\n"
            f"  Ridge point     : {self.ridge_point_flops_per_byte:.2f} FLOP/byte\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-file loader — reads from configs/hardware/{platform}.yaml
# ─────────────────────────────────────────────────────────────────────────────

def _hardware_dir() -> str:
    """Return absolute path to configs/hardware/ directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "hardware"
    )


def list_available_hardware() -> list:
    """
    Scan configs/hardware/ and return list of hardware_ids.
    Auto-discovers any new .yaml files — no code changes needed.
    """
    d = _hardware_dir()
    if not os.path.isdir(d):
        return []
    platforms = []
    for fname in sorted(os.listdir(d)):
        if fname.endswith(".yaml"):
            fpath = os.path.join(d, fname)
            try:
                with open(fpath) as f:
                    cfg = yaml.safe_load(f)
                hw_id = cfg.get("hardware_id", fname.replace(".yaml", ""))
                platforms.append(hw_id)
            except Exception:
                pass
    return platforms


def load_hardware_spec(platform: str) -> HardwareSpec:
    """
    Load hardware spec from configs/hardware/{platform}.yaml

    Each platform has its own dedicated YAML file.
    To add a new platform: create configs/hardware/new_platform.yaml
    No Python code changes needed.

    Args:
        platform: hardware_id matching the YAML filename
                  e.g. 'cerebras_wse2', 'sambanova_sn30', 'graphcore_bow_ipu'

    Returns:
        HardwareSpec instance
    """
    d = _hardware_dir()
    fpath = os.path.join(d, f"{platform}.yaml")

    if not os.path.isfile(fpath):
        available = list_available_hardware()
        raise FileNotFoundError(
            f"Hardware spec '{platform}' not found.\n"
            f"Expected file : {fpath}\n"
            f"Available     : {available}\n"
            f"To add a new platform: create configs/hardware/{platform}.yaml"
        )

    with open(fpath, "r") as f:
        s = yaml.safe_load(f)

    return HardwareSpec(
        name=s.get("name", platform),
        execution_model=s.get("execution_model", "unknown"),
        peak_flops_fp16=s.get("peak_flops_fp16", 0.0),
        peak_flops_bf16=s.get("peak_flops_bf16", 0.0),
        peak_flops_fp32=s.get("peak_flops_fp32", 0.0),
        total_onchip_sram_gb=s.get("total_onchip_sram_gb", 0.0),
        onchip_bandwidth_gbps=s.get("onchip_bandwidth_gbps", 0.0),
        offchip_memory_gb=s.get("offchip_memory_gb", 0.0),
        offchip_bandwidth_gbps=s.get("offchip_bandwidth_gbps",
                                     s.get("memoryx_bandwidth_gbps", 0.0)),
        hbm_gb=s.get("hbm_gb", 0.0),
        hbm_bandwidth_gbps=s.get("hbm_bandwidth_gbps", 0.0),
        ddr_gb=s.get("ddr_gb", 0.0),
        ddr_bandwidth_gbps=s.get("ddr_bandwidth_gbps", 0.0),
        inter_chip_bandwidth_gbps=s.get("inter_chip_bandwidth_gbps", 0.0),
        inter_chip_latency_us=s.get("inter_chip_latency_us", 0.0),
        pipeline_depth=s.get("pipeline_depth", 1),
        ridge_point_flops_per_byte=s.get("ridge_point_flops_per_byte", 1.0),
        supports_weight_streaming=s.get("supports_weight_streaming", False),
        supports_operator_fusion=s.get("supports_operator_fusion", False),
        exchange_fabric_bandwidth_gbps=s.get("exchange_fabric_bandwidth_gbps", 0.0),
        threads_per_tile=s.get("threads_per_tile", 1),
        num_tiles=s.get("num_tiles", 1),
        onchip_sram_per_tile_kb=s.get("onchip_sram_per_tile_kb", 0.0),
        num_pcus=s.get("num_pcus", 0),
        num_pmus=s.get("num_pmus", 0),
        pcu_simd_lanes=s.get("pcu_simd_lanes", 1),
        functional_units_per_pcu=s.get("functional_units_per_pcu", 1),
    )


# Auto-discovered from configs/hardware/ — no hardcoded list
SUPPORTED_PLATFORMS = list_available_hardware() or [
    "cerebras_wse2", "sambanova_sn30", "graphcore_bow_ipu"
]