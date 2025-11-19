import queue
from numpy import double
import torch 
import torch.fx as fx
import threading
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")   # No window
import matplotlib.pyplot as plt
import numpy as np
import os
"""
    Use a dictionary to store the survival time of this tensor for each round of training or inference.
    such as:
            dict_micro1{
                '%primal':(0,15),
                '%relu':(1,2),
            }

"""
@dataclass
class LifeTime:
    start_time: double = 0.0
    end_time: double = 0.0
    wait : bool = False
    shape : tuple = ()
    dtype : str = ""


@dataclass
class LifeTime_mp:
    start_time: double = 0.0
    end_time: double = 0.0
    wait : bool = False
    ready : bool = False



class Memory:
    
    def __init__(self,mem_dict) -> None:
        self.mem_dict = mem_dict


    def memory_manage() -> None:
        """
            Memory Management, Modeling Memory Changes.
            
        """
        
        pass

    def print_memory_tensor_report(self, device_id: int, save_dir="MemReport"):
        """
            Memory Usage + Tensor Count (two subplots, CCF-A style).
            Peak marked with a small red dot + vertical line,
            Peak time shown above the x-axis at the bottom of the red line (NOT in annotation text).

        """

        import torch.fx as fx
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        os.makedirs(save_dir, exist_ok=True)
        lifetimes = self.mem_dict

        dtype_size = {
            "float32": 4, "float": 4, "fp32": 4,
            "float16": 2, "half": 2, "fp16": 2,
            "bfloat16": 2, "bf16": 2,
            "int32": 4, "int64": 8,
        }

     
        clean_items = []
        for key, info in lifetimes.items():
            if isinstance(key, fx.Node):
                continue
            if not hasattr(info, "start_time"):
                continue
            clean_items.append((key, info))

        if not clean_items:
            print("No LifeTime entries.")
            return

      
        times = sorted({info.start_time for _, info in clean_items} |
                    {info.end_time for _, info in clean_items})

      
        mem_MB = []
        tensor_count = []

        for t in times:
            total_b = 0
            active = 0
            for name, info in clean_items:
                if info.start_time <= t <= info.end_time:
                    active += 1
                    numel = np.prod(info.shape) if isinstance(info.shape, tuple) else 0
                    b = dtype_size.get(str(info.dtype).lower(), 4)
                    total_b += numel * b

            mem_MB.append(total_b / (1024**2))
            tensor_count.append(active)

        mem_MB = np.array(mem_MB)
        tensor_count = np.array(tensor_count)

     
        peak_idx = np.argmax(mem_MB)
        peak_time = times[peak_idx]
        peak_value = mem_MB[peak_idx]

    
        def nice_ticks(lo, hi, n=6):
            raw = np.linspace(lo, hi, n)
            return np.round(raw).astype(int)

        xticks = list(nice_ticks(times[0], times[-1], 6))

     
        BLUE = "#0B3C5D"
        GOLD = "#AC8E00"
        RED = "#B30000"
        RED_LIGHT = "#E8B4B4"

     
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 10

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(7.4, 5.0),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 2]}
        )


        ax1.plot(times, mem_MB, color=BLUE, linewidth=1.8)


        ax1.scatter([peak_time], [peak_value], color=RED, s=15, zorder=4)


        ax1.axvline(
            peak_time, linestyle="--", color=RED_LIGHT, alpha=0.7, linewidth=1.0
        )


        ax1.annotate(
            f"Peak: {peak_value:.2f} MB",
            xy=(peak_time, peak_value),
            xytext=(peak_time + (xticks[-1] - xticks[0]) * 0.03,
                    peak_value * 1.03),
            arrowprops=dict(arrowstyle="->", color=RED),
            fontsize=9,
            color="black"
        )

        ax1.set_ylabel("Memory Usage (MB)", fontsize=10)
        ax1.text(-0.07, 1.02, "(a)", transform=ax1.transAxes,
                fontsize=12, fontweight="bold")

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)


        ax2.plot(times, tensor_count, color=GOLD, linewidth=1.6)

        ax2.axvline(
            peak_time, linestyle="--", color=RED_LIGHT, alpha=0.7, linewidth=1.0
        )

        ax2.set_ylabel("Tensor Count", fontsize=10)
        ax2.set_xlabel("Time (ms)", fontsize=10)
        ax2.text(-0.07, 1.02, "(b)", transform=ax2.transAxes,
                fontsize=12, fontweight="bold")

        ax2.set_xticks(xticks)

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)


        ax2.text(
            peak_time,
            0.10,
            f"{peak_time:.2f} ms",
            transform=ax2.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9
        )

        plt.tight_layout()

        out_pdf = os.path.join(save_dir, f"device_{device_id}_memory_tensor.pdf")
        fig.savefig(out_pdf, dpi=330, bbox_inches="tight")
        plt.close(fig)

        print(f"[Memory+Tensor] Peak={peak_value:.2f} MB")
        print(f"[Memory+Tensor] Saved: {out_pdf}")



    def print_memory_only_report(self, device_id: int, save_dir="MemReport"):
        import torch.fx as fx
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        os.makedirs(save_dir, exist_ok=True)
        lifetimes = self.mem_dict

        dtype_size = {
            "float32": 4, "float": 4, "fp32": 4,
            "float16": 2, "half": 2, "fp16": 2,
            "bfloat16": 2, "bf16": 2,
            "int32": 4, "int64": 8,
        }

        clean_items = []
        for key, info in lifetimes.items():
            if isinstance(key, fx.Node):
                continue
            if not hasattr(info, "start_time"):
                continue
            clean_items.append((key, info))

        times = sorted({i.start_time for _, i in clean_items} |
                    {i.end_time for _, i in clean_items})

        mem_MB = []
        for t in times:
            total_bytes = 0
            for name, info in clean_items:
                if info.start_time <= t <= info.end_time:
                    numel = np.prod(info.shape) if isinstance(info.shape, tuple) else 0
                    total_bytes += numel * dtype_size.get(str(info.dtype).lower(), 4)
            mem_MB.append(total_bytes / (1024**2))

        mem_MB = np.array(mem_MB)

        peak_idx = np.argmax(mem_MB)
        peak_time = times[peak_idx]
        peak_value = mem_MB[peak_idx]

        xticks = np.round(np.linspace(times[0], times[-1], 7)).astype(int)

        BLUE = "#0B3C5D"
        RED = "#B30000"
        RED_LIGHT = "#E8B4B4"

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 10

        fig, ax = plt.subplots(figsize=(7.0, 2.7))

        ax.plot(times, mem_MB, linewidth=1.8, color=BLUE)

        ax.scatter([peak_time], [peak_value], color=RED, s=15, zorder=4)

        ax.axvline(
            peak_time, linestyle="--", color=RED_LIGHT, alpha=0.7, linewidth=1.0
        )

        ax.annotate(
            f"Peak: {peak_value:.2f} MB",
            xy=(peak_time, peak_value),
            xytext=(peak_time + (xticks[-1]-xticks[0])*0.03,
                    peak_value*1.03),
            arrowprops=dict(arrowstyle="->", color=RED),
            fontsize=9
        )

   
        ax.text(
            peak_time,
            0.04,
            f"{peak_time:.2f} ms",
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=9
        )

        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Memory Usage (MB)", fontsize=10)
        ax.set_xticks(xticks)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        out_pdf = os.path.join(save_dir, f"device_{device_id}_memory_only.pdf")
        fig.savefig(out_pdf, dpi=330, bbox_inches="tight")
        plt.close(fig)

        print(f"[Memory Only] Peak={peak_value:.2f} MB")
        print(f"[Memory Only] PDF saved: {out_pdf}")