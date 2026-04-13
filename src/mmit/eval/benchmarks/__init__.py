from mmit.eval.benchmarks.base import Benchmark
from mmit.eval.benchmarks.textvqa import TextVQABenchmark
from mmit.eval.benchmarks.vqav2 import VQAv2Benchmark
from mmit.eval.benchmarks.pope import POPEBenchmark
from mmit.eval.benchmarks.mmbench import MMBenchBenchmark
from mmit.eval.benchmarks.gqa import GQABenchmark
from mmit.eval.benchmarks.scienceqa import ScienceQABenchmark
from mmit.eval.benchmarks.mme import MMEBenchmark
from mmit.eval.benchmarks.seed import SEEDBenchmark
from mmit.eval.benchmarks.vizwiz import VizWizBenchmark

__all__ = [
    "Benchmark",
    "TextVQABenchmark",
    "VQAv2Benchmark",
    "POPEBenchmark",
    "MMBenchBenchmark",
    "GQABenchmark",
    "ScienceQABenchmark",
    "MMEBenchmark",
    "SEEDBenchmark",
    "VizWizBenchmark",
]
