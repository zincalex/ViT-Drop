import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ResultsLog:
    def __init__(self, output_dir: str, dataset: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / f"selection_log_{dataset}.json"
        self.data = self._load_or_init()

    def _load_or_init(self) -> Dict[str, Any]:
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                return json.load(f)
        return {
            "hyperparameters": None,
            "baselines": {},
            "search_results": {},
            "winner": None,
            "deep_finetune": None,
            "final_test": None,
            "started_at": datetime.now().isoformat(),
        }

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)


    # --- Hyperparameters ---

    def has_hyperparameters(self) -> bool:
        return self.data.get("hyperparameters") is not None

    def record_hyperparameters(self, hparams: Dict[str, Any]):
        """Record, once per run, the settings used for each phase.

        Stores the common settings plus a per-phase block (baseline, search,
        deep_finetune) describing epochs, optimizer/scheduler, and the dataset
        splits used for training and evaluation.
        """
        self.data["hyperparameters"] = hparams
        self._save()

    def _key(self, arch: str, method: str) -> str:
        return f"{arch}/{method}"

    def _variant_key(self, arch: str, method: str, drop_n: int) -> str:
        return f"{arch}/{method}/drop{drop_n}"

    # --- Baseline ---

    def has_baseline(self, arch: str, method: str) -> bool:
        return self._key(arch, method) in self.data["baselines"]

    def get_baseline_accuracy(self, arch: str, method: str) -> Optional[float]:
        entry = self.data["baselines"].get(self._key(arch, method))
        return entry["accuracy"] if entry else None

    def record_baseline(self, arch: str, method: str, metrics: Dict[str, float]):
        self.data["baselines"][self._key(arch, method)] = {
            **metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def get_any_baseline_for_arch(self, arch: str) -> Optional[Dict]:
        """Return the first recorded baseline entry for this architecture, if any.

        drop0 is the same unpruned model for every pruning method, so any method's
        baseline can be reused to populate the others.
        """
        prefix = f"{arch}/"
        for key, entry in self.data["baselines"].items():
            if key.startswith(prefix):
                return entry
        return None

    # --- Search results ---

    def has_search_result(self, arch: str, method: str, drop_n: int) -> bool:
        return self._variant_key(arch, method, drop_n) in self.data["search_results"]

    def get_search_accuracy(self, arch: str, method: str, drop_n: int) -> Optional[float]:
        entry = self.data["search_results"].get(self._variant_key(arch, method, drop_n))
        return entry["accuracy"] if entry else None

    def record_search_result(self, arch: str, method: str, drop_n: int, metrics: Dict[str, float], early_stopped: bool = False):
        self.data["search_results"][self._variant_key(arch, method, drop_n)] = {
            **metrics,
            "early_stopped": early_stopped,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    # --- Winner ---

    def record_winner(self, arch: str, method: str, drop_n: int, val_accuracy: float):
        self.data["winner"] = {
            "arch": arch,
            "method": method,
            "drop_n": drop_n,
            "val_accuracy": val_accuracy,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def get_winner(self) -> Optional[Dict]:
        return self.data["winner"]

    # --- Deep fine-tune & final test ---

    def record_deep_finetune(self, metrics: Dict[str, float]):
        self.data["deep_finetune"] = {
            **metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def record_final_test(self, metrics: Dict[str, float]):
        self.data["final_test"] = {
            **metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self.data["completed_at"] = datetime.now().isoformat()
        self._save()

    def has_final_test(self) -> bool:
        return self.data["final_test"] is not None

    def get_best_variant(self) -> Optional[Dict]:
        best = None
        best_acc = -1.0
        best_drop = -1

        for key, entry in self.data["search_results"].items():
            if entry.get("early_stopped"):
                continue
            parts = key.split("/")
            arch, method = parts[0], parts[1]
            drop_n = int(parts[2].replace("drop", ""))
            acc = entry["accuracy"]
            # Tie-break: prefer higher drop count (more compression)
            if acc > best_acc or (acc == best_acc and drop_n > best_drop):
                best_acc = acc
                best_drop = drop_n
                best = {"arch": arch, "method": method, "drop_n": drop_n, "val_accuracy": acc}

        return best

    def print_summary(self):
        print("\n" + "=" * 80)
        print("MODEL SELECTION SUMMARY")
        print("=" * 80)
        winner = self.data["winner"]
        if winner:
            print(f"Winner: {winner['arch']} / {winner['method']} / drop={winner['drop_n']}")
            print(f"Val accuracy: {winner['val_accuracy']:.4f}")
        if self.data["final_test"]:
            t = self.data["final_test"]
            print(f"Test accuracy: {t['accuracy']:.4f} | F1: {t['f1']:.4f}")
        print("=" * 80)
