DATA_CONFIG = "dataset_info.json"

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

IGNORE_INDEX = -100

LAYERNORM_NAMES = {"norm", "ln"}

LOG_FILE_NAME = "../../../results_pt/debug/trainer_log.jsonl"

METHODS = ["full", "freeze", "lora"]

PEFT_METHODS = ["lora"]
