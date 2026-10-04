from typing import TYPE_CHECKING, Optional, Tuple

from transformers import AutoTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
from .adapter import init_adapter
from .patcher import patch_config, patch_model, patch_tokenizer
from .utils import register_autoclass, is_vision_model
from ..extras.logging import get_logger
from ..extras.misc import count_parameters, get_current_device, try_download_model_from_ms

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from ..hparams import FinetuningArguments, ModelArguments

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoImageProcessor

try:
    from auto_gptq import AutoGPTQForCausalLM
except ImportError:
    AutoGPTQForCausalLM = None


logger = get_logger(__name__)


def load_model_and_tokenizer(
        model_args: "ModelArguments",
        finetuning_args: "FinetuningArguments",
        is_trainable: Optional[bool] = False,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    try_download_model_from_ms(model_args)

    # config_kwargs = {
    #     "trust_remote_code": True,
    #     "cache_dir": model_args.cache_dir,
    #     "revision": model_args.model_revision,
    #     "token": model_args.hf_hub_token,
    #     "attn_implementation": "flash_attention_2",  # 🔍
    # }
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
        "attn_implementation": "eager",  # 🔍
    }

    is_vision = is_vision_model(model_args.model_name_or_path)
    if is_vision: # For vision models, use image processor instead of tokenizer
        tokenizer = AutoImageProcessor.from_pretrained(
            model_args.model_name_or_path,
            **config_kwargs,
        )
    else: # For language models, use tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **config_kwargs,
        )
        patch_tokenizer(tokenizer)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    config.use_cache=False
    is_vision = is_vision_model(model_args.model_name_or_path)
    if not is_vision:
        print(config)
    else:
        print(f"{config.__class__.__name__} {{"
              f'"model_type": "{getattr(config, "model_type", "unknown")}", '
              f'"num_hidden_layers": {getattr(config, "num_hidden_layers", "N/A")}, '
              f'"hidden_size": {getattr(config, "hidden_size", "N/A")}, '
              f'"image_size": {getattr(config, "image_size", "N/A")}'
              f"}}")
    patch_config(config, tokenizer, model_args, config_kwargs, is_trainable)

    model = None
    if is_trainable and model_args.use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore

        unsloth_kwargs = {
            "model_name": model_args.model_name_or_path,
            "max_seq_length": model_args.model_max_length,
            "dtype": model_args.compute_dtype,
            "load_in_4bit": model_args.quantization_bit == 4,
            "token": model_args.hf_hub_token,
            "device_map": {"": get_current_device()},
            "rope_scaling": getattr(config, "rope_scaling", None),
        }
        try:
            model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
        except NotImplementedError:
            logger.warning("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
            model_args.use_unsloth = False

        if model_args.adapter_name_or_path:
            model_args.adapter_name_or_path = None
            logger.warning("Unsloth does not support loading adapters.")

    if model is None:
        if not model_args.autogptq:
            if is_vision:
                # DINOv3ViT has no ForImageClassification class — use AutoModel instead
                model_type = getattr(config, "model_type", None)
                auto_cls = AutoModel if model_type == "dinov3_vit" else AutoModelForImageClassification
                model = auto_cls.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    torch_dtype=model_args.compute_dtype,
                    low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
                    **config_kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    torch_dtype=model_args.compute_dtype,
                    low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
                    **config_kwargs,
                )
        else:
            if AutoGPTQForCausalLM is None:
                raise ImportError("auto-gptq is required when `model_args.autogptq` is enabled.")
            model = AutoGPTQForCausalLM.from_quantized(
                model_args.model_name_or_path,
                trust_remote_code=False,
                # model_basename=None if autogptq is True else Path(autogptq).stem,
                use_safetensors=True
                if model_args.autogptq is True
                else model_args.autogptq.endswith(".safetensors"),
                # **model_kwargs,
            )

    patch_model(model, tokenizer, model_args, is_trainable)
    register_autoclass(config, model, tokenizer)

    model = init_adapter(model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        model = model.to(model_args.compute_dtype) if not getattr(model, "quantization_method", None) else model
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    logger.info(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )
    for name, module in model.named_modules():
        if hasattr(module, "sparseThreshold"):
            module.sparseThreshold.requires_grad = True

    return model, tokenizer



