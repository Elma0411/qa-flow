# 文件作用：命令行训练知识三级标签模型。
# 关联说明：调用 modeling、taxonomy、train_utils，产出 predictor 可加载的模型。

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from qa.knowledge_tagging_3lvl.dataset_io import read_jsonl, write_json
from qa.knowledge_tagging_3lvl.modeling import HierarchicalTagger, ModelSpec, default_hf_endpoint, save_model
from qa.knowledge_tagging_3lvl.taxonomy import build_label_mappings, parse_taxonomy
from qa.knowledge_tagging_3lvl.train_utils import (
    JsonlClassificationDataset,
    accuracy,
    collate_fn,
    format_steps,
    set_seed,
    weighted_ce_loss,
)


def _load_label_maps(labels_file: str) -> dict:
    leaves = parse_taxonomy(labels_file)
    return build_label_mappings(leaves)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def _eval(model: HierarchicalTagger, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    a1 = a2 = a3 = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            y1 = batch.y1.to(device)
            y2 = batch.y2.to(device)
            y3 = batch.y3.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            a1 += accuracy(out["logits1"], y1) * len(y1)
            a2 += accuracy(out["logits2"], y2) * len(y2)
            a3 += accuracy(out["logits3"], y3) * len(y3)
            n += len(y1)
    if n == 0:
        return {"acc_level1": 0.0, "acc_level2": 0.0, "acc_leaf": 0.0}
    return {"acc_level1": a1 / n, "acc_level2": a2 / n, "acc_leaf": a3 / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Path to 三级知识标签.txt")
    ap.add_argument("--train", required=True, help="train.jsonl")
    ap.add_argument("--val", required=True, help="val.jsonl")
    ap.add_argument("--out", required=True, help="Output model dir")
    ap.add_argument("--model-name", default="hfl/rbt3")
    ap.add_argument("--resume-from", default="", help="Resume from a previous output dir (model_state.pt)")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    ap.add_argument("--amp", action="store_true", help="Enable torch.cuda.amp mixed precision (CUDA only)")
    ap.add_argument("--grad-accum-steps", type=int, default=1, help="Accumulate gradients to simulate larger batch")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss-w1", type=float, default=0.2)
    ap.add_argument("--loss-w2", type=float, default=0.3)
    ap.add_argument("--loss-w3", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    default_hf_endpoint()

    label_maps = _load_label_maps(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "labels.json", label_maps)
    write_json(out_dir / "train_args.json", vars(args))

    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.val)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    resume_dir = Path(args.resume_from) if args.resume_from else None
    base_model_name = args.model_name
    resume_spec = None
    if resume_dir:
        spec_path = resume_dir / "model_spec.json"
        state_path = resume_dir / "model_state.pt"
        if not spec_path.exists() or not state_path.exists():
            raise FileNotFoundError(f"Missing resume files in {resume_dir}")
        resume_spec = ModelSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
        base_model_name = resume_spec.base_model_name
        if resume_spec.num_level1 != len(label_maps["level1"]):
            raise ValueError(
                f"resume num_level1={resume_spec.num_level1} != current num_level1={len(label_maps['level1'])}"
            )
        if resume_spec.num_level2 != len(label_maps["level2_pairs"]):
            raise ValueError(
                f"resume num_level2={resume_spec.num_level2} != current num_level2={len(label_maps['level2_pairs'])}"
            )
        if resume_spec.num_leaf != len(label_maps["leaf_labels"]):
            raise ValueError(
                f"resume num_leaf={resume_spec.num_leaf} != current num_leaf={len(label_maps['leaf_labels'])}"
            )
        tokenizer = AutoTokenizer.from_pretrained(str(resume_dir), use_fast=True)
        base_config = None
        base_config_path = resume_dir / "base_config.json"
        if base_config_path.exists():
            base_config = AutoConfig.from_pretrained(str(base_config_path))
        model = HierarchicalTagger(
            base_model_name=base_model_name,
            num_level1=resume_spec.num_level1,
            num_level2=resume_spec.num_level2,
            num_leaf=resume_spec.num_leaf,
            base_model_config=base_config,
            local_files_only=True,
        )
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
        except TypeError:  # older torch
            state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = HierarchicalTagger(
            base_model_name=args.model_name,
            num_level1=len(label_maps["level1"]),
            num_level2=len(label_maps["level2_pairs"]),
            num_leaf=len(label_maps["leaf_labels"]),
        )

    train_ds = JsonlClassificationDataset(train_rows, tokenizer, label_maps, max_length=args.max_length)
    val_ds = JsonlClassificationDataset(val_rows, tokenizer, label_maps, max_length=args.max_length)

    pin_memory = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    optim_steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    total_steps = args.epochs * optim_steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_leaf = -1.0
    history: list[dict] = []
    use_amp = bool(args.amp and device.type == "cuda")
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]
        autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)  # type: ignore[attr-defined]
    else:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            y1 = batch.y1.to(device, non_blocking=True)
            y2 = batch.y2.to(device, non_blocking=True)
            y3 = batch.y3.to(device, non_blocking=True)
            weight = batch.weight.to(device, non_blocking=True)

            with autocast():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss1 = weighted_ce_loss(out["logits1"], y1, weight)
                loss2 = weighted_ce_loss(out["logits2"], y2, weight)
                loss3 = weighted_ce_loss(out["logits3"], y3, weight)
                loss = args.loss_w1 * loss1 + args.loss_w2 * loss2 + args.loss_w3 * loss3
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            do_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))
            if do_step:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()

            if step % 50 == 0 or step == len(train_loader):
                print(f"epoch {epoch} step {format_steps(step, len(train_loader))} loss={loss.item():.4f}")

        metrics = _eval(model, val_loader, device=device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        if metrics["acc_leaf"] > best_leaf:
            best_leaf = metrics["acc_leaf"]
            spec = ModelSpec(
                base_model_name=base_model_name,
                num_level1=len(label_maps["level1"]),
                num_level2=len(label_maps["level2_pairs"]),
                num_leaf=len(label_maps["leaf_labels"]),
                max_length=args.max_length,
            )
            save_model(out_dir, model, tokenizer, spec)

    write_json(out_dir / "train_history.json", history)
    print(f"best_val_leaf_acc={best_leaf:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
