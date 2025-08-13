
import hashlib
import io
import os
import threading
from typing import Dict, List, Optional, Set, Tuple, Union

import valohai
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator

PathLike = Union[str, os.PathLike]

def _is_event_file(path: str) -> bool:
    return os.path.basename(path).startswith("events.out.tfevents")

def _iter_event_files(src: PathLike) -> List[str]:
    src = str(src)
    if os.path.isfile(src):
        return [src] if _is_event_file(src) else []
    found: List[str] = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.startswith("events.out.tfevents"):
                found.append(os.path.join(root, f))
    return sorted(found)

def _safe_tag(tag: str) -> str:
    return tag.replace("/", "_").replace("\\", "_")

def _image_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def list_tb_tags(src: PathLike) -> Dict[str, List[str]]:
    cats = {"scalars": set(), "images": set(), "histograms": set(), "tensors": set()}
    for ev_path in _iter_event_files(src):
        try:
            ea = EventAccumulator(ev_path, size_guidance={"scalars": 0, "images": 0, "histograms": 0, "tensors": 0})
            ea.Reload()
            tags = ea.Tags()
            for k in cats:
                for t in tags.get(k, []):
                    cats[k].add(t)
        except Exception:
            continue
    return {k: sorted(v) for k, v in cats.items()}

def extract_images_once(
    src: PathLike,
    outdir: PathLike,
    tag_substring: Optional[str] = None,
    max_per_tag: Optional[int] = None,
    since_step: Optional[int] = None,
    live_upload: bool = True,
    output_name: str = "images",
) -> int:
    """
    One-shot extraction of image summaries to PNG files.
    Filters by `tag_substring`, limits per tag with `max_per_tag`, and min step with `since_step`.
    If live_upload=True, marks each PNG read-only so Valohai uploads it mid-run.
    """
    os.makedirs(outdir, exist_ok=True)
    written = 0

    for ev_path in _iter_event_files(src):
        try:
            ea = EventAccumulator(ev_path, size_guidance={"images": 0})
            ea.Reload()
            tags = ea.Tags().get("images", [])
            if tag_substring:
                tags = [t for t in tags if tag_substring in t]

            for tag in tags:
                imgs = ea.Images(tag)
                if since_step is not None:
                    imgs = [ev for ev in imgs if ev.step >= since_step]

                tag_dir = os.path.join(str(outdir), _safe_tag(tag))
                os.makedirs(tag_dir, exist_ok=True)

                count = 0
                for idx, img_ev in enumerate(imgs):
                    if max_per_tag is not None and count >= max_per_tag:
                        break
                    try:
                        im = Image.open(io.BytesIO(img_ev.encoded_image_string))
                        fname = f"{_safe_tag(tag)}_step-{img_ev.step:06d}_{idx:03d}.png"
                        out_path = os.path.join(tag_dir, fname)
                        im.save(out_path, format="PNG")
                        written += 1
                        count += 1

                        if live_upload:
                            rel_to_root = os.path.relpath(out_path, valohai.outputs().dir_path)  # => "images//file.png"
                            valohai.outputs().live_upload(rel_to_root)
                    except Exception:
                        continue
        except Exception:
            continue

    return written

class _MirrorThread(threading.Thread):
    def __init__(
        self,
        logdir: PathLike,
        outdir: PathLike,
        tag_substring: Optional[str],
        poll_secs: float,
        since_step: Optional[int],
        live_upload: bool,
        output_name: str,
    ):
        super().__init__(name="tb_image_live_mirror", daemon=True)
        self.logdir = str(logdir)
        self.outdir = str(outdir)
        self.tag_substring = tag_substring
        self.poll_secs = poll_secs
        self.since_step = since_step
        self.live_upload = live_upload
        self.output_name = output_name
        self._stop_event = threading.Event()
        self._seen: Set[Tuple[str, int, str, str]] = set()

    def stop(self):
        self._stop_event.set()

    def run(self):
        os.makedirs(self.outdir, exist_ok=True)
        while not self._stop_event.is_set():
            for ev_path in _iter_event_files(self.logdir):
                try:
                    ea = EventAccumulator(ev_path, size_guidance={"images": 0})
                    ea.Reload()
                    tags = ea.Tags().get("images", [])
                    if self.tag_substring:
                        tags = [t for t in tags if self.tag_substring in t]

                    for tag in tags:
                        imgs = ea.Images(tag)
                        if self.since_step is not None:
                            imgs = [ev for ev in imgs if ev.step >= self.since_step]

                        tag_dir = os.path.join(self.outdir, _safe_tag(tag))
                        os.makedirs(tag_dir, exist_ok=True)

                        for idx, img_ev in enumerate(imgs):
                            try:
                                b = img_ev.encoded_image_string
                                key = (os.path.basename(ev_path), int(img_ev.step), tag, _image_hash(b))
                                if key in self._seen:
                                    continue
                                self._seen.add(key)

                                im = Image.open(io.BytesIO(b))
                                fname = f"{_safe_tag(tag)}_step-{img_ev.step:06d}_{idx:03d}.png"
                                out_path = os.path.join(tag_dir, fname)
                                im.save(out_path, format="PNG")
                                
                                bucket_base = valohai.outputs(self.output_name).path("")
                                rel_to_bucket = os.path.relpath(out_path, bucket_base)
                                valohai.outputs(self.output_name).live_upload(rel_to_bucket)

                                if self.live_upload:
                                    rel_path = os.path.relpath(out_path, valohai.outputs().dir_path)
                                    valohai.outputs(self.output_name).live_upload(rel_path)
                            except Exception:
                                continue
                except Exception:
                    continue
            # sleep but wake early if stop requested
            self._stop_event.wait(self.poll_secs)

def start_live_image_mirror(
    logdir: PathLike,
    outdir: PathLike,
    tag_substring: Optional[str] = None,
    poll_secs: float = 5.0,
    since_step: Optional[int] = None,
    live_upload: bool = True,
    output_name: str = "images",
) -> _MirrorThread:
    """
    Start a background thread that tails TB event files under `logdir`
    and writes filtered images as PNGs into `outdir`.
    """
    t = _MirrorThread(
        logdir=logdir,
        outdir=outdir,
        tag_substring=tag_substring,
        poll_secs=poll_secs,
        since_step=since_step,
        live_upload=live_upload,
        output_name=output_name,
    )
    t.start()
    return t

def stop_mirror(thread: _MirrorThread) -> None:
    """Stop a thread started by start_live_image_mirror()."""
    if thread and thread.is_alive():
        thread.stop()
        thread.join(timeout=10)
