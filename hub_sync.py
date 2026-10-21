"""Keep training checkpoints OFF the Kaggle box.

/kaggle/working is not durable: a stopped session, a crash or a timeout can
take checkpoints/ with it, even with "Persistence: Files only". This module
mirrors a run folder to a PRIVATE Hugging Face model repo while training runs,
and pulls it back at the start of the next session.

Never raises into the training loop: a failed push prints a warning and
training continues; the next push retries with newer files.

    Kaggle > Add-ons > Secrets: add HF_TOKEN (fine-grained, write access to this
    one repo only) and tick "Attached" for the notebook.

What is pushed:   adapter-last/, adapter-best/, train_state.pt, curve.csv,
                  result.json, data_report.json
What is NOT:      gate/ (contains the private safety probes), dotfiles such as
                  .ddp_data.pt (tokenised corpus cache), anything else.

The repo must be private. A public repo is refused unless
HANNY_HUB_ALLOW_PUBLIC=1, because an adapter trained on private text (m4) must
never land in a public repo by accident.
"""
import os
import shutil
import tempfile
import threading

PUSH_NAMES = ("adapter-last", "adapter-best", "train_state.pt", "curve.csv",
              "result.json", "data_report.json")


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


class HubSync:
    def __init__(self, repo, run):
        self.repo, self.run = repo, run
        self._thread = None
        self._checked = False
        self._ok = False

    @property
    def enabled(self):
        return bool(self.repo and _token())

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=_token())

    def _ensure_private_repo(self):
        if self._ok:
            return True
        api = self._api()
        try:
            api.create_repo(self.repo, private=True, exist_ok=True, repo_type="model")
        except Exception:
            # The repo normally exists already, and a token scoped to that one repo
            # may not be allowed to "create". repo_info below is the real check; if
            # it raises, the caller logs it and the next push retries.
            pass
        info = api.repo_info(self.repo, repo_type="model")
        if not getattr(info, "private", True) and \
                os.environ.get("HANNY_HUB_ALLOW_PUBLIC") != "1":
            print(f"[hub_sync] {self.repo} is PUBLIC; refusing to push. Make it "
                  "private or set HANNY_HUB_ALLOW_PUBLIC=1.")
            return False
        self._ok = True
        return True

    # push
    def push(self, run_dir, step, wait=False):
        """Snapshot run_dir locally (fast), then upload the snapshot in a
        background thread so the GPUs are not idle during the upload. Skips if
        the previous upload is still running."""
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            print(f"[hub_sync] previous upload still running; skipping step {step}")
            return
        # Outside /kaggle/working so the snapshot never lands in the notebook
        # output or in Cell 7's zip.
        stage = os.path.join(tempfile.gettempdir(), f"hub_stage_{self.run}")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        for name in PUSH_NAMES:
            src = os.path.join(run_dir, name)
            dst = os.path.join(stage, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                shutil.copy2(src, dst)

        def work():
            try:
                if not self._ensure_private_repo():
                    return
                self._api().upload_folder(
                    folder_path=stage, repo_id=self.repo, repo_type="model",
                    path_in_repo=self.run,
                    commit_message=f"{self.run} step {step}")
                print(f"[hub_sync] pushed {self.run} step {step} -> {self.repo}")
            except Exception as e:  # never kill training over a network blip
                print(f"[hub_sync] push FAILED at step {step}: {e!r}")

        self._thread = threading.Thread(target=work, daemon=False)
        self._thread.start()
        if wait:
            self.flush()

    def flush(self, timeout=900):
        """Call before the process exits so the last upload is not cut off."""
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                print("[hub_sync] last upload still running after "
                      f"{timeout}s; exiting anyway")

    # pull
    def pull(self, ckpt_root):
        """Download <run>/ from the repo into ckpt_root/<run>/. Returns True if
        a resumable state (train_state.pt) arrived."""
        if not self.enabled:
            print("[hub_sync] no HF_TOKEN / repo; nothing to pull")
            return False
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(repo_id=self.repo, repo_type="model",
                              local_dir=ckpt_root, token=_token(),
                              allow_patterns=[f"{self.run}/*"])
        except Exception as e:
            print(f"[hub_sync] pull failed or repo empty: {e!r}")
            return False
        return os.path.exists(os.path.join(ckpt_root, self.run, "train_state.pt"))