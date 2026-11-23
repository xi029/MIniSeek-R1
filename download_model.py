from huggingface_hub import snapshot_download


if __name__ == "__main__":
    snapshot_download("internlm/internlm2-1_8b-reward", local_dir="./reward/internlm2-1_8b-reward")
