from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="StarVLA/RoboTwin-Randomized",
    repo_type="dataset",
    local_dir="./RoboTwin-Randomized",
    allow_patterns=[
        "Randomized/stack_blocks_three/**",
        "Randomized/stack_blocks_two/**",
        "Randomized/stack_bowls_three/**",
        "Randomized/stack_bowls_two/**",
        "Randomized/stamp_seal/**",
        "Randomized/turn_switch/**",
    ],
)
