def test_create_diffusion_imports():
    from starVLA.utils.diffusion_utils import create_diffusion
    assert callable(create_diffusion)


def test_create_diffusion_accepts_integer_respacing():
    from starVLA.utils.diffusion_utils import create_diffusion

    diffusion = create_diffusion(
        timestep_respacing=10,
        noise_schedule="cosine",
        learn_sigma=False,
    )
    assert diffusion.num_timesteps == 10
