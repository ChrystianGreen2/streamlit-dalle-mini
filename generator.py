
import random
import numpy as np

import jax
import wandb
import jax.numpy as jnp

from PIL import Image
from functools import partial

from dalle_mini import DalleBartProcessor
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

from flax.training.common_utils import shard_prng_key
from flax.jax_utils import replicate

wandb.login(key="your key")
jax.default_backend()

DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  
# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
DALLE_COMMIT_ID = None

VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

params = replicate(params)
vqgan_params = replicate(vqgan_params)

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)


def get_images(prompts, n_predictions, gen_top_k=None, gen_top_p=None, temperature=None):
    prompts = [prompts]
    tokenized_prompts = processor(prompts)

    tokenized_prompt = replicate(tokenized_prompts)

    cond_scale = 10.0

    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    print(f"Generating....\n")
    images = []
    for i in range(max(n_predictions // jax.device_count(), 1)):
        print(f"Image {i}")
        key, subkey = jax.random.split(key)
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        encoded_images = encoded_images.sequences[..., 1:]
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)
    return images