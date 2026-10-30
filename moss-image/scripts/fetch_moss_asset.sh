#!/usr/bin/env bash
# Download exactly one pinned public MOSS-Image file and verify size + SHA-256.
# Refuses to overwrite a non-matching existing file. Resumes a partial .part file.
#
#   MODELS=<models> bash fetch_moss_asset.sh <id>
#
# ids: base_int8 sft_int8 te_int8 vae_small sft4_lora edit_hq_armv5_native
#      anime_comfy charref_v2 pixelart16 dmd8_speed_lora
set -euo pipefail
models="${MODELS:?set MODELS to your ComfyUI models directory}"
id="${1:-}"

case "$id" in
  sft_int8)
    repo='Comfy-Org/MOSS-Image'; rev='530a4c94f92c35ef6818b1b035a983fe9dcf5f54'
    remote='diffusion_models/moss_image_sft_int8_convrot.safetensors'; size=5413698958
    sha='b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f' ;;
  base_int8)
    repo='Comfy-Org/MOSS-Image'; rev='530a4c94f92c35ef6818b1b035a983fe9dcf5f54'
    remote='diffusion_models/moss_image_base_int8_convrot.safetensors'; size=5413698958
    sha='f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f' ;;
  te_int8)
    repo='Comfy-Org/MOSS-Image'; rev='530a4c94f92c35ef6818b1b035a983fe9dcf5f54'
    remote='text_encoders/qwen3.5_9b_int8_convrot.safetensors'; size=9748121532
    sha='32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a' ;;
  vae_small)
    repo='Comfy-Org/MOSS-Image'; rev='530a4c94f92c35ef6818b1b035a983fe9dcf5f54'
    remote='vae/flux2-vae_small.safetensors'; size=249521340
    sha='d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b' ;;
  sft4_lora)
    repo='akhaliq/MOSS-Image-SFT-4-steps'; rev='2ded95c37a029360245926fe72e475e30039bbfb'
    remote='moss_image_sft_4step_v2.0.safetensors'; size=1189176752
    sha='154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f' ;;
  edit_hq_armv5_native)
    repo='akhaliq/MOSS-Image-Edit-HQ'; rev='3b1f2a27ddf2cbe38074531be80ba137b165ee01'
    remote='weights/armV5/armV5.safetensors'; size=495514936
    sha='1fbfae4eaaafc62c66a298e63e16fc5211566e1e3de1c58f83c35c6bc226c9f9' ;;
  anime_comfy)
    repo='axiomofmind/Moss-image-anime-lora'; rev='38fa56fa567b70412828ac67c050a4c2c1a5c40a'
    remote='MOSS-Image-Anime-LoRA-ComfyUI.safetensors'; size=74379584
    sha='6661f88183757c7be9044578d3907335d6abe83a736995bd502b78db2c5b83cf' ;;
  charref_v2)
    repo='7err4/MOSS-Image-CharRef'; rev='70db7d88c478bff21c86a3709673d17655f6e162'
    remote='moss_image_charref_v2.safetensors'; size=148697656
    sha='1a77e98228242d7ca79dd01df737d751f211f2ae676f16cce664987dbf0bac05' ;;
  pixelart16)
    repo='Angais/MOSS-Image-PixelArt-16x16'; rev='c3062e0999f33c59d985457dd290ffb00c0e28b7'
    remote='moss-image-pixelart-16x16.safetensors'; size=148689216
    sha='955900cd37741e51b475c26383b1af6bb6be27d656559d0e3f6d59ec790a69b7' ;;
  dmd8_speed_lora)
    repo='lodestones/MOSS-Image-SFT-8-steps-dmd'; rev='f95df83b5e90ae91ea412434b3f9b7f1c8c6314f'
    remote='lora.safetensors'; size=1189176800
    # The HF tree's lfs.oid is the file SHA-256. The CDN etag (xet hash) is NOT.
    sha='4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe' ;;
  *)
    sed -n '2,9p' "$0" >&2; exit 2 ;;
esac

case "$id" in
  dmd8_speed_lora) dst="$models/loras/moss_sft_8step_dmd_speed.safetensors" ;;
  sft4_lora|edit_hq_armv5_native|anime_comfy|charref_v2|pixelart16) dst="$models/loras/${remote##*/}" ;;
  *) dst="$models/$remote" ;;
esac
tmp="$dst.part"
mkdir -p -- "$(dirname -- "$dst")"

verify() {
  [[ -f "$1" ]] || return 1
  [[ $(stat -c %s "$1") == "$size" ]] || return 1
  printf '%s  %s\n' "$sha" "$1" | sha256sum --check --status
}

if verify "$dst"; then echo "VERIFIED_EXISTING $id $dst"; exit 0; fi
if [[ -e "$dst" ]]; then echo "Refusing to overwrite a non-matching file: $dst" >&2; exit 3; fi
if verify "$tmp"; then mv -- "$tmp" "$dst"; echo "FETCH_SHA256_OK $id $dst (existing complete .part)"; exit 0; fi

url="https://huggingface.co/$repo/resolve/$rev/$remote"
echo "FETCH_START $id expected=$size bytes -> $dst"
curl --fail --location --silent --show-error --retry 8 --retry-delay 3 \
  --retry-all-errors --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
  --continue-at - --output "$tmp" "$url"
if ! verify "$tmp"; then
  echo "Size or SHA-256 mismatch; kept $tmp for inspection" >&2
  exit 4
fi
mv -- "$tmp" "$dst"
echo "FETCH_SHA256_OK $id $dst"
