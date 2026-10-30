#!/usr/bin/env bash
# Download the two pinned patientx INT8 source files (AGPL-3.0) and their LICENSE
# into the bridge custom node folder, then verify SHA-256. Refuses to overwrite.
# Usage: bash fetch_patientx_int8.sh <ComfyUI>/custom_nodes/RDNA2-Qwen21-INT8
set -euo pipefail
dest="${1:?usage: fetch_patientx_int8.sh <custom_node_dir>}"
commit="a9560174bfe78cf16eb8fd21705c869d49eddf9d"
base="https://raw.githubusercontent.com/patientx/ComfyUI-INT8-Fast-ROCM/${commit}"
mkdir -p -- "$dest"

fetch() {
  local remote="$1" local_name="$2" sha="$3"
  local out="$dest/$local_name"
  if [[ -e "$out" ]]; then
    echo "exists, not overwriting: $out" >&2
  else
    curl --fail --location --silent --show-error --output "$out.part" "$base/$remote"
    mv -- "$out.part" "$out"
  fi
  printf '%s  %s\n' "$sha" "$out" | sha256sum --check
}

fetch rocm_int8_kitchen_patch.py rocm_int8_kitchen_patch.py e30de1392612bbce0c5b370222d7eae42403babcf4f3f932de95fc982278ed3f
fetch int8_fused_kernel.py       int8_fused_kernel.py       70c014d3c394b69cc907c60dd7086e367861c6bf03031320524c3ba2dcf86603
fetch LICENSE                    LICENSE-AGPL-3.0           7d76c2ceb53e4d9c68d9cfc9271c9c5770c50606986c2b6e71bc9f3473377cc3
