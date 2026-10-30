# RDNA2(gfx1030) 이미지·영상 생성 레시피

**AMD Radeon RX 6900 XT**(RDNA2, `gfx1030`, VRAM 16 GB), RAM 64 GB, Ubuntu 환경에서 최근 생성 모델 세 개를 ComfyUI로 실행한 구성을 정리한다. PyTorch는 AMD ROCm 10.2 nightly 휠을 사용했다.

gfx1030에는 최신 GPU에 있는 일부 커널(hipBLASLt INT8, Flash attention)이 없다. 각 레시피에는 실행에 필요했던 파일·패치·실행 옵션, 측정 조건을 붙인 시간, 검증하지 않은 항목을 함께 적었다.

| 레시피 | 작업 | 대표 결과(단일 실행) |
|---|---|---|
| [qwen-image/](qwen-image/README.md) | Qwen-Image-2.1 INT8 ConvRot, T2I와 단일 참조 이미지 편집 | 640×960 편집 25스텝 80.05초, 최대 12.41 GiB. Turbo LoRA 8스텝 48.03초 |
| [moss-image/](moss-image/README.md) | MOSS-Image SFT INT8, 고속 T2I와 Edit-HQ 편집 | 768×1152 T2I 4스텝 warm 24.02초, 14.55 GiB. 768×576 DMD 8스텝 편집 cold 56.04초 |
| [minimax-h3/](minimax-h3/README.md) | MiniMax H3 FL2VA GGUF, 124프레임 T2V·I2V | 608×352 T2V 4스텝 전체 228.32초. 512×768 I2V 4스텝 전체 462.56초 |

먼저 [hardware.md](hardware.md)를 읽는다. 장비 구성, ROCm·PyTorch 버전, 세 레시피가 공통으로 우회한 gfx1030 문제, 측정 방법이 정리되어 있다. 각 레시피 문서는 영어로 작성했다.

## 수치를 읽을 때 주의할 점

- 시간·VRAM 수치는 거의 모두 **1회 실행** 결과다. 같은 Qwen 그래프를 따로 세 번 실행했을 때 80.05초, 81.08초, 91.19초가 나왔다.
- VRAM은 sysfs 값을 0.25~2초 간격으로 읽은 최댓값이다. 할당자 캐시가 포함되고 짧은 순간 최댓값은 놓칠 수 있다.
- 표마다 "시간"의 정의가 다르다(ComfyUI prompt 실행 시간, API 대기열 등록부터 history 기록까지, 서버 기동을 포함한 전체 시간). 각 표에 정의를 적었다.
- 품질에 관한 서술은 실험실에서 한두 장의 이미지·영상을 직접 보고 남긴 기록이다. 픽셀·SSIM 지표는 설정에 따라 출력이 달라졌음을 보여 줄 뿐 품질 순위를 뜻하지 않는다.

## 저장소 구성

```
hardware.md                 장비·ROCm·ComfyUI 공통 사항
scripts/gpu_preflight.py    GPU 유휴 여부를 sysfs로만 확인하는 읽기 전용 검사
scripts/submit_workflow.py  localhost ComfyUI에 API 워크플로를 보내고 시간을 기록
qwen-image/                 문서, 실행 스크립트, INT8 브리지 노드, 패치, 워크플로
moss-image/                 문서, 실행 스크립트, INT8 GEMM 노드, 다운로드 스크립트, 패치, 워크플로
minimax-h3/                 문서, 실행 스크립트, 패치, torchaudio 대체 모듈, 워크플로
```

워크플로는 `POST /prompt`용 ComfyUI **API 형식** JSON이며 UI 워크플로가 아니다. 프롬프트와 입력 이미지 이름은 예시 값이다.

## 포함하지 않은 것

- 모델 가중치. 각 문서에 Hugging Face 저장소, 고정 리비전, 파일 크기, SHA-256을 적었다.
- patientx INT8 Triton 파일(AGPL-3.0). `qwen-image/scripts/fetch_patientx_int8.sh`가 고정 커밋에서 내려받는다.
- ComfyUI 본체와 커스텀 노드 저장소. 문서에 체크아웃할 커밋을 적었다.
- 같은 실험실에서 수행한 클라우드 NVIDIA L4 실험 등 RX 6900 XT가 아닌 환경의 결과.

## 라이선스

이 저장소의 코드와 문서는 MIT 라이선스([LICENSE](LICENSE))를 따른다. 다음은 예외다.

- `*/patches/*.patch`는 제3자 파일을 수정하므로 원래 파일의 라이선스를 따른다(ComfyUI: GPL-3.0, ComfyUI-GGUF: Apache-2.0, Ostris MOSS 편집 노드: MIT). 각 `patches/README.md`를 참고한다.
- 모델·LoRA·업스케일러는 각자의 라이선스를 따르며 각 레시피 문서에 링크를 달았다. 연구용으로만 허용된 것도 있으므로 사용 전에 확인한다.

## 안전 관련 사항

모든 서버는 `127.0.0.1`에만 바인딩한다. GPU 작업은 한 번에 하나만 실행하고, 끝나면 서버를 종료해 VRAM을 반환한다. 다운로드 스크립트를 직접 실행하지 않는 한 어떤 스크립트도 파일을 내려받지 않는다.
