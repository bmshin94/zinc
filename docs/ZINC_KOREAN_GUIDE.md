# ZINC 한국어 가이드 — 분석 · 설치 · 활용 · 수익화

> 이 문서는 ZINC 저장소를 직접 분석하고 정리한 한국어 종합 가이드입니다.
> 개념 설명부터 설치, 사용법, 수익화 아이디어, React/PHP 연동 구조까지 한 곳에 담았습니다.

**저장소 주소**

- 이 저장소(포크): https://github.com/bmshin94/zinc
- 원본 저장소(upstream): https://github.com/zolotukhin/zinc
- 공식 문서: https://zolotukhin.ai/zinc/docs/
- 벤치마크: https://zolotukhin.ai/zinc/benchmarks/#rdna-rocm
- Discord: https://discord.gg/QRUgWH2aGV

라이선스: MIT (상업적 이용 가능). 단, 원작자 표기를 유지하는 것을 권장합니다.

---

## 목차

1. [ZINC이 무엇인가](#1-zinc이-무엇인가)
2. [저장소 구조 분석](#2-저장소-구조-분석)
3. [설치 가이드](#3-설치-가이드)
4. [사용법](#4-사용법)
5. [API 레퍼런스 요약](#5-api-레퍼런스-요약)
6. [알려진 한계](#6-알려진-한계)
7. [수익화 아이디어](#7-수익화-아이디어)
8. [React / PHP 연동 아키텍처](#8-react--php-연동-아키텍처)
9. [트러블슈팅](#9-트러블슈팅)

---

## 1. ZINC이 무엇인가

**한 줄 요약: 내 컴퓨터의 GPU로 LLM(대형 언어 모델)을 직접 돌리는 추론 엔진.**

ChatGPT처럼 외부 API를 호출하는 방식이 아니라, GGUF 형식의 모델 파일을 로컬 그래픽카드에 올려서 직접 추론합니다. `llama.cpp`, `Ollama`, `LM Studio`와 같은 카테고리의 도구입니다.

### 비유로 이해하기

| | 배달 음식 (ChatGPT API) | 집밥 (ZINC) |
|---|---|---|
| 재료 | 없음 | GGUF 모델 파일 |
| 불 | 남의 서버 | 내 그래픽카드 |
| 비용 | 토큰당 과금 | 전기값만 |
| 프라이버시 | 데이터가 외부로 나감 | 내 컴퓨터 밖으로 안 나감 |

### 핵심 특징

| 항목 | 내용 |
|---|---|
| 언어 | **Zig** (약 157,000줄) |
| 배포 형태 | 단일 바이너리 — CLI + 웹 채팅 UI + 모델 매니저 + OpenAI 호환 API 전부 내장 |
| GPU 백엔드 | AMD(Vulkan/ROCm), Intel Arc(Vulkan), Apple Silicon(Metal), NVIDIA(CUDA, 실험적) |
| 모델 형식 | GGUF (Q4_K, Q5_K, Q6_K, Q8_0, Q5_0, MXFP4, F16, F32) |
| 라이선스 | MIT |

### 다른 도구와의 차별점

대부분의 로컬 LLM 도구는 **NVIDIA CUDA**를 1군으로 두고 AMD/Intel을 후순위로 둡니다.
**ZINC은 반대로 AMD·Intel·Apple을 1군으로 두고 CUDA를 실험적 백엔드로 둡니다.**

저장소의 주장에 따르면 AMD Radeon AI PRO R9700(ROCm) 환경에서 6개 모델 전부 prefill·decode·종합 시간에서 llama.cpp(Vulkan) 대비 우위입니다.
단, 이는 **GPU 1종 + 모델 6종 한정의 스코프된 결과**이며, ZINC은 ROCm / llama.cpp는 Vulkan 백엔드로 측정한 비교입니다. 저장소 자체도 "모든 모델·GPU에 대한 주장이 아니다"라고 명시하고 있습니다.

---

## 2. 저장소 구조 분석

### 핵심 엔진

| 경로 | 역할 |
|---|---|
| `src/model/` | GGUF 파서, 토크나이저, 모델 카탈로그, Hugging Face 다운로더 |
| `src/compute/` | 어텐션, 행렬곱(dmmv), argmax 등 실제 연산 그래프 |
| `src/server/` | HTTP 서버, 라우팅, 모델 매니저, 내장 채팅 UI(`chat.html`) |
| `src/scheduler/` | KV 캐시, 요청 큐, 스케줄링 |
| `src/vulkan/` `src/metal/` `src/cuda/` `src/rocm/` | GPU 백엔드별 구현 |
| `src/shaders/` | **GPU 컴퓨트 셰이더 154개** |
| `src/zinc_rt/` | 자체 GPU 런타임 (드라이버 우회, 자체 IR/ISA — 실험적) |

`src/shaders/` 파일명을 보면 최적화 밀도가 드러납니다. 예: `dmmv_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1.comp` — MoE 모델의 gate와 up 연산을 하나로 융합한 커널을 조합별로 전부 준비해 둔 형태입니다.

### 개발 · 실험 인프라

| 경로 | 역할 |
|---|---|
| `loops/` | **AI 에이전트(claude/codex) 자동 최적화 하네스.** 코드 수정 → 벤치마크 → 빨라지면 유지 / 느려지면 롤백을 반복 |
| `benchmarks/` | 벤치마크 코드 + 결과 JSON (커밋 해시, 바이너리 SHA-256까지 기록) |
| `tools/performance_suite.mjs` | 벤치마크 측정 도구 (공개 수치의 출처) |
| `research/` | **실패한 최적화 기록** (`*_DEAD_*.md`, `*_BLOCKED_*.md`) |
| `docs/` | 하드웨어 레퍼런스, 설계 문서, 로드맵 등 29개 문서 |
| `specs/` | 기능별 스펙 문서 |
| `site/` | Astro 기반 공식 웹사이트 소스 |
| `AGENTS.md` | AI 코딩 에이전트를 위한 작업 지침서 (약 25,000자) |
| `scripts/` | 설치 스크립트, 벤치마크 스윕, 배포 스크립트 |

특히 `research/` 폴더에 **실패 기록을 남기는 문화**와, 벤치마크에 커밋 해시·프롬프트·원본 샘플까지 기록하는 재현성 중심 접근은 참고할 만합니다.

---

## 3. 설치 가이드

### 3.0 지원 플랫폼 (가장 먼저 확인)

> **네이티브 윈도우는 지원하지 않습니다.**

| 환경 | 지원 | 백엔드 |
|---|---|---|
| macOS (M1~M5) | ✅ 지원, 설치 가장 쉬움 | Metal |
| Linux + AMD RDNA3/RDNA4 | ✅ 최우선 지원 | Vulkan 1.3 / ROCm(HIP) |
| Linux + Intel Arc (Xe2/Battlemage) | ✅ 지원 | Vulkan 1.3+ |
| Linux 또는 WSL2 + NVIDIA RTX 40/50 | ⚠️ 실험적 | CUDA |
| 네이티브 Windows | ❌ 미지원 | — |

메모리 가이드:

| VRAM / 통합메모리 | 실행 가능 범위 |
|---|---|
| 8GB | 대부분 모델에 빠듯함 |
| 16GB | 2B~8B 클래스 여유롭게 |
| 24GB | 20B 클래스 |
| 32GB | 27B 덴스 / 35B MoE |

### 3.1 방법 A — 설치 스크립트 (Linux x86_64 / macOS ARM64)

```bash
curl -fsSL https://raw.githubusercontent.com/zolotukhin/zinc/main/scripts/install.sh | bash
```

기본 설치 위치: `~/.local/share/zinc`, 심볼릭 링크: `~/.local/bin/zinc`
환경변수 `ZINC_VERSION`, `ZINC_INSTALL_DIR`, `ZINC_BIN_DIR`로 변경 가능합니다.
릴리즈 바이너리가 없거나 지원 플랫폼이 아니면 방법 B를 사용하세요.

### 3.2 방법 B — 소스 빌드 (권장)

#### ① Zig 설치 (0.15.2 이상 필수)

**macOS**

```bash
brew install zig
xcode-select --install
# Apple Silicon은 이 두 줄이면 준비 완료 (Vulkan/glslc/Python 불필요)
```

**Linux** — https://ziglang.org/download/ 에서 다운로드

```bash
wget https://ziglang.org/download/0.15.2/zig-linux-x86_64-0.15.2.tar.xz
tar xf zig-linux-x86_64-0.15.2.tar.xz
sudo mv zig-linux-x86_64-0.15.2 /opt/zig
echo 'export PATH=/opt/zig:$PATH' >> ~/.bashrc && source ~/.bashrc
zig version   # 0.15.2 이상 확인
```

#### ② GPU 준비물 (macOS는 생략)

**AMD / Intel Arc (Vulkan)**

```bash
sudo apt update
sudo apt install -y git libvulkan-dev vulkan-tools glslc
vulkaninfo --summary    # 여기서 GPU가 보여야 함
```

**AMD ROCm (RDNA4 검증 경로)**

```bash
rocminfo                # GPU 인식 확인
# amdhip64, hiprtc, hipblas 개발 라이브러리 필요
```

검증된 레퍼런스 스택: Linux kernel 7.2.2 / ROCm userspace 7.2.4 / Radeon AI PRO R9700 (`gfx1201`)

**NVIDIA (실험적)** — CUDA Driver API, NVRTC, cuBLAS, CUDA 런타임 필요

#### ③ 클론 및 빌드

```bash
git clone https://github.com/bmshin94/zinc.git
cd zinc
zig build -Doptimize=ReleaseFast
```

백엔드를 명시적으로 선택하는 경우:

```bash
# AMD ROCm
ROCM_PATH=/opt/rocm zig build -Dbackend=rocm -Doptimize=ReleaseFast

# NVIDIA CUDA (실험적)
CUDA_HOME=/usr/local/cuda zig build -Dbackend=cuda -Doptimize=ReleaseFast
```

옵션을 주지 않으면 Linux는 Vulkan, macOS는 Metal이 자동 선택됩니다.
빌드 결과물: `./zig-out/bin/zinc`

### 3.3 설치 검증

```bash
./zig-out/bin/zinc --check
```

`READY [OK]`가 출력되면 GPU 감지, 셰이더 자산, 런타임 초기화가 모두 정상입니다.

> **RDNA4 + Vulkan 조합에서는 아래 환경변수가 필수입니다.** 설정하지 않으면 느리거나 출력이 잘못될 수 있습니다.
>
> ```bash
> export RADV_PERFTEST=coop_matrix
> ```
>
> ROCm, Intel Arc, CUDA, macOS 사용자는 설정하지 않아도 됩니다.

---

## 4. 사용법

### 4.1 모델 다운로드

```bash
./zig-out/bin/zinc model list          # 내 GPU 프로필에 맞는 모델
./zig-out/bin/zinc model list --all    # 전체 카탈로그
./zig-out/bin/zinc model pull qwen35-9b-q4k-m   # SHA-256 검증 후 캐시에 설치
```

**카탈로그 모델**

| 모델 | 모델 ID | 필요 메모리 |
|---|---|---|
| Qwen 3.5 9B Q4_K_M | `qwen35-9b-q4k-m` | 8GB+ |
| Gemma 4 26B-A4B Q4_K_M | `gemma4-26b-a4b-q4k-m` | 16GB+ |
| Qwen 3.6 35B-A3B Q4_K_XL | `qwen36-35b-a3b-q4k-xl` | 24GB+ |
| Gemma 4 31B Q4_K_M | `gemma4-31b-q4k-m` | 24GB+ |
| Qwen 3.8 27B Q4_K_M | `qwen38-27b-q4k-m` | 24GB+ / 통합 32GB+ |

### 4.2 CLI로 한 번 실행

```bash
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "대한민국의 수도는?" --chat
```

`--chat`은 모델의 채팅 템플릿(시스템 프롬프트, 역할 태그)을 적용합니다.
**인스트럭션 튜닝 모델에서는 사실상 필수**이며, 없으면 단순 텍스트 이어쓰기로 동작합니다.

시스템 프롬프트를 명시하려면 (`--chat`, `--prompt`와 함께 사용):

```bash
./zig-out/bin/zinc --model-id qwen38-27b-q4k-m \
  --chat \
  --system-prompt "You are a helpful assistant. Answer directly." \
  --prompt "Review this code for concurrency bugs."
```

정상 동작 시 로그:

```
info(loader): Loading model: ...
info(forward): Prefill complete: ...
info(forward): Generated 256 tokens in ... ms — XX.XX tok/s
info(zinc): Output text: ...
```

### 4.3 웹 채팅 UI

```bash
./zig-out/bin/zinc chat
```

기본 포트 **9090**으로 서버가 뜨고 브라우저에 내장 채팅 UI가 열립니다.

### 4.4 API 서버 모드

```bash
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m -p 8080
```

기본 포트 **8080**. `http://localhost:8080/v1`이 OpenAI 호환 엔드포인트입니다.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")

res = client.chat.completions.create(
    model="qwen35-9b-q4k-m",
    messages=[{"role": "user", "content": "안녕!"}],
    stream=True,
)
```

### 4.5 로컬 파일 / Hugging Face 직접 사용

```bash
./zig-out/bin/zinc -m /path/to/model.gguf --prompt "안녕" --chat
./zig-out/bin/zinc -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --prompt "안녕" --chat
```

### 4.6 모델 관리

```bash
./zig-out/bin/zinc model use qwen35-9b-q4k-m   # 기본 모델 지정
./zig-out/bin/zinc model active                # 현재 기본 모델 확인
./zig-out/bin/zinc model rm -f qwen35-9b-q4k-m # 캐시에서 제거
```

### 4.7 주요 옵션

| 옵션 | 설명 |
|---|---|
| `-m, --model <path>` | GGUF 파일 경로 |
| `--model-id <id>` | 카탈로그 모델 ID |
| `-hf, --hf-repo <spec>` | Hugging Face 저장소 (`owner/model[:quant]`) |
| `--prompt <text>` | CLI 1회 실행 |
| `--chat` | 채팅 템플릿 적용 |
| `--system-prompt <text>` | 시스템 턴 추가 (`--chat` 필요) |
| `--raw` | 채팅 템플릿 자동 적용 안 함 |
| `-n, --max-tokens <n>` | 최대 생성 토큰 (기본 256) |
| `-c, --context <size>` | 컨텍스트 길이 (기본 자동) |
| `-d, --device <id>` | GPU 장치 번호 |
| `-p, --port <port>` | 서버 포트 (기본 8080, `chat`은 9090) |
| `--parallel <n>` | 최대 동시 요청 수 (기본 4) |
| `--kv-quant <bits>` | KV 캐시 양자화 0/2/3/4 — **메모리 부족 시 유용** |
| `--check` | 시스템 진단 |
| `--profile` | 런타임 프로파일링 |
| `--debug` | 상세 디버그 로그 |
| `--help-all` | 개발자 옵션 포함 전체 도움말 |

---

## 5. API 레퍼런스 요약

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/v1/chat/completions` | 채팅 추론 (스트리밍 SSE 지원) |
| POST | `/v1/completions` | 원시 텍스트 완성 (비스트리밍) |
| GET | `/v1/models` | 카탈로그, 설치 상태, 활성 모델, 메모리 상태 |
| POST | `/v1/models/pull` | 모델 비동기 다운로드 |
| POST | `/v1/models/activate` | 설치된 모델 활성화 |
| POST | `/v1/models/remove` | 캐시 모델 제거 |
| GET | `/health` | 헬스체크, 큐 카운터, 업타임, GPU 메모리/컨텍스트 상태 |
| GET | `/` 또는 `/chat` | 내장 채팅 UI |

### `/v1/chat/completions` 주요 필드

| 필드 | 기본값 | 설명 |
|---|---|---|
| `model` | 현재 모델 | 설치된 카탈로그 모델이면 자동 활성화 후 생성 |
| `session_id` | 없음 | **ZINC 고유 기능.** 같은 ID 재사용 시 프롬프트 접두부를 재활용해 속도 향상 |
| `messages` | 필수 | OpenAI 형식 (`system`/`user`/`assistant`/`tool`) |
| `max_tokens` | 256 | 최대 생성 토큰 |
| `temperature` | 0.0 | 0이면 greedy, 0~2로 클램프 |
| `top_p` | 1.0 | 0~1로 클램프 |
| `enable_thinking` | 모델 기본값 | thinking 블록 요청 |
| `stream` | false | SSE 스트리밍 |
| `tools` / `tool_choice` | 없음 / `auto` | ChatML·Qwen 계열 템플릿에서 함수 호출 지원 |

툴 콜링은 기본 활성화이며 `ZINC_TOOL_CALLING=0`으로 끌 수 있습니다.
ZINC은 툴을 **실행하지 않고**, 모델이 뱉은 `<tool_call>` 블록을 파싱해 구조화된 `tool_calls`로 돌려줄 뿐입니다. 실제 실행은 클라이언트 몫입니다.

---

## 6. 알려진 한계

공식 문서에 명시된 현재 상태입니다. 버그가 아니라 의도된 현 시점의 스펙입니다.

1. **디코드가 직렬화됨** — 동시 HTTP 요청은 받지만 생성은 엔진 락 하나 뒤에서 순차 처리됩니다. N번째 요청은 N-1이 끝날 때까지 대기합니다. 병렬 스트림을 전제로 설계하면 안 됩니다.
2. **`/v1/embeddings` 미구현** — 임베딩 엔드포인트가 없습니다. RAG가 필요하면 임베딩은 별도 프로세스(llama.cpp `llama-embedding`, 외부 API 등)로 처리해야 합니다.
3. **`/v1/completions`는 비스트리밍이며 샘플링 옵션이 무시됨** — 스트리밍·`temperature`·`top_p`·`enable_thinking`이 필요하면 `/v1/chat/completions`를 쓰세요.
4. **클라이언트 `stop` 시퀀스 미지원** — 모델/템플릿 종료 마커(`<|im_end|>` 등)로만 중단됩니다.
5. **`/v1/audio/*`, 이미지 입력, 파인튜닝, 배치 API 없음**
6. **전체적으로 실험적 소프트웨어** — 문서 첫 줄에 "Experimental software"로 명시되어 있습니다.

---

## 7. 수익화 아이디어

### 전제 조건 3가지

1. **ZINC 자체를 파는 것은 불가능합니다.** MIT 라이선스라 누구나 무료로 받을 수 있습니다.
2. **다만 상업적 이용은 완전히 합법입니다.** 부품으로 쓰는 것은 자유입니다.
3. **이 저장소는 원본의 포크입니다.** "ZINC 기반으로 구축했습니다"라고 밝히는 편이 법적으로도 평판상으로도 유리합니다.

> 핵심: **코드가 아니라 코드 주변에서 수익이 발생합니다.**

### 아이디어 6가지

#### 1. AMD·Intel GPU 로컬 AI 구축 대행 ⭐ 추천

로컬 AI 시장은 사실상 NVIDIA 독점이라, Radeon·Arc 사용자는 시작조차 못 하고 포기합니다.
ZINC은 정확히 그 공백을 겨냥한 도구입니다.

- 타겟: 라데온 게이밍 PC 유저, AMD 워크스테이션 보유 기업, Intel Arc 사용자
- 상품: 원격 세팅 대행 / 사내 챗봇 구축 / 월 유지보수
- 초기 비용: 0원 · 난이도: 낮음

#### 2. 온프레미스 사내 AI 구축 (단가 최대)

병원·법무법인·회계법인·제조업체는 데이터 외부 반출이 금지되어 ChatGPT를 못 씁니다.
ZINC의 OpenAI 호환 API 덕분에 기존 솔루션 연동도 쉽습니다.

- 구조: 서버 1대 + GGUF + ZINC
- 주의: **동시 처리 불가** → 5~10명 규모 팀부터 시작
- 난이도: 높음 (영업 비중 큼)

#### 3. GUI 설치 앱 (LM Studio 포지션)

ZINC의 최대 약점은 윈도우 미지원 + 터미널 필수라는 점입니다. 일반 사용자는 접근이 어렵습니다.

- 무료: 기본 채팅 / 유료: 다중 모델 관리, 문서 첨부, API 서버 모드
- 현실적 범위: macOS + Linux부터 (윈도우 문제는 Zig/Vulkan 빌드 레벨이라 해결 난이도가 높음)

#### 4. `loops/` 기반 성능 자동 튜닝 SaaS (독창성 1위)

`loops/`의 구조 — 코드 수정 → 벤치마크 → 개선 시 유지 / 퇴행 시 롤백 — 은 GPU 커널에 국한되지 않고 어떤 프로젝트에도 적용 가능합니다.

- 상품: GitHub 연동 → 야간 자동 실행 → "N% 개선된 PR" 자동 생성
- 시장성: 현재 AI 코딩 도구는 대부분 "기능 구현"에 몰려 있고, **성능 최적화 자동화는 비어 있는 영역**

#### 5. 콘텐츠 + 제휴 마케팅 (리스크 0)

"AMD 라데온으로 로컬 AI 돌리기"는 검색 수요 대비 콘텐츠가 희소합니다.

- 수익: 애드센스 + 하드웨어 제휴 + 1번 컨설팅으로의 유입 채널
- 다른 아이디어의 마케팅 채널 역할도 겸함

#### 6. 오픈소스 기여를 통한 커리어 자산화

`docs/ROADMAP.md`에 외부 기여가 필요한 항목이 정리되어 있습니다 (빌드 수정, 재현 가능한 버그 리포트, 테스트 커버리지, 문서 개선 등 진입 난이도가 낮은 항목 포함).
GPU 커널 최적화 인력은 희소하며, 공개 기여 이력이 그대로 포트폴리오가 됩니다.

### 추천 로드맵

| 시점 | 실행 |
|---|---|
| 즉시 (0원) | 5번 콘텐츠로 "AMD 로컬 AI" 키워드 선점 |
| 1~3개월 | 유입 문의를 1번 세팅 대행으로 전환, 첫 매출 |
| 6개월~ | 레퍼런스 축적 후 2번 기업 온프레미스로 단가 상승 |
| 별도 트랙 | 기술 지향이면 4번 자동 튜닝 SaaS |

### 하지 말아야 할 것

- ZINC을 그대로 리브랜딩해 판매 (평판 리스크, 시장성 없음)
- ZINC 기반 공개 API 크레딧 판매 (**디코드 직렬화로 물리적으로 불가**, 대형 사업자와 가격 경쟁 불가)
- 원작자 표기 없이 "자체 개발 엔진"으로 마케팅

---

## 8. React / PHP 연동 아키텍처

### 8.1 엔진을 React/PHP로 만들 수 있는가 → 불가능하며, 불필요

| | React / PHP | ZINC (Zig) |
|---|---|---|
| 실행 위치 | 브라우저 / 웹서버 | GPU 바로 위 |
| 역할 | UI, DB, 비즈니스 로직 | 초당 수십억 회 행렬 연산 |
| GPU 제어 | 불가 | Vulkan/Metal/CUDA 직접 제어 |
| 메모리 | 자동 관리 | 수동 정밀 제어 |

llama.cpp는 C++, PyTorch도 내부는 C++/CUDA입니다. 추론 엔진을 스크립트 언어로 쓰는 사례는 없습니다.

### 8.2 올바른 구조 — ZINC을 부품으로 사용

```
[React 프론트]  ←→  [PHP 백엔드]  ←→  [ZINC :8080]  ←→  [GPU]
   직접 개발          직접 개발          그대로 사용        하드웨어

   채팅 UI           회원/결제/DB        AI 추론
   관리자 화면       권한/사용량 제한
   모델 상태         대화 기록 저장
```

ZINC이 OpenAI 호환 API를 제공하므로, PHP 입장에서는 **"주소만 localhost인 ChatGPT"** 입니다.

### 8.3 PHP — 비스트리밍 호출

```php
<?php
$body = json_encode([
    'model'    => 'qwen35-9b-q4k-m',
    'messages' => [
        ['role' => 'system', 'content' => '너는 친절한 상담원이야.'],
        ['role' => 'user',   'content' => $_POST['message']],
    ],
    'max_tokens'  => 512,
    'temperature' => 0.7,
]);

$ch = curl_init('http://localhost:8080/v1/chat/completions');
curl_setopt_array($ch, [
    CURLOPT_POST           => true,
    CURLOPT_POSTFIELDS     => $body,
    CURLOPT_HTTPHEADER     => ['Content-Type: application/json'],
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT        => 300,   // 로컬 추론은 오래 걸릴 수 있음
]);
$res = json_decode(curl_exec($ch), true);

echo $res['choices'][0]['message']['content'];
```

### 8.4 PHP — SSE 스트리밍 프록시

```php
<?php
header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('X-Accel-Buffering: no');   // nginx 버퍼링 비활성화 (누락 시 스트리밍 안 됨)

$ch = curl_init('http://localhost:8080/v1/chat/completions');
curl_setopt_array($ch, [
    CURLOPT_POST       => true,
    CURLOPT_POSTFIELDS => json_encode([
        'messages'   => json_decode($_POST['messages'], true),
        'session_id' => $_POST['session_id'],  // 채팅방 ID를 넣으면 프리필 재사용
        'stream'     => true,
    ]),
    CURLOPT_HTTPHEADER    => ['Content-Type: application/json'],
    CURLOPT_WRITEFUNCTION => function ($ch, $chunk) {
        echo $chunk;
        @ob_flush(); flush();
        return strlen($chunk);
    },
]);
curl_exec($ch);
```

### 8.5 React — 스트리밍 수신

```jsx
async function send(text) {
  const res = await fetch('/api/stream.php', {
    method: 'POST',
    body: new URLSearchParams({
      messages: JSON.stringify([...history, { role: 'user', content: text }]),
      session_id: roomId,
    }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let answer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    for (const line of decoder.decode(value).split('\n')) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') return;

      const delta = JSON.parse(data).choices[0]?.delta?.content;
      if (delta) setAnswer(answer += delta);
    }
  }
}
```

OpenAI 스트리밍 코드와 동일한 형태이므로, 나중에 클라우드 API로 전환할 때 URL만 바꾸면 됩니다.

### 8.6 관리자 화면에 쓸 수 있는 엔드포인트

| 엔드포인트 | 용도 |
|---|---|
| `GET /health` | 서버 상태, 대기 큐 길이, GPU 메모리 → 관리자 대시보드 |
| `GET /v1/models` | 설치 목록 + 활성 모델 → 모델 선택 UI |
| `POST /v1/models/pull` | 모델 다운로드 버튼 |
| `POST /v1/models/activate` | 모델 전환 버튼 |

### 8.7 반드시 고려할 제약

**ZINC은 한 번에 한 요청만 처리합니다(디코드 직렬화).** 애플리케이션 레벨에서 방어해야 합니다.

- 요청 큐 구현 (Redis 등) 후 "대기 N명" 표시
- 사용자당 동시 요청 1개로 제한
- `/health`의 큐 카운터를 폴링해 혼잡 시 사전 안내
- cURL 타임아웃 넉넉히 (최소 300초)
- 서버 기동 시 워밍업 요청 1회 (첫 요청은 모델 로딩으로 느림)
- 임베딩이 필요하면 별도 프로세스로 분리
- `stop` 시퀀스가 필요하면 애플리케이션에서 후처리

### 8.8 적용 예시

| 제품 | React 담당 | PHP 담당 | 난이도 |
|---|---|---|---|
| 사내 AI 챗봇 | 채팅 UI, 관리자 | 로그인, 권한, 대화 저장 | 낮음 (시작점 추천) |
| 고객사 상담봇 | 임베드 위젯 | 대화 로그, 통계 | 중간 |
| 문서 요약 SaaS | 업로드 UI | 파일 파싱 → ZINC 호출 | 중간 |
| 모델 관리 대시보드 | 상태 차트 | `/health` 폴링 | 낮음 |

---

## 9. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `zig: command not found` | Zig PATH 확인 (`zig version`, 0.15.2 이상) |
| `--check`에서 GPU 미인식 | `vulkaninfo --summary` 또는 `rocminfo`가 먼저 통과해야 함 |
| 출력이 이상하거나 매우 느림 (RDNA4 Vulkan) | `export RADV_PERFTEST=coop_matrix` 설정 |
| 메모리 부족 | 더 작은 모델 / `-c 2048` / `--kv-quant 4` |
| `hipErrorNoBinaryForGpu` | `amdgpu` 드라이버 ↔ ROCr ↔ ROCm 유저스페이스 버전 불일치 (`docs/ROCM.md` 참고) |
| ROCm 라이브러리 로드 실패 | `export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH` |
| 스트리밍이 한 번에 몰려서 옴 | nginx `X-Accel-Buffering: no`, PHP `ob_flush()`/`flush()` 확인 |
| 동시 요청 시 응답 지연 | 정상 동작 (디코드 직렬화) — 애플리케이션 큐로 대응 |

---

## 참고 문서

저장소 내부 문서:

- `docs/GETTING_STARTED.md` — 설치 및 첫 실행
- `docs/HARDWARE_REQUIREMENTS.md` — GPU·OS·드라이버 요구사항
- `docs/RUNNING_ZINC.md` — CLI 플래그, 서버 모드
- `docs/API.md` — API 전체 레퍼런스
- `docs/ROCM.md` — AMD ROCm 설정 및 튜닝
- `docs/RDNA4_TUNING.md` — RDNA4 성능 튜닝
- `docs/DEVELOPMENT.md` — 빌드, 테스트, 기여
- `docs/ROADMAP.md` — 기여가 필요한 영역
- `AGENTS.md` — AI 에이전트 작업 지침
