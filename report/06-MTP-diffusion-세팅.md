# 06 — MTP / Diffusion 세팅 가이드 (복붙 + 트러블슈팅)

> 단일 H100 80GB · vLLM docker 기준. AR(기본) 서빙은 [04-재현과데이터.md](04-재현과데이터.md) §3, 모델별 파서/이미지 정본은 04 §4. 본 문서는 **MTP·diffusion 두 기법의 서빙을 실행 가능한 형태로** + **무엇이 어렵고 어떻게 푸는지**를 다룬다.

공통 docker 인자(아래 명령에서 반복): `--gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 --no-enable-prefix-caching --host 0.0.0.0 --port 8000`. 정확도 측정 시 `--reasoning-parser gemma4|qwen3` + thinking 프록시(:8001), 속도만 잴 땐 둘 다 불필요.

---

## 1. MTP (speculative decoding)

### 1.1 두 방식 — 모델 계열로 갈림

| 방식 | 대상 | `--speculative-config` |
|---|---|---|
| **별도 드래프터** | Gemma 4 전부 | `'{"model":"<base>-assistant","num_speculative_tokens":4}'` |
| **임베디드 MTP head** | Qwen 3.6 전부 | `'{"method":"mtp","num_speculative_tokens":2}'` |

판별: 모델 카드에 동반 드래프터(`*-assistant`)가 있으면 별도 드래프터형, 모델 자체에 MTP head가 있으면 임베디드형.

### 1.2 복붙 serve 명령

**Gemma 31B-fp8 + MTP (별도 드래프터):**
```bash
HF=<HF_TOKEN>
sudo docker run -d --name gd --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
  vllm/vllm-openai:nightly \
  --model google/gemma-4-31B-it --quantization fp8 \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}' \
  --max-model-len 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --host 0.0.0.0 --port 8000
  # 정확도 측정이면 + --reasoning-parser gemma4
```

**Qwen27-fp8 + MTP (임베디드 head):**
```bash
sudo docker run -d --name gd ... vllm/vllm-openai:v0.22.1 \
  --model Qwen/Qwen3.6-27B-FP8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --max-num-seqs 256 --max-model-len 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching
```

### 1.3 셀별 정본 (실측에 쓴 9셀)

| 셀 | base model | 이미지 | speculative-config | 추가 플래그 |
|---|---|---|---|---|
| 12B-bf16 | `google/gemma-4-12B-it` | nightly | `{"model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":4}` | — |
| 12B-fp8 | `google/gemma-4-12B-it` | nightly | (위와 동일) | `--quantization fp8` |
| 26B-bf16 | `google/gemma-4-26B-A4B-it` | nightly | `{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}` | — |
| 26B-fp8 | (위) | nightly | (위) | `--quantization fp8` |
| 26B-int8 | (위) | nightly | (위) | `--quantization int8_per_channel_weight_only` |
| 31B-fp8 | `google/gemma-4-31B-it` | nightly | `{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}` | `--quantization fp8` |
| 31B-qat | `google/gemma-4-31B-it-qat-w4a16-ct` | nightly | `{"model":"google/gemma-4-31B-it-qat-q4_0-unquantized-assistant","num_speculative_tokens":4}` | — |
| qwen35-fp8 | `Qwen/Qwen3.6-35B-A3B-FP8` | nightly | `{"method":"mtp","num_speculative_tokens":2}` | — |
| qwen27-fp8 | `Qwen/Qwen3.6-27B-FP8` | v0.22.1 | `{"method":"mtp","num_speculative_tokens":2}` | `--max-num-seqs 256` |

### 1.4 속도(speedup)·무손실 측정
- **속도**: `bench_tp.py`(non-streaming total_tp). **streaming `decode_tok/s` 절대 금지** — spec decode는 토큰을 버스트로 커밋해 순간율이 왜곡 → MTP를 ~3배 과소측정(라운드1 "MTP 손해" 오결론의 원인).
- **무손실**: ① 정확도 = MTP-on으로 GPQA full을 AR과 동일조건 직접 측정(Δ≈0 확인). ② 토큰동일성 = greedy(temp=0) off-vs-on, **반드시 대조군(같은 AR 두 번 off-vs-off2)과 함께** — vLLM은 temp=0이라도 비결정이라 대조군 없이는 분기 원인 귀속 불가. (상세 [03 §C](03-결과와해석.md))

### 1.5 무엇이 어려운가 → 해결

| 난점 | 증상 | 해결 |
|---|---|---|
| **드래프터+KV OOM** | 31B-bf16+MTP가 engine init 실패 | bf16은 가중치 62GB라 드래프터+cudagraph+KV 동시 수용 불가 → **fp8/qat에서만 MTP**(31B-bf16-MTP는 NA로 명시). util을 **낮추면 더 악화**(util은 점유 상한 — KV가 더 작아짐). |
| **cudagraph 메모리 부족** | 서빙은 되나 capture OOM | `-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` + util 0.85(드래프터 여유). 최후엔 `--enforce-eager`(단 속도 왜곡 → 비교 무효, 각주 처리) |
| **enforce-eager 아티팩트** | short < 8K 역전(예 short 21 < 8K 59) | eager+MTP는 스텝당 다중 forward라 short에서 런치 오버헤드 지배 → 그 수치는 cudagraph 수치와 **같은 표에 섞지 말 것** |
| **qat 드래프터 불일치** | 31B-qat에 일반 드래프터 쓰면 mismatch | qat 전용 `...-qat-q4_0-unquantized-assistant` 사용 |
| **이미지 회귀** | Qwen27이 nightly에서 MTP DIED | Qwen27은 **v0.22.1**로 고정(nightly `Qwen3_5MTP` 회귀) |
| **`num_speculative_tokens` 과다** | acceptance 낮아 오히려 느림 | Gemma 드래프터 4, Qwen head 2가 실측 최적. 수용률 낮으면 줄임 |

---

## 2. Diffusion (block diffusion, diffusiongemma-26B-A4B)

### 2.1 복붙 serve 명령
```bash
sudo docker run -d --name gd --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -p 8000:8000 \
  vllm/vllm-openai:gemma \
  --model google/diffusiongemma-26B-A4B-it \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 4 \
  --no-enable-prefix-caching --host 0.0.0.0 --port 8000
  # 정밀도: + --quantization fp8 (또는 int8_per_channel_weight_only); bf16은 플래그 없음
  # 파서 없음(--reasoning-parser 미지정)
```
- **이미지 = `vllm/vllm-openai:gemma`**(블록 디퓨전 전용). nightly/stable로는 안 뜸.
- 측정 파라미터: `max_gen_toks=4096`, concurrency=4, temp0.6/top_p0.95.

### 2.2 무엇이 어려운가 → 해결

| 난점 | 증상 | 해결 |
|---|---|---|
| **콜드스타트 오측정** | 첫 측정이 82 tok/s로 느림("뭔가 틀렸나?") | 첫 컴파일/워밍업이 섞임 → **warmup 1회 후 재측정**(순수 generation 615/864 tok/s). diffusion은 prefill/decode 경계가 없어 warmup 필수 |
| **파서 없음 → thinking 혼입** | content에 `thought\n...` 섞여 나옴 | **reasoning-parser 미지정이 정상**(:gemma는 파서 없음). 답 추출은 robust하게 — lm-eval flexible / inspect / HRET 마지막-(X). 파서 억지로 붙이지 말 것 |
| **동시성 과다 OOM** | max-num-seqs 크면 죽음 | **`--max-num-seqs 4`**(canvas가 시퀀스당 메모리 큼). concurrency도 4로 |
| **canvas/컨텍스트 한계 오해** | 16384만 되는 줄 앎 | 실측은 **32K·48K까지 서빙·NIAH 됨**(16384는 기본일 뿐). 단 긴문맥 **말단(depth 90%) 회수는 실패**(2/3) — 하드추론 약점과 같은 결 |
| **NIAH HTTP400** | max-model-len=출력여유 초과 시 400 | needle 프롬프트 + max_tokens(512)가 cap 초과 → target을 cap보다 충분히 낮춰 출력여유 확보(예 16384 셀은 400, 32K/48K는 정상) |
| **과제 적합성** | GPQA가 낮게(0.57) 나옴 | diffusion은 **추론깊이에 손실 집중**(GPQA −0.20). 언어이해·지시준수는 거의 보존 → 저지연·대량·얕은추론에 쓰고 하드추론엔 부적합(특성, 버그 아님) |

---

## 3. 빠른 점검 체크리스트 (둘 공통)
1. **이미지 맞나** — MTP(Gemma/Qwen35=nightly, Qwen27=v0.22.1), diffusion=gemma.
2. **`-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** 넣었나(cudagraph OOM 회피).
3. **`--no-enable-prefix-caching`** — 공정 속도 측정 전제.
4. **속도는 non-streaming total_tp**, MTP는 warmup, diffusion도 warmup.
5. 정확도면 **파서(MTP) / 무파서(diffusion) + thinking 프록시**, 추출은 robust(마지막 답).
6. OOM이면 **util을 올려라**(내리지 말 것) + fp8/kv-fp8.
