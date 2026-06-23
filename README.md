# LLM 추론 평가 (Gemma 4 / Qwen 3.6, H100)

> **완료.** 상세 정본 = [report/](report/README.md) (01프로토콜·02시행착오·03결과·04재현·05-2부리그). 본 README가 결과 요약·진입점이다. Phase 시절 단편(REPORT/MATRIX/RESULTS_PHASE3/PLAN)은 [legacy/phase-docs/](legacy/phase-docs/)로 아카이브.

Gemma 4(12B·26B-A4B·31B·diffusiongemma)·Qwen 3.6(27B·35B-A3B)을 단일 H100 80GB에서 **정밀도 × 추론기법 × 컨텍스트**로 측정. 정확도는 thinking-on 동등조건 하에 **3프레임워크(lm-eval/inspect_ai/HRET)** 교차.

**핵심 결론:** ① **양자화(fp8/int8-MoE/qat) 정확도 무손실** — 속도/메모리만 이득. ② **MTP(speculative decoding) 정확도 무손실 가속**(n=198 확정, 평균 Δ≈0; dense 2.6× / MoE 1.4~1.8×). ③ **저활성 MoE가 프론티어 지배** — dense급 정확도에 3~5× 속도; **diffusion은 4× 최速이나 하드추론·긴문맥 말단 약함**.

## 1부 — 통합 결과표 (Gemma 4 / Qwen 3.6)

한 행 = (모델 × 정밀도 × 기법). AR/MTP/diffusion 독립 행. 해석·축별 슬라이스는 [report/03](report/03-결과와해석.md), 전 수치 근거 [results_consolidated.csv](results_consolidated.csv).

> 범례 — 속도=tok/s(short/8K), MTP 행 괄호=AR 대비 speedup · 6지표=accuracy(1.0) · MTP 행은 속도+GPQA(lm/insp)만 측정(나머지 4지표 —) · **†**=eager 폴백(공정 속도 아님, 단일 H100 메모리 제약).

| 모델 | 정밀도 | 기법 | 속도 s/8K | lm-GPQA | insp-GPQA | MMLU-Pro | IFEval | haerae | KMMLU | NIAH |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gemma 12B** dense | bf16 | AR | 82/78 | 0.662 | 0.652 | 0.788 | 0.913 | 0.832 | 0.596 | — |
|  |  | MTP | 184/148 (2.24×) | 0.657 | 0.611 | — | — | — | — | — |
|  | fp8 | AR | 117/110 | 0.652 | 0.616 | 0.814 | 0.916 | 0.844 | 0.610 | 3/3 @241K |
|  |  | MTP | 199/182 (1.70×) | 0.641 | 0.636 | — | — | — | — | — |
|  | qat | AR | 131/114 | 0.636 | 0.581 | 0.782 | 0.910 | 0.826 | 0.570 | — |
| **Gemma 26B-A4B** MoE | bf16 | AR | 200/188 | 0.763 | 0.727 | 0.870 | 0.930 | 0.890 | 0.644 | — |
|  |  | MTP | 322/312 (1.61×) | 0.773 | 0.717 | — | — | — | — | — |
|  | fp8 | AR | 226/212 | 0.773 | 0.707 | 0.798 | 0.919 | 0.902 | 0.626 | 3/3 @241K |
|  |  | MTP | 389/336 (1.72×) | 0.783 | 0.737 | — | — | — | — | — |
|  | int8 | AR | 218/201 | 0.758 | 0.732 | 0.824 | 0.923 | 0.890 | 0.614 | — |
|  |  | MTP | 382/331 (1.75×) | 0.808 | 0.712 | — | — | — | — | — |
| **Gemma 31B** dense | bf16(kvfp8) | AR | 40/35 | 0.828 | 0.788 | 0.840 | 0.941 | 0.904 | 0.692 | 3/3 @241K |
|  |  | MTP | NA¹ | — | — | — | — | — | — | — |
|  | fp8 | AR | 68/62 | 0.849 | 0.763 | 0.862 | 0.944 | 0.898 | 0.708 | — |
|  |  | MTP | 178/150 (2.62×) | 0.818 | 0.747 | — | — | — | — | — |
|  | qat | AR | 90/71 | 0.833 | 0.773 | 0.850 | 0.943 | 0.904 | 0.688 | — |
|  |  | MTP | 214/143 (2.38×) | 0.823 | 0.758 | — | — | — | — | — |
| **Qwen35-A3B** MoE | bf16† | AR | 15/15† | 0.808 | — | — | — | — | — | — |
|  | fp8 | AR | 217/205 | 0.803 | 0.838 | 0.862 | 0.904 | 0.834 | 0.586 | 3/3 @241K |
|  |  | MTP | 311/318 (1.43×) | 0.788 | 0.828 | — | — | — | — | — |
|  | int8† | AR | 15/15† | 0.788 | — | — | — | — | — | — |
| **Qwen27** dense-hybrid | bf16 | AR | 48/45 | 0.798 | 0.848 | 0.856 | 0.939 | 0.846 | 0.628 | — |
|  | fp8 | AR | 79/74 | 0.828 | 0.854 | 0.846 | 0.925 | 0.850 | 0.634 | 3/3 @241K |
|  |  | MTP | 147/142 (1.86×) | 0.818 | 0.854 | — | — | — | — | — |
|  | int8 | AR | 48/45 | 0.818 | 0.874 | 0.868 | 0.932 | 0.854 | 0.658 | — |
| **diffusion 26B-A4B** | fp8 | diff | 864/578 | 0.571 | 0.606 | 0.706 | 0.890 | 0.85 | 0.576 | 2/3 @32~48K² |
|  | bf16 | diff | 616/372 | 0.596 | 0.631 | 0.738 | 0.893 | 0.848 | 0.576 | — |
|  | int8 | diff | 759/510 | 0.576 | 0.672 | 0.744 | 0.897 | 0.85 | 0.544 | — |

¹ 31B-bf16+MTP는 드래프터+KV가 단일 H100 80GB에 안 들어가 측정 불가. ² diffusion NIAH는 depth 90% 실패로 2/3. NIAH "—"행은 동일 아키텍처 대표셀로 검증 대체.

**한 눈 요약:** 정밀도를 바꿔도 6지표 거의 불변(양자화 무손실) · MTP 행은 AR 대비 1.4~2.6×(dense>MoE)인데 GPQA는 ±0.05 노이즈 · NIAH AR 3/3 vs diffusion 2/3.

## 2부 — 신규 6모델 경량 평가

<details>
<summary><b>2026 신규 6종 결과 펼치기</b> — 서드파티·distill·finetune (GPQA + MMLU-Pro + KMMLU, thinking-on·fp8). 상세: <a href="report/05-2부리그.md">report/05</a></summary>

| 모델 | 종류 | 속도 s/8K | GPQA | MMLU-Pro | KMMLU |
|---|---|---:|---:|---:|---:|
| **Nex-N2-mini** | 35B 하이브리드(agentic) | 211/199 | **0.747** | 0.836 | 0.634 |
| **EXAONE-4.5-33B** | dense VL (LG) | 70/64 | 0.556 | **0.858** | **0.675** |
| GLM-4.7-Flash | 30B-A3B MoE | 177/163 | 0.545 | 0.744 | 0.494 |
| VibeThinker-3B | 3B 추론특화 | 276/254 | 0.535 | 0.310✱ | 0.266✱ |
| Qwen3.6-35B-A3B-Claude-distill | 35B-A3B MoE | 211/200 | 0.515 | 0.756 | 0.518 |
| Nemotron-3-Nano-Omni-30B-A3B | Mamba2-Tx 하이브리드 MoE | 331/310 | 0.449 | 0.658 | 0.434 |

> EXAONE만 MMLU-Pro/KMMLU N=200(과대생성으로 N=500 비현실적), 나머지 N=500 / GPQA full 198 · ✱VibeThinker MMLU-Pro/KMMLU는 off-domain(수학/코드 특화) · gemma-4-12B-coder는 vLLM GGUF 미지원(gemma4_unified)으로 제외.

**해석:** **EXAONE-4.5·Nex-N2가 지식(MMLU-Pro 0.84~0.86)·한국어(KMMLU 0.63~0.68)는 1부급**이나 GPQA(하드추론)는 중간. **VibeThinker-3B는 GPQA 0.535로 30B급**(STEM 추론 특화)이지만 지식·한국어는 바닥(off-domain). **Claude-distill은 base(qwen35 0.80/0.86)보다 하락** — 화제성 distill ≠ 개선. 서빙 안정화 R&D(모델별 런타임/파서/mm비활성/native·online fp8)는 [report/05 §C](report/05-2부리그.md).

</details>

## 구조
| 경로 | 내용 |
|---|---|
| **[report/](report/README.md)** | **★ 상세 정본** — 01프로토콜·02시행착오·03결과·04재현·05-2부리그·06-MTP/diffusion세팅 |
| [results_consolidated.csv](results_consolidated.csv) | 마스터 데이터 224행 (전 표의 근거) |
| `context/` | Phase 2 — (모델×정밀도)별 최대 안정 컨텍스트([FINDINGS](context/FINDINGS.md)) |
| `evals/` | 평가 배터리 러너 (lm-eval / inspect_ai / HRET) |
| `legacy/h100-round1/` | 1차 H100 라운드(매트릭스 + 트러블슈팅) — 아카이브 |
| `legacy/phase-docs/` | Phase 시절 단편(REPORT/MATRIX/RESULTS_PHASE3/PLAN) — 아카이브 |
| `legacy/` (그 외) | 초기 A100 라운드 보고서·로그 |
