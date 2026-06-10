# paper/ — 논문 설계·서술 문서 모음

> Anti-VEGF VLM XAI 논문의 **디자인·흐름·작성 담당 문서**를 한곳에 모은 폴더.
> 실험 코드·데이터·결과 산출물은 옆 폴더 `../paper_xai_antivegf/`에 그대로 둔다.
> (2026-06-10: `paper_xai_antivegf/`에서 이동.)

## 현재 spine (2026-06-10 재정렬)
**"The Single-Image Illusion: Why SFT Cannot Instill Anti-VEGF Treatment Decisions
in Medical VLMs"** — 단일 pre-treatment 이미지로 anti-VEGF continue/stop 결정을 instill
하려는 시도가 (SFT·counterfactual·attention-KL 어느 것으로도) 정보론적·기제적으로 막혀
있음을 3개 백본에서 실증. 포지셔닝 = 모델 결함이 아니라 **패러다임 한계의 인과 실증**.

## 문서
| 파일 | 내용 |
|------|------|
| `DESIGN.md` | thesis·동기·기여(C1~C3)·포지셔닝·데이터·**핵심 발견 (A)~(E)**·위험 |
| `EXPERIMENTAL_PROTOCOL.md` | 백본 3·과제 4·instill arm(A/B/C/D+meta)·분할·메트릭·XAI/KG·실험매트릭스 |
| `PAPER_OUTLINE.md` | 섹션 골격(3단계 서사)·claim↔도표 매핑 |
| `references.bib` | 인용(attention-not-explanation, shortcut learning, forgetting, Wang KG 등) |

## 발견 라벨 (A)~(E) — 전 문서·`../paper_xai_antivegf/sft_data/matrix.md`와 1:1
- **(A)** 보이는 biomarker는 SFT로 instill (백본무관, 유일 positive)
- **(B)** continue/stop 결정 collapse + 예후/반응 ≤majority (정보론적 천장, 백본무관)
- **(C)** attention≠explanation (attention-KL 정렬해도 결정 불변)
- **(D)** 막힌 신호 일부는 pre-treatment 메타에 (백본의존)
- **(E)** ★ SFT의 역설 — 정형 CoT SFT가 그 메타 판별력을 덮음 (tier2 한정)

## 경로 규약
문서는 여기(`paper/`). **코드/데이터/결과(`code/`, `sft_data/`, `fluid_masks_v2/`)는
`../paper_xai_antivegf/` 기준 상대경로.** 결과 원자료: `../paper_xai_antivegf/sft_data/matrix.{md,json}`,
분석 전문: `../paper_xai_antivegf/sft_data/MATRIX_ANALYSIS_v0.3.md`, 핸드오프:
`../paper_xai_antivegf/HANDOFF.md`.
