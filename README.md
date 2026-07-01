# Retail THOR Demo

This project builds a retail-shelf embodied planning data generation pipeline on AI2-THOR.

The first-stage runtime is AI2-THOR only. InternUtopia and SARI are used only as read-only references for robot appearance, store layout, grocery concepts, shelf interactions, NPC roles, and checkout/cart ideas. They are not engineering dependencies for this stage.

## Scope

- Robot: AI2-THOR default agent, abstracted as a humanoid high-level embodied agent.
- Shelf: by default, one AI2-THOR receptacle is used as the retail shelf: `FloorPlan1` / `shelf_FloorPlan1_010` / `CounterTop|-00.08|+01.15|00.00`.
- Products: only products on the configured shelf are exported by default. The current default shelf contains `Apple`, `Bowl`, `Bread`, and `Tomato` after semantic enrichment from `configs/products.yaml`.
- Tasks:
  - `object_loco_navigation`: find a product from a user description.
  - `social_loco_navigation`: interact with an NPC to clarify or recommend substitutes.
  - `loco_manipulation`: pick a product and place it in a cart-like receptacle or other target.

This stage does not claim real humanoid low-level control, real grasp control, full supermarket reconstruction, or Sim2Real control transfer.

## Setup

```bash
conda create -n thor_retail python=3.10 -y
conda activate thor_retail
pip install -r retail_thor_demo/requirements.txt
```

On mac M4, run AI2-THOR with the local display first. If Unity startup fails, record the error from `outputs/reports/runtime_env.json` and try a compatible Python or AI2-THOR version before changing the project design.

`scripts/00_ensure_ai2thor_build.py` downloads the macOS AI2-THOR Unity build with resume support. Large build downloads explicitly bypass Clash/system proxy variables by using `curl --noproxy "*"` and removing `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` from the download subprocess.

## Commands

```bash
cd retail_thor_demo
python scripts/00_ensure_ai2thor_build.py
python scripts/01_smoke_test.py
python scripts/02_scan_scene.py
python scripts/03_build_catalog.py
python scripts/04_generate_episodes.py --scenes FloorPlan1 --num-episodes 20 --seed 0 --task-types find_product,dialogue_find_or_substitute,pick_and_place --demo-mode --output-root outputs/single_shelf_demo
python scripts/05_replay_episode.py --episode outputs/single_shelf_demo/episodes/episode_000001.json
python scripts/06_validate_dataset.py --episodes outputs/single_shelf_demo/episodes
```

`configs/single_shelf.yaml` controls the default one-shelf scope. To rebuild a multi-shelf catalog for debugging, run `python scripts/03_build_catalog.py --disable-single-shelf`.

To run VLM-driven navigation, set `OPENAI_API_KEY` and optionally `OPENAI_VLM_MODEL`, then run:

```bash
python scripts/08_vlm_navigate.py --instruction "导航到货架前，并在看清货架商品后停止。" --max-steps 12
```

## Data Format

Each episode stores instruction text, structured target constraints, product catalog snapshot, shelf graph, NPC dialogue, high-level plan, simulator action trace, observations, metrics, success flag, failure reason, and provenance.

Key fields:

- `capability_family`: one or more of `object_loco_navigation`, `social_loco_navigation`, `loco_manipulation`.
- `high_level_plan`: actions such as `navigate_to_shelf`, `search_object`, `ask_npc`, `pick_object`, `place_object`, `finish`.
- `sim_action_trace`: AI2-THOR actions such as `Teleport`, `OpenObject`, `PickupObject`, `PutObject`, `Done`.
- `physical_control`: first-stage status is `not_supported`.

## Asset References

`configs/asset_references.yaml` records ideas borrowed from InternUtopia and SARI: humanoid robot appearance, GRScenes object categories, SARI store layouts, grocery barcode/expiration metadata, checkout counter concepts, and openable/sliding door interactions. These references guide semantic metadata and future custom shelf model specs; they do not require importing those runtimes.
