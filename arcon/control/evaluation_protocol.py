#!/usr/bin/env python3
"""Material settings, motion lengths and mesh-level mapping for paper evaluation."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import sys

MOTIONS = {"01_01": 270, "07_02": 112, "104_54": 92, "55_27": 240}
GARMENTS = ("tank", "tshirt", "longsleeve", "dress")
LEVELS = {"Lv.1": ("garments_dict.pkl", ""),
          "Lv.2": ("garments_dict_multi_res.pkl", "_1.5"),
          "Lv.3": ("garments_dict_multi_res.pkl", "_2.0"),
          "Lv.4": ("garments_dict_high_res.pkl", "")}
MATERIAL = dict(density=0.20022, lame_mu=23600., lame_lambda=44400., bending_coeff=3.962e-5)
SIX = ("stretching_energy_loss", "bending_energy_loss", "collision_penalty_loss",
       "inertia_loss", "gravitational_energy_loss", "friction_energy_loss")
QUALITY = ("penetration_fraction", "mean_penetration_depth",
           "contact_neighbor_green_strain_rms", "contact_neighbor_edge_stretch_rms",
           "inertia_loss", "affected_rest_area_fraction", "persistent10_area_fraction")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def finite_mean(values):
    """Never turn a partly invalid sequence into a complete finite mean."""
    return math.fsum(values) / len(values) if values and all(v is not None and math.isfinite(v) for v in values) else None


def case_dir(root, model, case):
    return Path(root) / "evals" / model / case


def input_spec(manifest, case, model=None):
    c = manifest["cases"][case]
    spec = dict(sequence=manifest["sequences"][c["motion"]]["converted"],
                mesh_path=c["prepared_mesh"], mesh_key=c["mesh_key"], horizon=c["horizon"])
    if model is not None:
        spec["checkpoint"] = manifest["models"][model]["checkpoint"]
    return spec


def prepare(args):
    import numpy as np
    root, library = args.output.resolve(), args.library_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("A prepared manifest is immutable: " + str(root))
    source = library / "external/PB4U-GNet-main"
    os.environ.update(PB4U_PROJECT=str(source), PB4U_DATA=str(library / "data/pb4u_official"))
    sys.path.insert(0, str(source))
    from utils.data_making import convert_vto_to_pkl
    from utils.coarse import make_graph_from_faces, make_coarse_edges
    import networkx as nx
    earlier = read(args.native_preparation)
    catalog = read(earlier["source_catalog"])
    available = {"_".join(Path(r["path"]).stem.split("_")[-2:]) for r in catalog["tree"]
                 if r["path"].startswith("tshirt/simulations/tshirt_shape00_") and r["path"].endswith(".pkl")}
    train_path = library / "data/pb4u_official/aux_data/datasplits/train.csv"
    with train_path.open() as stream:
        train = {"_".join(r["id"].split("_")[-2:]) for r in csv.DictReader(stream)}
    if available - train != set(MOTIONS):
        raise ValueError("Native held-out motion inventory differs")
    result = dict(state="prepared", protocol="native_full_sequence_20261017_v1", created_at=now(),
        control_revision=args.control_revision, source_work=str(args.source_work.resolve()),
        library_root=str(library), output=str(root), material=MATERIAL,
        model_source_revision="928ec6c", reference_model="Pb4U",
        motion_derivation=dict(public_count=len(available), train_count=len(train), held_out=sorted(available-train),
                               source_catalog=earlier["source_catalog"], train_split=str(train_path)),
        sequences={}, meshes={}, cases={}, models={},
        initialization=dict(body_shape="shape00", smpl="SMPL_FEMALE.pkl", separate_arms=True,
            conversion=dict(start=0, n_frames=None, n_inter_steps=0, n_zeropose_interpolation_steps=30),
            common_initial_ts=1/3, regular_ts=1/30, ground_truth_reset=False,
            native_method_initialization_retained=True),
        aggregation="Full per-case temporal means; equal mean of all16 cases within EACH level; explicit failed denominators",
        reference_regions="Mature Pb4U full native prestate, signed distance<2mm plus one mesh hop; same masks for every method",
        paper_scope="Unified native full-sequence reevaluation, not an undocumented exact Table1 row-for-row reproduction",
        cost=dict(reference="Pb4U", fractions=[.25,.5,.75,1.], warmups=1, repeats=3,
                  includes="native online sample preparation and entire model", excludes="static loading, metrics, export"))
    for motion, expected in MOTIONS.items():
        raw_path = Path(earlier["sequences"][motion]["source"])
        target = root / "inputs/sequences" / (motion + ".pkl")
        convert_vto_to_pkl(raw_path, target, n_zeropose_interpolation_steps=30)
        with raw_path.open("rb") as stream:
            raw = pickle.load(stream, encoding="latin1")
        with target.open("rb") as stream:
            converted = pickle.load(stream, encoding="latin1")
        horizon = len(converted["body_pose"]) - 2
        if horizon != expected:
            raise ValueError((motion, horizon, expected))
        result["sequences"][motion] = dict(source=str(raw_path), converted=str(target),
            raw_frames=len(raw["pose"]), converted_frames=horizon+2, horizon=horizon,
            betas=np.asarray(converted["betas"]).tolist(), initial_translation=converted["transl"][0].tolist(),
            prediction_pose_indices=[2,horizon+1])
    aux = library / "data/pb4u_official/aux_data"
    for level, (filename, suffix) in LEVELS.items():
        with (aux / filename).open("rb") as stream:
            meshes = pickle.load(stream)
        for garment in GARMENTS:
            key = garment + suffix
            selected = meshes[key]
            # Geometry, LBS and handles remain the author asset; only cache its
            # exact native hierarchy so all cases can reuse the same input.
            if "center" not in selected:
                selected["center"] = nx.center(make_graph_from_faces(selected["faces"]), usebounds=True)
            selected.setdefault("coarse_edges", {})
            for center in selected["center"]:
                cached = selected["coarse_edges"].get(center, {})
                if not all(i in cached for i in range(4)):
                    computed = make_coarse_edges(selected["faces"], center, n_levels=4)
                    for i in range(4):
                        if i in cached and not np.array_equal(cached[i], computed[i]):
                            raise ValueError("Native hierarchy differs from asset: " + key)
                    selected["coarse_edges"][center] = computed
            prepared = root / "inputs/meshes" / (level + "_" + garment + ".pkl")
            prepared.parent.mkdir(parents=True, exist_ok=True)
            with prepared.open("wb") as stream:
                pickle.dump({key: selected}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            result["meshes"][level+"_"+garment] = dict(source=str(aux/filename), key=key,
                prepared=str(prepared), vertices=len(selected["rest_pos"]), faces=len(selected["faces"]),
                hierarchy_centers=[int(v) for v in selected["center"]], cached_native_levels=4,
                geometry_modified=False)
            print("MESH",level,garment,result["meshes"][level+"_"+garment]["vertices"],flush=True)
        del meshes
    for level, motion, garment in itertools.product(LEVELS, MOTIONS, GARMENTS):
        mesh=result["meshes"][level+"_"+garment]
        case=motion+"_"+garment+"_"+level
        result["cases"][case]=dict(motion=motion, garment=garment, level=level, horizon=MOTIONS[motion],
            mesh_key=mesh["key"], prepared_mesh=mesh["prepared"], vertices=mesh["vertices"], faces=mesh["faces"])
    for identity, backend in (("Pb4U","pb4u"),("Fine15","fine15"),("HOOD","hood"),("ContourCraft","contourcraft")):
        folder="contourcraft_official" if backend=="contourcraft" else "pb4u_official"
        checkpoint=library/"data"/folder/"trained_models"/(backend+".pth")
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        result["models"][identity]=dict(backend=backend, checkpoint=str(checkpoint), arm=None,
            source="published_checkpoint", step=None, available_at_preparation=True)
    for arm, identity in (("Cell-Scale","Scale"),("Cell-ScaleFixed","ScaleFixed")):
        directory=args.training_root/"trains"/arm/"checkpoints"
        paths=sorted(directory.glob("step_*.pth"))
        actual=[int(p.stem.split("_")[-1]) for p in paths]
        if actual!=list(range(2500,25001,2500)):raise ValueError((arm,actual))
        for path, step in zip(paths,actual):
            result["models"][f"{identity}_{step:06d}"]=dict(backend="shared_cell", arm=arm, step=step,
                checkpoint=str(path.resolve()), source="project_fresh_long_course", available_at_preparation=True)
    for step in (27500,30000):
        checkpoint=args.continuation_root/"trains/Cell-ScaleFixed/checkpoints"/f"step_{step:010d}.pth"
        result["models"][f"ScaleFixed_{step:06d}"]=dict(backend="shared_cell", arm="Cell-ScaleFixed", step=step,
            checkpoint=str(checkpoint.resolve()), source="project_approved_continuation", available_at_preparation=checkpoint.is_file(),
            completion_file=str(args.continuation_root/"trains/Cell-ScaleFixed/train_summary.json"))
    result["cases_per_model"]=len(result["cases"])
    result["frames_per_model"]=sum(c["horizon"] for c in result["cases"].values())
    result["planned_models"]=len(result["models"])
    result["planned_cases"]=len(result["cases"])*len(result["models"])
    if (result["cases_per_model"],result["frames_per_model"],result["planned_models"])!=(64,11424,26):
        raise ValueError("Unexpected complete matrix")
    write(root/"manifest.json",result)
    print(json.dumps(dict(state="prepared",models=26,cases=result["planned_cases"],frames=11424*26)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("library-root","source-work","output","training-root","continuation-root","native-preparation"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--control-revision",required=True)
    prepare(parser.parse_args())

if __name__=="__main__":
    main()
