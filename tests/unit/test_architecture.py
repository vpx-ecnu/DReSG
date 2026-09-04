"""Stable architecture boundaries for the public repository."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "dresg"
TOOLS_ROOT = REPO_ROOT / "tools"


def source_files(*parts: str) -> list[Path]:
    root = SRC_ROOT.joinpath(*parts)
    return list(root.rglob("*.py")) if root.exists() else []


def imports_from(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    return imports


def import_violations(
    paths: list[Path],
    forbidden: tuple[str, ...],
) -> list[str]:
    return [
        f"{path.relative_to(SRC_ROOT)} imports {module}"
        for path in paths
        for module in imports_from(path)
        if module.startswith(forbidden)
    ]


def test_source_has_no_untyped_object_annotations() -> None:
    failures = [
        str(path.relative_to(SRC_ROOT))
        for path in source_files()
        if ": object" in path.read_text(encoding="utf-8")
    ]
    assert not failures, "\n".join(failures)


def test_source_has_no_path_mutation_or_wildcard_imports() -> None:
    failures: list[str] = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        if "sys.path.insert" in text:
            failures.append(f"{path.relative_to(SRC_ROOT)} mutates sys.path")
        tree = ast.parse(text)
        if any(
            isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(tree)
        ):
            failures.append(f"{path.relative_to(SRC_ROOT)} uses a wildcard import")
    assert not failures, "\n".join(failures)


def test_runtime_code_has_no_try_finally_blocks() -> None:
    failures = [
        f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
        for path in [*source_files(), *TOOLS_ROOT.glob("*.py")]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Try) and node.finalbody
    ]
    assert not failures, "\n".join(failures)


def test_evaluation_models_are_owned_by_the_evaluation_run() -> None:
    evaluation_root = SRC_ROOT / "evaluation"
    assert not (evaluation_root / "runtime.py").exists()
    model_code = "\n".join(
        (evaluation_root / name).read_text(encoding="utf-8")
        for name in ("features.py", "consistency.py")
    )
    assert "_CACHE" not in model_code
    assert "clear_evaluation_model_caches" not in model_code
    assert "build_consistency_evaluator" not in model_code
    assert "class ClipEncoder" in model_code
    assert "class DinoEncoder" in model_code
    assert "class ConsistencyEvaluator" in model_code


def test_training_does_not_depend_on_entrypoints_or_tools() -> None:
    failures = import_violations(
        source_files("training"),
        ("dresg.apps", "dresg.tools", "tools"),
    )
    assert not failures, "\n".join(failures)


def test_model_domains_do_not_depend_on_training_or_each_other() -> None:
    diffusion_failures = import_violations(
        source_files("models", "diffusion"),
        (
            "dresg.training",
            "dresg.apps",
            "dresg.tools",
            "dresg.models.gs",
            "dresg.inference",
        ),
    )
    gs_failures = import_violations(
        source_files("models", "gs"),
        (
            "dresg.training",
            "dresg.apps",
            "dresg.tools",
            "dresg.models.diffusion",
            "dresg.inference",
        ),
    )
    assert not diffusion_failures + gs_failures, "\n".join(diffusion_failures + gs_failures)


def test_apps_only_enter_through_public_training_api() -> None:
    failures = import_violations(source_files("apps"), ("dresg.training.",))
    assert not failures, "\n".join(failures)


def test_public_subsystems_are_explicit() -> None:
    expected = {
        "apps",
        "data",
        "evaluation",
        "inference",
        "models",
        "training",
        "utils",
    }
    packages = {path.name for path in SRC_ROOT.iterdir() if path.is_dir() and not path.name.startswith("__")}
    assert packages == expected


def test_historical_top_level_namespaces_are_absent() -> None:
    forbidden = {
        "core",
        "engine",
        "diffusion",
        "gaussian",
        "guidance",
        "io",
        "losses",
        "optimization",
        "teachers",
        "visualization",
    }
    assert not {name for name in forbidden if (SRC_ROOT / name).exists()}


def test_public_trainer_uses_dresg_vocabulary() -> None:
    paths = [SRC_ROOT / "training" / "trainer.py"]
    forbidden = (
        "ProjectedAD",
        "TexturedConfig",
        "BaseADTrainer",
        "projected_ad_textured_schema",
    )
    failures = [
        f"{path.relative_to(SRC_ROOT)} contains {token}"
        for path in paths
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]
    assert not failures, "\n".join(failures)


def test_train_entry_materializes_one_typed_config() -> None:
    code = (SRC_ROOT / "apps" / "train.py").read_text(encoding="utf-8")
    assert "register_dresg_config" in code
    assert "to_typed_config" in code
    assert "HydraConfig.get().runtime.output_dir" in code
    assert "_config_to_namespace" not in code


def test_hydra_types_stay_at_configuration_boundaries() -> None:
    allowed = {
        Path("apps/train.py"),
        Path("config.py"),
        Path("inference/run.py"),
        Path("utils/view_selection/workflow.py"),
    }
    failures = [
        str(path.relative_to(SRC_ROOT))
        for path in source_files()
        if "omegaconf" in path.read_text(encoding="utf-8")
        and path.relative_to(SRC_ROOT) not in allowed
    ]
    assert not failures, "\n".join(failures)


def test_data_domains_do_not_eagerly_import_colmap_resources() -> None:
    data_root = SRC_ROOT / "data" / "__init__.py"
    root_tree = ast.parse(data_root.read_text(encoding="utf-8"))
    root_imports = {
        node.module or ""
        for node in root_tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert root_imports == set()
    camera_tree = ast.parse(
        (SRC_ROOT / "data" / "cameras.py").read_text(encoding="utf-8")
    )
    eager_modules = {
        node.module or ""
        for node in camera_tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert "dresg.data.colmap" not in eager_modules


def test_removed_single_camera_helpers_are_absent() -> None:
    camera_code = (SRC_ROOT / "data" / "cameras.py").read_text(encoding="utf-8")
    assert "def camera_tensors(" not in camera_code
    assert "def scaled_camera(" not in camera_code
    assert "def load_scaled_camera(" not in camera_code


def test_inference_package_has_no_eager_runtime_imports() -> None:
    for relative_path in (Path("inference/__init__.py"), Path("inference/paths/__init__.py")):
        tree = ast.parse((SRC_ROOT / relative_path).read_text(encoding="utf-8"))
        eager_modules = {
            node.module or ""
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }
        assert eager_modules == {"__future__", "typing"}


def test_inference_first_level_contains_only_public_modules() -> None:
    root = SRC_ROOT / "inference"
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py",
        "run.py",
        "video.py",
        "views.py",
    }


def test_inference_path_modules_match_the_public_layout() -> None:
    root = SRC_ROOT / "inference" / "paths"
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py",
        "codec.py",
        "geometry.py",
        "llff.py",
        "tnt.py",
        "trajectory.py",
    }


def test_diffusion_package_has_no_model_eager_imports() -> None:
    path = SRC_ROOT / "models" / "diffusion" / "__init__.py"
    code = path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    eager_modules = {
        node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert eager_modules == {"__future__", "typing"}


def test_diffusion_first_level_contains_only_the_model_facade() -> None:
    root = SRC_ROOT / "models" / "diffusion"
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py",
        "guidance.py",
    }
    assert {
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    } == {"attention", "latents", "scheduling"}
    expected_modules = {
        "attention": {
            "__init__.py",
            "capture.py",
            "features.py",
            "losses.py",
        },
        "latents": {"__init__.py", "bank.py", "codec.py"},
        "scheduling": {"__init__.py", "scale.py"},
    }
    for directory, modules in expected_modules.items():
        assert {path.name for path in (root / directory).glob("*.py")} == modules


def test_gaussian_scene_package_has_no_model_eager_imports() -> None:
    path = SRC_ROOT / "models" / "gs" / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    eager_modules = {
        node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert eager_modules == {"__future__", "typing"}


def test_gaussian_first_level_contains_only_the_public_scene_module() -> None:
    root = SRC_ROOT / "models" / "gs"
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py",
        "scene.py",
    }
    assert {
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    } == {"fitting", "rendering", "serialization"}
    expected_modules = {
        "fitting": {
            "__init__.py",
            "appearance.py",
            "dino.py",
            "fusion.py",
            "image.py",
        },
        "rendering": {"__init__.py", "rasterization.py"},
        "serialization": {"__init__.py", "ply.py", "sh.py"},
    }
    for directory, modules in expected_modules.items():
        assert {path.name for path in (root / directory).glob("*.py")} == modules


def test_gaussian_rasterization_does_not_repeat_domain_validation() -> None:
    code = (
        SRC_ROOT / "models" / "gs" / "rendering" / "rasterization.py"
    ).read_text(encoding="utf-8")
    assert "_validate_splat_tensor" not in code
    assert "_prepare_camera_batch" not in code
    assert "isfinite" not in code


def test_gaussian_loss_composition_does_not_render_scene_state() -> None:
    path = SRC_ROOT / "models" / "gs" / "fitting" / "appearance.py"
    code = path.read_text(encoding="utf-8")
    imported_modules = imports_from(path)
    assert "GaussianScene" not in code
    assert "CameraView" not in code
    assert "scene.render" not in code
    assert "dresg.models.gs.scene" not in imported_modules
    assert "dresg.data.cameras" not in imported_modules


def test_gaussian_scene_does_not_repeat_ply_validation() -> None:
    scene_code = (SRC_ROOT / "models" / "gs" / "scene.py").read_text(
        encoding="utf-8"
    )
    assert "SplatTensors" in scene_code
    assert "validate_gaussian_splats" not in scene_code


def test_gaussian_appearance_is_optimized_as_direct_rgb() -> None:
    scene_code = (SRC_ROOT / "models" / "gs" / "scene.py").read_text(
        encoding="utf-8"
    )
    config_code = (SRC_ROOT / "config.py").read_text(encoding="utf-8")
    assert "self.appearance_rgb = nn.Parameter" in scene_code
    assert "base_rgb" not in scene_code
    assert "appearance_delta" not in scene_code
    assert "RegularizationConfig" not in config_code
    assert "lambda_appearance_delta_l2" not in config_code


def test_dino_content_state_is_run_owned() -> None:
    content_code = (
        SRC_ROOT / "models" / "gs" / "fitting" / "dino.py"
    ).read_text(encoding="utf-8")
    trainer_code = (SRC_ROOT / "training" / "trainer.py").read_text(
        encoding="utf-8"
    )
    assert "class DinoPatchContentLoss" in content_code
    assert "_DINO_MODEL_CACHE" not in content_code
    assert "_DINO_BASE_CACHE" not in content_code
    assert "self.content_loss" in trainer_code


def test_external_domains_use_public_gaussian_scene_api() -> None:
    failures = import_violations(
        source_files("training")
        + source_files("inference")
        + source_files("utils"),
        ("dresg.models.gs.scene",),
    )
    assert not failures, "\n".join(failures)


def test_training_only_imports_public_gaussian_boundaries() -> None:
    failures = [
        f"{path.relative_to(SRC_ROOT)} imports {module}"
        for path in source_files("training")
        for module in imports_from(path)
        if module.startswith("dresg.models.gs.")
        and module != "dresg.models.gs.fitting"
    ]
    assert not failures, "\n".join(failures)


def test_training_only_imports_the_public_diffusion_boundary() -> None:
    failures = [
        f"{path.relative_to(SRC_ROOT)} imports {module}"
        for path in source_files("training")
        for module in imports_from(path)
        if module.startswith("dresg.models.diffusion.")
    ]
    assert not failures, "\n".join(failures)
    assert all(
        "DiffusionTeacher" not in path.read_text(encoding="utf-8")
        for path in source_files("training")
    )


def test_trainer_run_exposes_only_top_level_training_phases() -> None:
    tree = ast.parse(
        (SRC_ROOT / "training" / "trainer.py").read_text(encoding="utf-8")
    )
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DReSGTrainer"
    )
    run = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = {
        (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
        )
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert {
        "_run_guidance_stages",
        "_run_post_color_stage",
        "finalize",
    } <= calls
    assert "_release_heavy_runtime" not in calls
    assert {
        "GuidanceStage",
        "FeedbackStage",
        "ColorStage",
    }.isdisjoint(calls)


def test_trainer_is_the_only_training_stage_scheduler() -> None:
    trainer_code = (SRC_ROOT / "training" / "trainer.py").read_text(
        encoding="utf-8"
    )
    stages_root = SRC_ROOT / "training" / "stages"
    stage_code = "\n".join(
        path.read_text(encoding="utf-8") for path in stages_root.glob("*.py")
    )
    optimization_root = SRC_ROOT / "training" / "optimization"
    optimization_code = "\n".join(
        path.read_text(encoding="utf-8")
        for path in optimization_root.glob("*.py")
    )
    assert "config.schedule" in trainer_code
    assert "color_transfer.post_enabled" in trainer_code
    assert "color_transfer.post_fit_steps" in trainer_code
    assert "config.schedule" not in stage_code
    assert "active_prefixes" not in stage_code
    assert "final_prefix" not in stage_code
    assert "stage_metrics.json" not in stage_code
    assert "config.schedule" not in optimization_code
    assert "post_enabled" not in optimization_code
    assert "post_fit_steps" not in optimization_code


def test_training_stages_do_not_depend_on_progress_or_write_artifacts() -> None:
    stage_files = source_files("training", "stages")
    failures = import_violations(stage_files, ("dresg.training.output",))
    assert not failures, "\n".join(failures)
    assert all(
        "TrainingProgress" not in path.read_text(encoding="utf-8")
        and "SceneRuntime" not in path.read_text(encoding="utf-8")
        for path in stage_files
    )
    assert all(
        "save_rgb" not in path.read_text(encoding="utf-8")
        and ".mkdir(" not in path.read_text(encoding="utf-8")
        for path in stage_files
    )


def test_training_progress_owns_metric_persistence() -> None:
    trainer_code = (SRC_ROOT / "training" / "trainer.py").read_text(encoding="utf-8")
    output_code = (SRC_ROOT / "training" / "output.py").read_text(encoding="utf-8")

    assert '"summary.json"' in output_code
    assert '"aggregate_metrics.json"' in output_code
    assert "save_json" not in trainer_code
    assert "stage_rows[-1].update" not in trainer_code
    assert "annotate_last_stage_row" not in output_code


def test_training_package_has_cohesive_lifecycle_modules() -> None:
    training_root = SRC_ROOT / "training"
    modules = {path.name for path in training_root.glob("*.py")}
    assert modules == {
        "__init__.py",
        "output.py",
        "trainer.py",
        "validation.py",
    }
    packages = {
        path.name
        for path in training_root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert packages == {"optimization", "stages"}
    assert {path.name for path in (training_root / "stages").glob("*.py")} == {
        "__init__.py",
        "color.py",
        "feedback.py",
        "guidance.py",
    }
    assert {
        path.name
        for path in (training_root / "optimization").glob("*.py")
    } == {
        "__init__.py",
        "gs.py",
        "guidance.py",
    }


def test_models_do_not_own_training_optimizer_operations() -> None:
    forbidden = ("torch.optim", ".zero_grad(", ".backward(", ".step(")
    failures = [
        f"{path.relative_to(SRC_ROOT)} contains {token}"
        for path in source_files("models")
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]
    assert not failures, "\n".join(failures)


def test_trainer_directly_owns_runtime_domains() -> None:
    assert not (SRC_ROOT / "training" / "state.py").exists()

    trainer_path = SRC_ROOT / "training" / "trainer.py"
    trainer_tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    trainer_class = next(
        node for node in trainer_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DReSGTrainer"
    )
    constructor = next(
        node for node in trainer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert [argument.arg for argument in constructor.args.args] == ["self", "config"]
    assert not any(
        isinstance(node, ast.FunctionDef)
        and (
            node.name == "from_config"
            or node.name.startswith("_build_")
            or node.name.startswith("_prepare_")
        )
        for node in trainer_class.body
    )
    forbidden_aliases = {
        "active_prefixes",
        "final_prefix",
        "device",
        "output_dir",
        "stages_dir",
    }
    assert forbidden_aliases.isdisjoint(
        node.name
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef)
    )
    fields = {
        target.attr
        for node in constructor.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert fields == {
        "config",
        "source",
        "cameras",
        "scene",
        "scene_optimizer",
        "base_renders",
        "progress",
        "guidance",
        "content_loss",
    }


def test_resource_domains_use_canonical_free_builders() -> None:
    contracts = (
        (
            SRC_ROOT / "training" / "output.py",
            {"TrainingProgress"},
            {"build_training_progress"},
        ),
        (
            SRC_ROOT / "models" / "diffusion" / "guidance.py",
            {"DiffusionGuidance"},
            {"build_diffusion_guidance"},
        ),
        (
            SRC_ROOT / "models" / "gs" / "scene.py",
            {"GaussianScene"},
            {"build_gaussian_scene"},
        ),
    )
    for path, class_names, builder_names in contracts:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in class_names
        }
        assert set(classes) == class_names
        for class_node in classes.values():
            assert not any(
                isinstance(member, ast.FunctionDef)
                and any(
                    isinstance(decorator, ast.Name)
                    and decorator.id == "classmethod"
                    for decorator in member.decorator_list
                )
                for member in class_node.body
            )
        module_functions = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        assert set(module_functions[-len(builder_names):]) == builder_names


def test_lazy_dino_loss_uses_direct_construction() -> None:
    tree = ast.parse(
        (SRC_ROOT / "models" / "gs" / "fitting" / "dino.py").read_text(
            encoding="utf-8"
        )
    )
    assert not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "build_dino_content_loss"
        for node in tree.body
    )


def test_config_shadow_modules_are_absent() -> None:
    assert not (SRC_ROOT / "training" / "plans.py").exists()
    assert not (SRC_ROOT / "inference" / "paths" / "spec.py").exists()
    assert not (SRC_ROOT / "inference" / "paths" / "config.py").exists()


def test_native_gaussian_ply_requires_no_conversion_cli() -> None:
    tools = SRC_ROOT.parents[1] / "tools"
    assert not (tools / "convert.py").exists()
    assert not (tools / "convert_ply.py").exists()


def test_library_modules_do_not_define_cli_parsers() -> None:
    failures = [
        str(path.relative_to(SRC_ROOT)) for path in source_files() if "argparse" in path.read_text(encoding="utf-8")
    ]
    assert not failures, "\n".join(failures)
