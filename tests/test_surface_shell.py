import unittest
from unittest import mock
from pathlib import Path
import tempfile

import trimesh

import mesh_heal


class SurfaceShellTests(unittest.TestCase):
    def test_derive_surface_shell_output_path_uses_buffer_suffix(self):
        input_path = Path("sample_mesh.obj")

        output_path = mesh_heal.derive_surface_shell_output_path(input_path, 2.5)

        self.assertEqual(output_path.name, "sample_mesh_buffer_2.5.dxf")

    def test_open_surface_shell_stitches_boundary_loop(self):
        plane = trimesh.Trimesh(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            faces=[[0, 1, 2], [0, 2, 3]],
            process=False,
        )

        shell, report = mesh_heal.build_surface_shell(plane, 0.1)
        stats = mesh_heal.mesh_report(shell)

        self.assertTrue(stats.watertight)
        self.assertEqual(stats.boundary_edges, 0)
        self.assertEqual(report["boundary_loops_detected"], 1)
        self.assertEqual(report["boundary_loops_stitched"], 1)
        self.assertGreater(len(shell.faces), len(plane.faces))
        self.assertEqual(len(shell.split()), 1)

    def test_closed_mesh_shell_remains_watertight(self):
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

        shell, report = mesh_heal.build_surface_shell(box, 0.1)
        stats = mesh_heal.mesh_report(shell)

        self.assertTrue(stats.watertight)
        self.assertEqual(stats.boundary_edges, 0)
        self.assertEqual(report["boundary_loops_detected"], 0)
        self.assertEqual(report["boundary_loops_stitched"], 0)
        self.assertEqual(len(shell.split()), 2)
        self.assertGreater(len(shell.faces), len(box.faces))

    def test_execute_heal_strategy_dispatches_surface_shell(self):
        plane = trimesh.Trimesh(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            faces=[[0, 1, 2], [0, 2, 3]],
            process=False,
        )
        sentinel_mesh = plane.copy()

        with mock.patch("mesh_heal.build_surface_shell", return_value=(sentinel_mesh, {"mode": "surface-shell"})) as shell_builder:
            with mock.patch("mesh_heal.build_distance_hull") as distance_hull_builder:
                result = mesh_heal.execute_heal_strategy_on_mesh(
                    plane,
                    distance_model="surface-shell",
                    distance_offset=0.1,
                )

        shell_builder.assert_called_once()
        distance_hull_builder.assert_not_called()
        self.assertIs(result["mesh"], sentinel_mesh)
        self.assertEqual(result["report"]["distance_model"]["mode"], "surface-shell")

    def test_run_surface_shell_batch_pipeline_uses_default_buffer_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_a = root / "first.obj"
            input_b = root / "second.stl"
            input_a.write_text("placeholder", encoding="utf-8")
            input_b.write_text("placeholder", encoding="utf-8")

            with mock.patch("mesh_heal.run_heal_pipeline") as pipeline:
                pipeline.side_effect = lambda **kwargs: {
                    "input": str(kwargs["input_path"]),
                    "output": str(kwargs["output_path"]),
                    "distance_model": {"mode": kwargs["distance_model"]},
                }
                result = mesh_heal.run_surface_shell_batch_pipeline(
                    items=[
                        {"input_path": input_a, "distance_offset": 1.25},
                        {"input_path": input_b, "distance_offset": 3.0},
                    ],
                    output_directory=root / "buffers",
                )

        self.assertEqual(result["mode"], "surface-shell-batch")
        self.assertEqual(result["item_count"], 2)
        self.assertEqual(pipeline.call_count, 2)
        self.assertEqual(Path(result["results"][0]["output"]).name, "first_buffer_1.25.dxf")
        self.assertEqual(Path(result["results"][1]["output"]).name, "second_buffer_3.dxf")


if __name__ == "__main__":
    unittest.main()