import unittest
from unittest import mock

import trimesh

import mesh_heal


class SurfaceShellTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()