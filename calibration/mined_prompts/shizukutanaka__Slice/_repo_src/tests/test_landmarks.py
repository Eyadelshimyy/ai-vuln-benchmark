import unittest

from slice.landmarks import BONES, JOINTS, PARENT, ROOT, chain


class TestKinematicTree(unittest.TestCase):
    def test_every_bone_joint_has_parent(self):
        for j in JOINTS:
            if j in (ROOT, "spine"):
                continue
            self.assertIn(j, PARENT, j)

    def test_parent_links_are_bones(self):
        bone_set = {tuple(b) for b in BONES}
        for child, parent in PARENT.items():
            self.assertTrue((parent, child) in bone_set
                            or (child, parent) in bone_set,
                            f"{parent}->{child}")

    def test_tree_connected_to_root(self):
        for j in PARENT:
            self.assertEqual(chain(j)[0], ROOT, j)
            self.assertEqual(chain(j)[-1], j)

    def test_no_cycles(self):
        for j in PARENT:
            seen = set()
            node = j
            while node in PARENT:
                self.assertNotIn(node, seen, f"cycle at {j}")
                seen.add(node)
                node = PARENT[node]


if __name__ == "__main__":
    unittest.main()
