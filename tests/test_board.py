import unittest
from board import Board
from state import Pawn, Zone

class TestBoard(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_basic_forward_movement(self):
        pawn = Pawn(owner=0, zone=Zone.MAIN, index=5)
        path = self.board.get_path(pawn, steps=3)
        self.assertEqual(path, [6, 7, 8])

    def test_forward_wrap(self):
        pawn = Pawn(owner=0, zone=Zone.MAIN, index=52)
        path = self.board.get_path(pawn, steps=3)
        self.assertEqual(path, [53, 0, 1])

    def test_pass_home_stretch(self):
        pawn = Pawn(owner=0, zone=Zone.MAIN, index=15)
        path = self.board.get_path(pawn, steps=3)
        self.assertEqual(path, [16, 17, 18])

    def test_enter_home_stretch(self):
        pawn = Pawn(owner=0, zone=Zone.MAIN, index=15)
        path = self.board.get_path(pawn, steps=3, enter_home=True)
        self.assertEqual(path, [16, 54, 55])

    def test_overshoot_raises(self):
        pawn = Pawn(owner=0, zone=Zone.HOME, index=57)
        with self.assertRaises(ValueError):
            self.board.get_path(pawn, steps=1)

    def test_home_stretch(self):
        pawn = Pawn(owner=0, zone=Zone.HOME, index=55)
        path = self.board.get_path(pawn, steps=2)
        self.assertEqual(path, [56, 57])

    def test_home_stretch_alt(self):
        pawn = Pawn(owner=1, zone=Zone.HOME, index=58)
        path = self.board.get_path(pawn, steps=3)
        self.assertEqual(path, [59, 60, 61])

    def test_backward_movement(self):
        pawn = Pawn(owner=0, zone=Zone.MAIN, index=5)
        path = self.board.get_path_backward(pawn, steps=3)
        self.assertEqual(path, [4, 3, 2])

    def test_backward_wrap(self):
        pawn = Pawn(owner=0, zone=Zone.MAIN, index=1)
        path = self.board.get_path_backward(pawn, steps=3)
        self.assertEqual(path, [0, 53, 52])

    def test_backward_home_raises(self):
        pawn = Pawn(owner=0, zone=Zone.HOME, index=55)
        with self.assertRaises(ValueError):
            self.board.get_path_backward(pawn, steps=2)


if __name__ == '__main__':
    unittest.main()