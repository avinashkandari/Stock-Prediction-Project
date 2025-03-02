import unittest
from src.model import build_lstm_model

class TestModel(unittest.TestCase):
    def test_model_creation(self):
        model = build_lstm_model(input_shape=(100, 1))
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()