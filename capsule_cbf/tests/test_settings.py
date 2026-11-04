import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from capsule_cbf.settings import Settings, load_settings

CONFIG = Path(__file__).resolve().parents[1]/'config.json'


class SettingsTests(unittest.TestCase):
    def test_shipped_config_equals_defaults(self):
        self.assertEqual(asdict(load_settings(CONFIG)), asdict(Settings()))
        self.assertEqual(set(json.loads(CONFIG.read_text())), set(asdict(Settings())))

    def test_unknown_and_invalid_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cbf.json'
            path.write_text(json.dumps({'controller': 'sensitivity', 'ray_basins_enabled': True}))
            with self.assertRaisesRegex(ValueError, 'ray_basins_enabled'):
                load_settings(path)
        for bad in (dict(controller='ray_basins'), dict(horizon_frames=21), dict(margin=-.01),
                    dict(prediction_grid_cells=1), dict(shear_limit=1.), dict(conformal_quantile=1.),
                    dict(model_jacobian='exact'), dict(warmup_frames=-1)):
            with self.assertRaises(ValueError, msg=str(bad)):
                Settings(**bad)

    def test_unfiltered_controller_is_valid(self):
        self.assertEqual(Settings(controller='none').controller, 'none')


if __name__ == '__main__':
    unittest.main()
