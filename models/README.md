# Robot Models

This directory holds MuJoCo MJCF/URDF model files for the robots
supported by RoboClaw.

## Downloading Models

Models are sourced from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
maintained by Google DeepMind.

```bash
bash scripts/download_models.sh
```

This downloads:
- **Unitree H1** → `models/h1_unitree/`
- **Agility Digit v3** → `models/digit_v3/`

## Manual Setup

If you prefer to use your own models, place them under a subdirectory
here and update the `sim.model_path` in your robot config:

```yaml
# configs/robots/h1_unitree.yaml
sim:
  enabled: true
  model_path: "models/h1_unitree/scene.xml"
```

## Model Format

MuJoCo supports two formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| MJCF | `.xml` | MuJoCo's native format. Preferred. |
| URDF | `.urdf` | ROS standard. Supported via MuJoCo's URDF parser. |

## Testing Without Downloaded Models

The unit tests use an embedded minimal humanoid MJCF model defined in
`tests/conftest.py`. This means you can run all MuJoCo tests without
downloading any external models:

```bash
pip install -e ".[sim]"
pytest tests/ -v
```

## Adding a New Robot

1. Add the model files under `models/<robot_name>/`
2. Create a robot config at `configs/robots/<robot_name>.yaml`
3. Set `sim.model_path` to point to the model's scene XML
4. Run the tests to verify: `pytest tests/ -v`
