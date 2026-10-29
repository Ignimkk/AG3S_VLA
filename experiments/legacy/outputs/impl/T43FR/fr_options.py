"""T43 FR — the finger_cover option set evaluated (keys -> build_constraint_robot_model(finger_cover=...))."""
MM = 1e-3
OPTIONS = {
    "fc25": {"overflow": 2.5 * MM},                      # current default (FC)
    "ov3": {"overflow": 3.0 * MM}, "ov4": {"overflow": 4.0 * MM},
    "ov5": {"overflow": 5.0 * MM}, "ov6": {"overflow": 6.0 * MM},
    "k8": {"overflow": 2.5 * MM, "max_spheres": 8}, "k12": {"overflow": 2.5 * MM, "max_spheres": 12},
    "k16": {"overflow": 2.5 * MM, "max_spheres": 16}, "k24": {"overflow": 2.5 * MM, "max_spheres": 24},
    "gap05": {"overflow": 2.5 * MM, "gap": 0.5 * MM}, "gap10": {"overflow": 2.5 * MM, "gap": 1.0 * MM},
    "ilp25": {"overflow": 2.5 * MM, "solver": "ilp"}, "ilp3": {"overflow": 3.0 * MM, "solver": "ilp"},
    "ilp4": {"overflow": 4.0 * MM, "solver": "ilp"}, "ilp5": {"overflow": 5.0 * MM, "solver": "ilp"},
    "ilp6": {"overflow": 6.0 * MM, "solver": "ilp"},
    "ilp25gap05": {"overflow": 2.5 * MM, "gap": 0.5 * MM, "solver": "ilp"},
    "ilp25gap10": {"overflow": 2.5 * MM, "gap": 1.0 * MM, "solver": "ilp"},
}
