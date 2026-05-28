import random

def lerp(a, b, t):
    return a + (b - a) * t

def generate_stratified_spotlights(
    grid_size=10,            # N x N grid
    xz_range=140,
    y_height=150.0,
    tilt_strength=6.0,
    cutoff_base=5.0,
    cutoff_jitter=2.5,
    jitter_ratio=0.3        # how much of the cell can be jittered
):
    lights = []
    cell_size = (2 * xz_range) / grid_size
    half_range = xz_range

    for ix in range(grid_size):
        for iz in range(grid_size):
            # Cell center
            cx = -half_range + (ix + 0.5) * cell_size
            cz = -half_range + (iz + 0.5) * cell_size

            # Jitter within the cell
            jitter = cell_size * jitter_ratio * 0.5
            ox = cx + random.uniform(-jitter, jitter)
            oz = cz + random.uniform(-jitter, jitter)
            oy = y_height

            # Target almost straight down
            tx = ox + random.uniform(-tilt_strength, tilt_strength)
            ty = 0.0
            tz = oz + random.uniform(-tilt_strength, tilt_strength)

            # Cold blue ↔ warm orange interpolation
            cold_blue = (37000, 180000, 740000)
            warm_orange = (740000, 180000, 37000)
            t = random.random()

            intensity = (
                int(lerp(cold_blue[0], warm_orange[0], t)),
                int(lerp(cold_blue[1], warm_orange[1], t)),
                int(lerp(cold_blue[2], warm_orange[2], t)),
            )

            cutoff = cutoff_base + random.uniform(-cutoff_jitter, cutoff_jitter)

            lights.append(f"""
<emitter type="spot">
    <transform name="to_world">
        <lookat origin="{ox:.3f}, {oy:.3f}, {oz:.3f}"
                target="{tx:.3f}, {ty:.3f}, {tz:.3f}"
                up="0, 0, 1"/>
    </transform>
    <rgb name="intensity" value="{intensity[0]} {intensity[1]} {intensity[2]}"/>
    <float name="cutoff_angle" value="{cutoff:.4f}"/>
</emitter>
""".strip())

    return lights


if __name__ == "__main__":
    lights = generate_stratified_spotlights()
    for l in lights:
        print(l)
        print()
