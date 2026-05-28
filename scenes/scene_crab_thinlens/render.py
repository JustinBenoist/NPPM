import mitsuba as mi
import pyexr

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    scene = mi.load_file("scene.xml")
    img = mi.render(scene)
    pyexr.write("test.exr", img.numpy())
    # Water surélevé d'environ 4