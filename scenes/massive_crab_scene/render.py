import mitsuba as mi
import pyexr
import drjit as dr

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    with dr.suspend_grad():
        scene = mi.load_file("scene.xml")
        img = mi.render(scene)
        pyexr.write("test.exr", img.numpy())