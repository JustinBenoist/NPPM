import os
import io
import random
import argparse
from typing import Tuple
from math import sqrt

class SceneGenerator:

    def __init__(self, outdir: str, n_scenes: int=50, nb_photonpaths: int=4000000,
                 resolution: int=256, generate_bumpmaps: bool=True, render: bool=False, glossy: bool=True):
        """Initializes scene generator

        Args:
            outdir (str): path where the output scenes while be stored
            photonmaps_outdir (str): path where the resulting photon maps will be written. If pbrt is installed on Windows make sure to use Windows syntax when writing the path.
            n_scenes (int, optional): number of scenes to generate. Defaults to 50.
            nb_photons (int, optional): number of photon paths for the photon maps. Defaults to 1000000.
            resolution (int, optional): resolution of the scenes. Defaults to 256.
            generate_bumpmaps (bool, optional): specifies if we create new bump maps, if yes the bumpmaps are written in [outdir]/bumpmaps. Defaults to True.
            render (bool, optional): indicates if we must generate scenes for rendering (defaults, photon map generation). Defaults to False.
            glossy (bool, optional): indicates if we must generate glossy surfaces. Defaults to True.
        """
        if generate_bumpmaps:
            from BumpMapGenerator import generate_bumpmap
        self.outdir = outdir
        self.n_scenes = n_scenes
        self.nb_photons = nb_photonpaths
        self.resolution = resolution
        self.render = render
        self.glossy = glossy
        self.integrator = "ptracer"
        self.maxdepth = 5
        self.pixelsamples = 2000
        try:
            os.makedirs(self.outdir)
        except OSError:
            pass
        # Generate bump maps
        if generate_bumpmaps:
            generate_bumpmap(os.path.join(outdir, "bumpmaps/"), n=5, res=2048, octave=3)
        try:
            os.makedirs(os.path.join(self.outdir, "models/"))
        except OSError:
            pass
        # Count number of models available
        self.nb_models = -1
        for path in os.listdir(os.path.join(outdir, "models/")):
            if os.path.isfile(os.path.join(os.path.join(outdir, "models/"), path)) and path[:9] == "new_model":
                self.nb_models += 1
            

    def __generate_color(self) -> Tuple[float]:
        r, g, b = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
        norm = sqrt(r**2 + g**2 + b**2)
        r, g, b = r/norm, g/norm, b/norm
        return r, g, b
    
    def __generate_k(self) -> Tuple[float]:
        r, g, b = random.uniform(1, 3), random.uniform(1, 3), random.uniform(1, 3)
        return r, g, b

    def generate_shearing(self) -> str:
        axis = random.randint(0, 2)
        shear_1 = random.uniform(0.5, 1.3)
        shear_2 = random.uniform(0.5, 1.3)
        if axis == 0:
            return f'Transform [ 1 {shear_1} {shear_2} 0  0 1 0 0  0 0 1 0  0 0 0 1 ]\n'
        elif axis == 1:
            return f'Transform [ 1 0 0 0  {shear_1} 1 {shear_2} 0  0 0 1 0  0 0 0 1 ]\n'
        if axis == 2:
            return f'Transform [ 1 0 0 0  0 1 0 0  {shear_1} {shear_2} 1 0  0 0 0 1 ]\n'

    def __write_render_params(self, file:io.TextIOWrapper, index_scene: int):
        # Randomize camera position and direction
        rotation_x, rotation_y, rotation_z = random.uniform(0, 1), random.uniform(-1, 1), random.uniform(-1, 1)
        angle = random.uniform(0, 15)
        fov = random.uniform(20, 40)
        file.write(f'<scene version="3.0.0">\n')
        file.write(f'    <integrator type="{self.integrator}">\n')
        file.write(f'       <integer name="max_depth" value="{self.maxdepth}"/>\n')
        file.write(f'    </integrator>\n')
        file.write(f'   <sensor type="perspective">\n')
        file.write(f'       <float name="fov" value="{fov}" />\n')
        file.write(f'       <transform name="to_world">\n')
        # file.write(f'           <matrix value="1 -0 -0 -0 -0 1 -0 -0 -0 -0 -1 -0 -0 -1 -6.8 1" />\n')
        # file.write(f'           <translate x="0" y="1" z="-4.8"/>\n')
        # file.write(f'            <rotate x="{rotation_x}" y="{rotation_y}" z="{rotation_z}" angle="{angle}"/>\n')
        # file.write(f'            <rotate x="{rotation_x}" angle="{angle}"/>\n')
        # angle = random.uniform(0, 15)
        # file.write(f'            <rotate z="{rotation_z}" angle="{angle}"/>\n')
        # file.write(f'           <rotate y="1" angle="180"/>\n')
        o_x, o_y, o_z = 3 * random.uniform(-0.4, 0.4), 1 + 3 * random.uniform(-0.3, 0.3), 3.8 + 3 * random.uniform(-0.4, 0.3)
        target_x, target_y, target_z = 0 + random.uniform(-0.2, 0.2), 1 + random.uniform(-0.2, 0.2), 0 + random.uniform(-0.2, 0.2)
        up_x, up_y, up_z = random.uniform(-0.2, 0.2), random.uniform(0.6, 1.0), random.uniform(-0.2, 0.2)
        norm = sqrt(up_x**2 + up_y**2 + up_z**2)
        up_x, up_y, up_z = up_x / norm, up_y / norm, up_z / norm
        file.write(f'           <lookat origin="{o_x}, {o_y}, {o_z}" target="{target_x}, {target_y}, {target_z}" up="{up_x}, {up_y}, {up_z}"/>\n')
        file.write(f'       </transform>\n')
        file.write(f'       <sampler type="independent">\n')
        file.write(f'           <integer name="sample_count" value="{self.pixelsamples}" />\n')
        file.write(f'       </sampler>\n')
        file.write(f'       <film type="hdrfilm">\n')
        file.write(f'           <integer name="width" value="{self.resolution}" />\n')
        file.write(f'           <integer name="height" value="{self.resolution}" />\n')
        file.write(f'           <string name="file_format" value="openexr" />\n')
        file.write(f'           <string name="pixel_format" value="rgb" />\n')
        file.write(f'           <rfilter type="tent" />\n')
        file.write(f'       </film>\n')
        file.write(f'   </sensor>\n')
        file.write('\n')
        file.write('\n')
    
    def __write_wall_material(self, file:io.TextIOWrapper, id: str):
        is_wall_colored = random.randint(0, 3)
        is_not_glossy = random.randint(0, 5)
        # if (not is_not_glossy) and self.glossy and (not is_floor):
        #     file.write('    "string type" [ "conductor" ]\n')
        #     file.write('    "float vroughness" [ 0.25 ]\n')
        #     file.write('    "float uroughness" [ 0.25 ]\n')
        #     file.write('    "bool remaproughness" [ false ]\n')
        #     r, g, b = self.__generate_k()
        #     file.write(f'    "rgb k" [ {r} {g} {b} ]\n')
        #     r, g, b = self.__generate_k()
        #     file.write(f'    "rgb eta" [ {r} {g} {b} ]\n')
        # else:
        file.write(f'    <bsdf type="twosided" id="{id}" >\n')
        file.write(f'        <bsdf type="diffuse" >\n')
        if is_wall_colored:
            r, g, b = self.__generate_color()
            file.write(f'            <rgb name="reflectance" value="{r}, {g}, {b}"/>\n')
        else:
            file.write(f'            <rgb name="reflectance" value="0.725, 0.71, 0.68"/>\n')
        file.write(f'        </bsdf>\n')
        file.write(f'    </bsdf>\n')

    def __write_materials(self, file:io.TextIOWrapper, n_diffuse: int, n_specular: int):
        # file.write('WorldBegin\n')
        file.write('\n')
        # Count number of bumpmaps
        nb_bumpmap = 0
        for path in os.listdir(os.path.join(self.outdir, "bumpmaps/")):
            if os.path.isfile(os.path.join(os.path.join(self.outdir, "bumpmaps/"), path)):
                nb_bumpmap += 1
        # Write bumpmaps textures for each specular objects
        for i in range(n_specular):
            has_bumpmap = random.uniform(0.0, 1.0) > 0.5
            index_bumpmap = random.randint(0, nb_bumpmap - 1)
            if has_bumpmap:
                file.write(f'    <bsdf type="bumpmap" id="specular_{i}">\n')
                file.write(f'        <texture name="arbitrary" type="bitmap">\n')
                file.write(f'            <boolean name="raw" value="true"/>\n')
                file.write(f'            <string name="filename" value="bumpmaps/bumpmap_{index_bumpmap}.png"/>\n')
                file.write(f'        </texture>\n')
            file.write(f'        <bsdf type="dielectric" {f'id="specular_{i}"' if not has_bumpmap else ''}>\n')
            file.write(f'           <string name="int_ior" value="benzene"/>\n')
            file.write(f'           <string name="ext_ior" value="air"/>\n')
            file.write(f'       </bsdf>\n')
            if has_bumpmap: 
                file.write(f'    </bsdf>\n')

        # Write bumpmaps textures for each diffuse objects
        for i in range(n_diffuse):
            file.write(f'    <bsdf type="diffuse" id="diffuse_{i}">\n')
            r, g, b = self.__generate_color()
            file.write(f'        <rgb name="reflectance" value="{r}, {g}, {b}"/>\n')
            file.write(f'    </bsdf>\n')
        # Write walls materials
        self.__write_wall_material(file, 'LeftWall')
        self.__write_wall_material(file, 'RightWall')
        self.__write_wall_material(file, 'Floor')
        self.__write_wall_material(file, 'Ceiling')
        self.__write_wall_material(file, 'BackWall')
        # Write light material
        file.write('    <bsdf type="twosided" id="Light" >\n')
        file.write('        <bsdf type="diffuse" >\n')
        file.write('            <rgb name="reflectance" value="0, 0, 0"/>\n')
        file.write('        </bsdf>\n')
        file.write('    </bsdf>\n')
        # Write luminaire material
        file.write('    <bsdf type="twosided" id="luminaire" >\n')
        file.write('        <bsdf type="conductor" >\n')
        file.write('            <string name="material" value="none"/>\n')
        file.write('        </bsdf>\n')
        file.write('    </bsdf>\n')
    
    def __write_cornell_box(self, file:io.TextIOWrapper):
        file.write(f' 	<shape type="rectangle" >\n')
        file.write(f'         <transform name="to_world" >\n')
        file.write(f'             <matrix value="-0 1 0 0 0 0 2 0 1 0 0 0 0 0 0 1"/>\n')
        file.write(f'         </transform>\n')
        file.write(f'         <ref id="Floor" />\n')
        file.write(f'     </shape>\n')
        file.write(f'     <shape type="rectangle" >\n')
        file.write(f'         <transform name="to_world" >\n')
        file.write(f'             <matrix value="-1 0 -0 0 0 0 -2 2 0 -1 -0 0 0 0 0 1"/>\n')
        file.write(f'         </transform>\n')
        file.write(f'         <ref id="Ceiling" />\n')
        file.write(f'     </shape>\n')
        file.write(f'     <shape type="rectangle" >\n')
        file.write(f'         <transform name="to_world" >\n')
        file.write(f'             <matrix value="0 1 0 0 1 0 -0 1 -0 0 -2 -1 0 0 0 1"/>\n')
        file.write(f'         </transform>\n')
        file.write(f'         <ref id="BackWall" />\n')
        file.write(f'     </shape>\n')
        file.write(f'     <shape type="rectangle" >\n')
        file.write(f'         <transform name="to_world" >\n')
        file.write(f'             <matrix value="0 -0 2 1 1 0 -0 1 0 1 0 0 0 0 0 1"/>\n')
        file.write(f'         </transform>\n')
        file.write(f'         <ref id="RightWall" />\n')
        file.write(f'     </shape>\n')
        file.write(f'     <shape type="rectangle" >\n')
        file.write(f'         <transform name="to_world" >\n')
        file.write(f'             <matrix value="-0 0 -2 -1 1 0 -0 1 0 -1 -0 0 0 0 0 1"/>\n')
        file.write(f'         </transform>\n')
        file.write(f'         <ref id="LeftWall" />\n')
        file.write(f'     </shape>\n')
    
    def __write_random_shape(self, file:io.TextIOWrapper, index: int, diffuse: bool=True):
        if diffuse:
            index_model = random.randint(0, self.nb_models)
        else:
            index_model = random.randint(0, 2)
        # Randomize scaling, shearing, position and rotation of the model
        translation_x = random.uniform(-0.8, 0.8)
        translation_y = random.uniform(0.5, 1.4) if not diffuse else random.uniform(0.1, 0.9)
        translation_z = random.uniform(-0.8, 0.8)
        rotation_x, rotation_y, rotation_z = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
        angle = random.uniform(0, 360)
        if index_model == 2:
            angle = random.uniform(0, 20)
        if not diffuse and index == 0:
            translation_x = random.uniform(-0.2, 0.2)
            translation_y = random.uniform(1.5, 1.7)
            translation_z = random.uniform(-0.5, 0.5)
            rotation_x, rotation_y, rotation_z = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
            angle = random.uniform(0, 15)
            scale_x, scale_y, scale_z = random.uniform(0.5, 0.7), random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
        if diffuse:
            scale_x, scale_y, scale_z = random.uniform(0.4, 0.7), random.uniform(0.4, 0.7), random.uniform(0.4, 0.7)
        else:
            scale_x, scale_y, scale_z = random.uniform(0.5, 0.7), random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
        intricate_model = random.uniform(0.0, 1.0) > 0.4
        model_name = f"model_{index_model}.obj"
        if intricate_model and not diffuse:
            model_name = f"new_model_{index_model}.obj"
        file.write(f'    <shape type="obj">\n')
        file.write(f'        <string name="filename" value="models/{model_name}" />\n')
        file.write(f'        <transform name="to_world" >\n')
        file.write(f'            <scale x="{scale_x}" y="{scale_y}" z="{scale_z}"/>\n')
        file.write(f'            <rotate x="{rotation_x}" y="{rotation_y}" z="{rotation_z}" angle="{angle}"/>\n')
        file.write(f'            <translate x="{translation_x}" y="{translation_y}" z="{translation_z}"/>\n')
        file.write(f'        </transform>\n')
        if not diffuse:
            file.write(f'        <ref id="specular_{index}" />\n')
        else:
            file.write(f'        <ref id="diffuse_{index}" />\n')
        file.write(f'    </shape>\n')
    
    def __write_light(self, file:io.TextIOWrapper, n_lights: int):
        for i in range(n_lights):
            if i == 0:
                angle = random.uniform(-10, 10)
            else:
                angle = random.uniform(0, 360)
            translation_x = random.uniform(-0.7, 0.7)
            translation_y = random.uniform(0.3, 1.8)
            translation_z = random.uniform(-0.7, 0.7)
            rotation_x, rotation_y, rotation_z = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
            scale_x, scale_y, scale_z = random.uniform(0.001, 0.005), random.uniform(0.001, 0.005), random.uniform(0.001, 0.005)
            ratio = 1 / (scale_x * scale_z)
            power_r = 1.7
            power_g = 1.7
            power_b = 1.7
            file.write(f'	<shape type="rectangle" >\n')
            file.write(f'        <transform name="to_world" >\n')
            file.write(f'            <scale x="{scale_x}" y="{scale_y}" z="{scale_z}"/>\n')
            if i == 0:
                angle = random.uniform(-30, 30)
                file.write(f'            <rotate x="1" y="0" z="0" angle="90"/>\n')
                file.write(f'            <translate x="0" y="1.96" z="0"/>\n')
                file.write(f'            <rotate x="{rotation_x}" y="{rotation_y}" z="{rotation_z}" angle="{angle}"/>\n')
            else:
                angle = random.uniform(0, 360)
                file.write(f'            <rotate x="{rotation_x}" y="{rotation_y}" z="{rotation_z}" angle="{angle}"/>\n')
                file.write(f'            <translate x="{translation_x}" y="{translation_y}" z="{translation_z}"/>\n')
            file.write(f'        </transform>\n')
            file.write(f'        <ref id="Light" />\n')
            file.write(f'        <emitter type="area" >\n')
            file.write(f'            <rgb name="radiance" value="{power_r * ratio / n_lights}, {power_g * ratio / n_lights}, {power_b * ratio / n_lights}"/>\n')
            file.write(f'        </emitter>\n')
            file.write(f'    </shape>\n')
            # Write luminaire
            if i == 0 and random.uniform(0, 1) > 0.5:
                luminaire_scale = random.uniform(0.1, 0.5)
                file.write(f'    <shape type="obj">\n')
                file.write(f'        <string name="filename" value="models/luminaire.obj" />\n')
                file.write(f'        <transform name="to_world" >\n')
                file.write(f'            <scale x="{scale_x*200}" y="{scale_y*200}" z="{scale_z*200}"/>\n')
                if i == 0:
                    angle = random.uniform(-30, 30)
                    file.write(f'            <rotate x="1" y="0" z="0" angle="90"/>\n')
                    file.write(f'            <translate x="0" y="1.96" z="0"/>\n')
                    file.write(f'            <rotate x="{rotation_x}" y="{rotation_y}" z="{rotation_z}" angle="{angle}"/>\n')
                else:
                    angle = random.uniform(0, 360)
                    file.write(f'            <rotate x="{rotation_x}" y="{rotation_y}" z="{rotation_z}" angle="{angle}"/>\n')
                    file.write(f'            <translate x="{translation_x}" y="{translation_y}" z="{translation_z}"/>\n')
                file.write(f'        </transform>\n')
                file.write(f'        <ref id="luminaire" />\n')
                file.write(f'    </shape>\n')

    def generate(self):
        for i in range(self.n_scenes):
            n_diffuse = random.randint(1, 2)
            n_specular = random.randint(2, 3)
            n_lights = random.randint(1, 3)
            with open(os.path.join(self.outdir, f"scene_{i}.xml"), 'w') as f:
                self.__write_render_params(f, i)
                self.__write_materials(f, n_diffuse, n_specular)
                self.__write_cornell_box(f)
                for i in range(n_diffuse):
                    self.__write_random_shape(f, i, diffuse=True)
                for i in range(n_specular):
                    self.__write_random_shape(f, i, diffuse=False)
                self.__write_light(f, n_lights=n_lights)
                f.write('</scene>')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, help='path where the output .xml scenes while be stored')
    parser.add_argument('--n_scenes', type=int, default=5, help='number of scenes to generate')
    parser.add_argument('--nb_photons', type=int, default=4000000, help='number of valid photons paths for the photon maps (4M is recommended)')
    parser.add_argument('--resolution', type=int, default=256, help='resolution of the scenes')
    parser.add_argument('--generate_bumpmaps', action="store_true", default=False, help="specifies if we create new bump maps, if yes the bumpmaps are written in [outdir]/bumpmaps")
    parser.add_argument('--render', action="store_true", default=False, help="indicates if we must generate scenes for rendering (defaults, photon map generation)")
    parser.add_argument('--glossy', action="store_true", default=False, help="indicates if we must generate scenes with glossy reflection (default True)")
    opt = parser.parse_args()
    scene_generator = SceneGenerator(opt.outfile,
                                     n_scenes=opt.n_scenes,
                                     nb_photonpaths=opt.nb_photons,
                                     resolution=opt.resolution,
                                     generate_bumpmaps=opt.generate_bumpmaps,
                                     render=opt.render,
                                     glossy=opt.glossy)
    scene_generator.generate()