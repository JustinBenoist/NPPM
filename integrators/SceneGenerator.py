import os
from typing import Tuple, List
import torch
import mitsuba as mi
import drjit as dr

from integrators.PPM import PPMIntegrator
from integrators.Utils import deepcopy_dict

class SceneGenerator:
    def __init__(self, resolution: int) -> None:
        self.resolution = resolution
        self.BASE_SCENE_DICT = {
            'type' : 'scene',
                'integrator' : {
                    'type' : 'ptracer'
                },
                'emitter' : {
                    'type': 'directional',
                    'direction': [0, 0, -1],
                    'irradiance': {
                        'type': 'rgb',
                        'value': [3, 3, 3]
                    }
                },
                'shape1' : {
                    'type' : 'rectangle',
                    'bsdf' : {
                        'type' : 'diffuse',
                        'reflectance' : {
                            'type' : 'srgb',
                            'color' : [0.7, 0.0, 0.0]
                        }
                    },
                    'to_world' : mi.ScalarTransform4f.translate([0, 0, 0])
                },
                'shape2' : {
                    'type' : 'rectangle',
                    'bsdf' : {
                        'type' : 'bumpmap',
                        'bsdf' : {
                            'type' : 'dielectric',
                            'int_ior' : 1.1,
                            'specular_reflectance': {
                                'type': 'srgb',
                                'color': [1, 1, 1]
                            },
                            'specular_transmittance': {
                                'type': 'srgb',
                                'color': [1, 1, 1]
                            }
                        },
                        'texture' : {
                            'type' : 'bitmap',
                            'filename' : 'textures/bumpmap_0.png'
                        }, 
                        'scale' : 4
                    },
                    'to_world' : mi.ScalarTransform4f.translate([0, 0, 4])    
                },
                'camera' : {
                    'type' : 'perspective',
                    'film' : {
                        'type' : 'hdrfilm',
                        'width' : resolution,
                        'height' : resolution,
                        'banner' : False
                    },
                    'sampler' : {
                        'type' : 'multijitter'
                    },
                    'fov' : 35,
                    'to_world' : mi.ScalarTransform4f.look_at(
                        origin=[0, 0, 3],
                        target=[0, 0, 0],
                        up=[0, 1, 0]
                    )
                }
            }

    def generate_scenes(self, nb_scenes: int, gt_file: str=None) -> Tuple[List[mi.Scene], mi.Scene, torch.Tensor, torch.Tensor]:
        """Generate the nb_scenes scenes dict and 1 testing scene for DPM training

        Args:
            resolution (int): resolution of the render
            nb_scenes (int): number of training scenes
            gt_file (str): path to existing precomputred ground truth. If None, the ground truth of the scenes is computed
            using PPM

        Returns:
            Tuple[List[mi.Scene], mi.Scene, torch.Tensor, torch.Tensor]: training scenes, test scene, training scenes
            ground truth and testing scene ground truth
        """
        scenes_dict = [deepcopy_dict(self.BASE_SCENE_DICT) for _ in range(nb_scenes)]
        for i in range(1, nb_scenes):
            scenes_dict[i]['shape2']['bsdf']['texture'] = {
                        'type' : 'bitmap',
                        'filename' : f'textures/bumpmap_{i}.png'
                    }

        scenes = [mi.load_dict(scene_dict) for scene_dict in scenes_dict]
        scene_test_dict = deepcopy_dict(scenes_dict[0])
        scene_test_dict['shape2']['bsdf']['texture'] = {
                        'type' : 'bitmap',
                        'filename' : f'textures/bumpmap_{nb_scenes}.png'
                    }
        scene_test = mi.load_dict(scene_test_dict)
        # Use cosinus of bumpmaps for very sharp caustics
        for i in range(nb_scenes):
            params = mi.traverse(scenes[i])
            tex = params['shape2.bsdf.nested_texture.data']
            tex = dr.cos(tex)
            params['shape2.bsdf.nested_texture.data'] = tex
            params.update()
        params = mi.traverse(scene_test)
        tex = params['shape2.bsdf.nested_texture.data']
        tex = dr.cos(tex)
        params['shape2.bsdf.nested_texture.data'] = tex
        params.update()
        
        if gt_file is None:
            # Compute PPM references
            ppm = PPMIntegrator(100000, 2000, device="cuda", init_radius=0.05, max_photons=100)
            estimates = []
            for i in range(nb_scenes):
                estimate, _ = ppm.run(scenes[i])
                estimates.append(estimate)
            estimate_test, _ = ppm.run(scene_test)

            estimates_tensor = estimates[0]
            for i in range(1, nb_scenes):
                estimates_tensor = torch.cat((estimates_tensor, estimates[i]))
            torch.save(estimates_tensor, "gt_data.pth")
            torch.save(estimate_test, "gt_test.pth")
        else:
            # Read references
            estimate = torch.load(os.path.join(gt_file, "gt_data.pth"))
            estimate_test = torch.nan_to_num(torch.load(os.path.join(gt_file, "gt_test.pth")), 0)
        return scenes, scene_test, estimate, estimate_test