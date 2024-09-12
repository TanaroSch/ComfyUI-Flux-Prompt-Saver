from datetime import datetime
import folder_paths
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import numpy as np
import torch
import os
import re



class FluxPromptSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "params": ("SAMPLER_PARAMS",),
                "positive": ("STRING",),
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "save_path": ("STRING",),
                "filename": ("STRING",),
            },
            "optional": {
                "negative": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(self, images, params, positive, model_name, save_path, filename, negative=""):
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            metadata.add_text("parameters", self.create_metadata_string(
                params, positive, negative, model_name))

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.get_save_image_path(filename, save_path, img.width, img.height, params[0]['seed'])
            print(f"Saving image to {filename}")
            filename = filename[1]

            filename = f"{filename}.png"
            base_directory = folder_paths.get_output_directory()
            base_path = os.path.join(base_directory, save_path)
            # Ensure the directory exists
            os.makedirs(base_path, exist_ok=True)

            file_path = os.path.join(base_path, filename)
            img.save(file_path, pnginfo=metadata, optimize=True)
            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "output"
            })

        return {"ui": {"images": results}}

    def create_metadata_string(self, params, positive, negative, model_name):
        p = params[0]
        sampler_scheduler = f"{p['sampler']}_{p['scheduler']}" if p['scheduler'] != 'normal' else p['sampler']
        return f"{positive}\nNegative prompt: {negative}\n" \
            f"Steps: {p['steps']}, Sampler: {sampler_scheduler}, CFG scale: 1.0, Seed: {p['seed']}, " \
            f"Size: {p['width']}x{p['height']}, Model hash: {p.get('model_hash', '')}, " \
            f"Model: {model_name}, Version: ComfyUI"

    def get_save_image_path(self, filename_prefix: str, output_dir: str, image_width=0, image_height=0, seed=0) -> tuple[str, str, int, str, str]:
        def map_filename(filename: str) -> tuple[int, str]:
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return digits, prefix

        def compute_vars(input: str, image_width: int, image_height: int, seed: int) -> str:
            now = datetime.now()

            # Replace width, height, and seed
            input = input.replace("%width%", str(image_width))
            input = input.replace("%height%", str(image_height))
            input = input.replace("%seed%", str(seed))

            # Function to replace date formats
            def date_replace(match):
                date_format = match.group(1)
                return now.strftime(date_format)

            # Replace %date:FORMAT% patterns
            input = re.sub(r'%date:(.*?)%', date_replace, input)

            # Replace legacy date patterns
            date_mappings = {
                'yyyy': '%Y',
                'yy': '%y',
                'MM': '%m',
                'dd': '%d',
                'HH': '%H',
                'mm': '%M',
                'ss': '%S'
            }
            for custom_code, strftime_code in date_mappings.items():
                input = input.replace(custom_code, now.strftime(strftime_code))

            return input

        if "%" in filename_prefix or any(code in filename_prefix for code in ['yyyy', 'yy', 'MM', 'dd', 'HH', 'mm', 'ss']):
            filename_prefix = compute_vars(filename_prefix, image_width, image_height, seed)

        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        full_output_folder = os.path.join(output_dir, subfolder)

        # Convert both paths to absolute paths before comparison
        abs_output_dir = os.path.abspath(output_dir)
        abs_full_output_folder = os.path.abspath(full_output_folder)

        if os.path.commonpath([abs_output_dir, abs_full_output_folder]) != abs_output_dir:
            err = "**** ERROR: Saving image outside the output folder is not allowed." + \
            "\n full_output_folder: " + abs_full_output_folder + \
                "\n         output_dir: " + output_dir + \
                "\n         commonpath: " + os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))
            print(f"Error: {err}")
            raise Exception(err)

        try:
            counter = max(filter(lambda a: os.path.normcase(a[1][:-1]) == os.path.normcase(filename) and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1
        return full_output_folder, filename, counter, subfolder, filename_prefix


NODE_CLASS_MAPPINGS = {
    "FluxPromptSaver": FluxPromptSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxPromptSaver": "Flux Prompt Saver"
}
