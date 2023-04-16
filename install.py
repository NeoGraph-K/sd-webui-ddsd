import launch
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(current_dir, 'requirements.txt')

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            lib_fin = lib
            if lib == 'groundingdino':
                lib_fin = 'git+https://github.com/IDEA-Research/GroundingDINO'
            launch.run_pip(
                f'install {lib_fin}',
                f'sd-webui-ddsd requirement: {lib_fin}'
            )