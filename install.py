import os
import platform

import launch


def check_system_machine():
    system = platform.system()
    machine = platform.machine()
    return (system, machine) in [('Windows', 'AMD64'), ('Linux', 'x86_64')]


def check_python_version(low: int, high: int):
    ver = platform.python_version_tuple()
    if int(ver[0]) == 3 and low <= int(ver[1]) <= high:
        return ver[0] + ver[1]
    return None


def install_pycocotools():
    base = 'https://github.com/Bing-su/dddetailer/releases/download/pycocotools/'
    urls = {
        'Windows': 'pycocotools-2.0.6-cp{ver}-cp{ver}-win_amd64.whl',
        'Linux': 'pycocotools-2.0.6-cp{ver}-cp{ver}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
    }

    python_version = check_python_version(8, 11)
    if not check_system_machine() or not python_version:
        launch.run_pip('install pycocotools', 'sd-webui-ddsd requirement: pycocotools')
        return

    url = urls[platform.system()].format(ver=python_version)
    launch.run_pip(f'install {base + url}', 'sd-webui-ddsd requirement: pycocotools')


def install_groundingdino():
    import torch
    from packaging.version import parse

    # torch_version: '1.13.1' or '2.0.0' or ...
    torch_version = parse(torch.__version__).base_version
    # cuda_version: '117' or '118' or 'None'
    cuda_version = torch.version.cuda.replace('.', '')
    python_version = check_python_version(9, 10)

    if (
        not check_system_machine()
        or (torch_version, cuda_version)
        not in [('1.13.1', '117'), ('2.0.0', '117'), ('2.0.0', '118')]
        or not python_version
    ):
        launch.run_pip('install git+https://github.com/IDEA-Research/GroundingDINO', 'sd-webui-ddsd requirement: groundingdino')
        return

    system = 'win' if platform.system() == 'Windows' else 'linux'
    machine = 'amd64' if platform.machine() == 'AMD64' else 'x86_64'

    url = 'https://github.com/Bing-su/GroundingDINO/releases/download/wheel-0.1.0/groundingdino-0.1.0+torch{torch}.cu{cuda}-cp{py}-cp{py}-{system}_{machine}.whl'
    url = url.format(
        torch=torch_version,
        cuda=cuda_version,
        py=python_version,
        system=system,
        machine=machine,
    )

    launch.run_pip(f'install {url}', 'sd-webui-ddsd requirement: groundingdino')


current_dir = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(current_dir, 'requirements.txt')

with open(req_file) as file:
    for lib in file:
        version = None
        lib = lib.strip()
        lib = 'skimage' if lib == 'scikit-image' else lib
        if '==' in lib:
            lib, version = [x.strip() for x in lib.split('==')]
        if not launch.is_installed(lib):
            if lib == 'pycocotools':
                install_pycocotools()
            elif lib == 'groundingdino':
                install_groundingdino()
            elif lib == 'skimage':
                launch.run_pip(
                    f'install scikit-image',
                    f'sd-webui-ddsd requirement: scikit-image'
                )
            elif lib == 'pillow_lut':
                launch.run_pip(
                    f'install pillow_lut',
                    f'sd-webui-ddsd requirement: pillow_lut'
                )
            else:
                lib = lib if version is None else lib + '==' + version
                launch.run_pip(
                    f'install {lib}',
                    f'sd-webui-ddsd requirement: {lib}'
                )
