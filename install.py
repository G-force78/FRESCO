import os
import platform

import requests


def build_ebsynth():
  if os.path.exists('src/ebsynth/deps/ebsynth/bin/ebsynth'):
    print('Ebsynth has been built.')
    return

  os_str = platform.system()

  if os_str == 'Windows':
    # ... Existing Windows build logic ...
  elif os_str == 'Linux':
    cmd = 'bash build-linux-cpu+cuda.sh'
    exe_file = 'src/ebsynth/deps/ebsynth/bin/ebsynth'

    # Grant executable permission before running the build script
    os.chmod('src/ebsynth/deps/ebsynth/bin/ebsynth', 0o755)

    os.chdir('src/ebsynth/deps/ebsynth')
    print(cmd)
    os.system(cmd)
    os.chdir('../../../..')

    if os.path.exists(exe_file):
      print('Ebsynth installed successfully.')
    else:
      print('Failed to install Ebsynth.')
  elif os.str == 'Darwin':
    # ... Existing Mac build logic ...
  else:
    print('Cannot recognize OS. Ebsynth installation stopped.')
    return

    os.chdir('src/ebsynth/deps/ebsynth')
    print(cmd)
    os.system(cmd)
    os.chdir('../../../..')
    if os.path.exists(exe_file):
        print('Ebsynth installed successfully.')
    else:
        print('Failed to install Ebsynth.')


def download(url, dir, name=None):
    os.makedirs(dir, exist_ok=True)
    if name is None:
        name = url.split('/')[-1]
    path = os.path.join(dir, name)
    if not os.path.exists(path):
        print(f'Install {name} ...')
        open(path, 'wb').write(requests.get(url).content)
        print('Install successfully.')


def download_gmflow_ckpt():
    url = ('https://huggingface.co/PKUWilliamYang/Rerender/'
           'resolve/main/models/gmflow_sintel-0c07dcb3.pth')
    download(url, 'model')


def download_egnet_ckpt():
    url = ('https://huggingface.co/PKUWilliamYang/Rerender/'
           'resolve/main/models/epoch_resnet.pth')
    download(url, 'model')

def download_hed_ckpt():
    url = ('https://huggingface.co/lllyasviel/Annotators/'
           'resolve/main/ControlNetHED.pth')
    download(url, 'src/ControlNet/annotator/ckpts')

def download_depth_ckpt():
    url = ('https://huggingface.co/lllyasviel/ControlNet/'
           'resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt')
    download(url, 'src/ControlNet/annotator/ckpts')

def download_ebsynth_ckpt():
    os_str = platform.system()
    if os_str == 'Linux':
        url = ('https://huggingface.co/PKUWilliamYang/Rerender/'
               'resolve/main/models/ebsynth')
        download(url, 'src/ebsynth/deps/ebsynth/bin')
    elif os_str == 'Windows':
        url = ('https://huggingface.co/PKUWilliamYang/Rerender/'
               'resolve/main/models/ebsynth.exe')
        download(url, 'src/ebsynth/deps/ebsynth/bin')  
        url = ('https://huggingface.co/PKUWilliamYang/Rerender/'
               'resolve/main/models/ebsynth_cpu.dll')
        download(url, 'src/ebsynth/deps/ebsynth/bin')
        url = ('https://huggingface.co/PKUWilliamYang/Rerender/'
               'resolve/main/models/ebsynth_cpu.exe')
        download(url, 'src/ebsynth/deps/ebsynth/bin')
    else:
        print('No available compiled Ebsynth.')

#build_ebsynth()
download_ebsynth_ckpt()
download_gmflow_ckpt()
download_egnet_ckpt()
download_hed_ckpt()
download_depth_ckpt()
