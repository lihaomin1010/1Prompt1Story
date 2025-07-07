## How To Use

```bash
# Clone this repository
$ git clone https://github.com/byliutao/1Prompt1Story

# Go into the repository
$ cd 1Prompt1Story

### Install dependencies ###
$ conda create --name 1p1s python=3.10
$ conda activate 1p1s
# choose the right cuda version of your device
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
$ pip install transformers==4.46.3  # or: conda install conda-forge::transformers 
$ conda install -c conda-forge diffusers
$ pip install opencv-python scipy gradio==4.44.1 sympy==1.13.1
### Install dependencies ENDs ###

# Run infer code
python main.py --model_path "stabilityai/stable-diffusion-xl-base-1.0" \
--control_model_path "diffusers/controlnet-depth-sdxl-1.0" \
--id_prompt "a realistic photo of a Bengal tiger in the wild, high resolution, natural lighting, ultra detailed, wildlife photography style, 50mm lens, National Geographic," \
--frame_prompt_list "Get ready to stand up from the tree" "walk into the forest" "prepare to attack deer" \
--control_image_list "resource/controlnet/depth00.png" "resource/controlnet/depth01.png" "resource/controlnet/depth02.png" "resource/controlnet/depth03.png"


python main.py --model_path "stabilityai/stable-diffusion-xl-base-1.0" \
--control_model_path "diffusers/controlnet-canny-sdxl-1.0" \
--id_prompt "A photo of Egret" \
--frame_prompt_list "drink water" "stand on the ground" "walk ast the shore" \
--control_image_list "resource/controlnet/canny10.png" "resource/controlnet/canny11.png" "resource/controlnet/canny12.png" "resource/controlnet/canny13.png"
```


