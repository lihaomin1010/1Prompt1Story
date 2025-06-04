python main.py --model_path "stabilityai/stable-diffusion-xl-base-1.0" \
--control_model_path "diffusers/controlnet-depth-sdxl-1.0-small" \
--id_prompt "A medieval swordsman in the West" \
--frame_prompt_list "practice his swordsmanship" "fight against monsters" "Smile of victory" \
--control_image_list "resource/controlnet/depth0.png" "resource/controlnet/depth1.png" "resource/controlnet/depth2.png" "resource/controlnet/depth3.png"


python main.py --model_path "stabilityai/stable-diffusion-xl-base-1.0" \
--control_model_path "diffusers/controlnet-depth-sdxl-1.0" \
--id_prompt "a realistic photo of a Bengal tiger in the wild, high resolution, natural lighting, ultra detailed, wildlife photography style, 50mm lens, National Geographic," \
--frame_prompt_list "Get ready to stand up from the tree" "walk into the forest" "prepare to attack deer" \
--control_image_list "resource/controlnet/depth00.png" "resource/controlnet/depth01.png" "resource/controlnet/depth02.png" "resource/controlnet/depth03.png"


python main.py --model_path "stabilityai/stable-diffusion-xl-base-1.0" \
--control_model_path "diffusers/controlnet-canny-sdxl-1.0" \
--id_prompt "A cute sparrow" \
--frame_prompt_list "stand on a branch" "flying in the forest" "catching something on branches" \
--control_image_list "resource/controlnet/canny0.png" "resource/controlnet/canny1.png" "resource/controlnet/canny2.png" "resource/controlnet/canny3.png"

python main.py --model_path "stabilityai/stable-diffusion-xl-base-1.0" \
--control_model_path "diffusers/controlnet-canny-sdxl-1.0" \
--id_prompt "A Black-necked Swan" \
--frame_prompt_list "drink water" "stand on the shore" "flutter wings" \
--control_image_list "resource/controlnet/canny10.png" "resource/controlnet/canny11.png" "resource/controlnet/canny12.png" "resource/controlnet/canny13.png"

python main.py --model_path "stabilityai/stable-diffusion-xl-base-1.0" \
--control_model_path "thibaud/controlnet-openpose-sdxl-1.0" \
--control_image_list "resource/controlnet/openpose0.png" "resource/controlnet/openpose1.png" "resource/controlnet/openpose2.png" "resource/controlnet/openpose3.png"


python main.py --model_path "stabilityai/stable-diffusion-xl-base-1.0" \
--control_model_path "diffusers/controlnet-canny-sdxl-1.0" \
--id_prompt "A photo of Egret" \
--frame_prompt_list "drink water" "stand on the ground" "walk ast the shore" \
--control_image_list "resource/controlnet/canny10.png" "resource/controlnet/canny11.png" "resource/controlnet/canny12.png" "resource/controlnet/canny13.png"




