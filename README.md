# About
Author: Roberts Kalvītis
This is practical work for Bachelor's paper. Goal is to increase model accuracy by generating synthetic data and using it for model training. Model is suppost to regognize whether OOC cell should grown further based on cell image.
Model used for training: MobileNet v2 - https://huggingface.co/docs/timm/en/models/mobilenet-v2. Simmilar studies: https://www.mdpi.com/2306-5729/9/2/28. 

# Setup
1. Download synthetic datasets from https://zenodo.org/records/11197341 and https://zenodo.org/records/11197532.
2. Download real dataset from https://zenodo.org/records/10203721
3. Place data in folders with same name without changing hierarhy.
4. If goal is to generate diffrent synthetic datasets, you need to download diffustion model from https://github.com/AUTOMATIC1111/stable-diffusion-webui


# Bakalaurs notes

Dziļo neironu tīkli - attēli atpazīšana

Problēma - nepietiekami dati
risinājums - ar sintētisko datu palīdzību ģenerēt papildus treniņu datus
Esošie pētījumi -
    https://www.mdpi.com/2306-5729/9/2/28 rakts par datu kopu.
    https://zenodo.org/records/10203721 reālo datu kopa.
    https://www.edi.lv/wp-content/uploads/2023/11/Synthetic_Image_Generation_With_a_Fine_Tuned_Latent_Diffusion_Model_for_Organ_on_Chip_Cell_Image_Classification.pdf raksts par sintētisko datu ģenerēšanu
Bakalaura darba darba metožu apraksts- iespējams klasifikātori
Modeļi ar kurus vajadzētu izmēģināt:
    MobileNet v2 - https://huggingface.co/docs/timm/en/models/mobilenet-v2
    MobileNet v3 - https://huggingface.co/docs/timm/en/models/mobilenet-v3
    EfficientNet - https://huggingface.co/docs/timm/en/models/efficientnet
    exeption modelis

sintētisko datu ģenerēšana - https://stability.ai/ 



