# About
Author: Roberts Kalvītis
This is practical work for Bachelor's paper. Goal is to increase model accuracy by generating synthetic data and using it for model training. Model is suppost to regognize whether OOC cell should grown further based on cell image.
Model used for training: MobileNet v2 - https://huggingface.co/docs/timm/en/models/mobilenet-v2. Simmilar studies: https://www.mdpi.com/2306-5729/9/2/28. 

# Setup
1. Download synthetic datasets from __ and __.
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


Titullapa
Anotācija/Abstract
Saturs
Apzīmējumu saraksts
Ievads
2. Apraksts par OOC
    kapec biomedicinas atteli ir sarezgiti
    uzrakstit teoriju par ooc organ on chip
    MI pielietojums medicina - ooc

1. Sintētiskie dati
    Kursa darbs
    detalizēt aprakstit sintetiko datu generēšanas metodes, forumas u.c. lietas

    
3. Sintētisko datu ģenerēšana
    Reālās datu kopas apraksts
    Stable diffusion vs generative adversarial network (viens ģenerē un viens cenšas atšķirt)
    Metodes

4. Modeļu apmācības process
    Pre-trained modeļu apraksts (MobileNetV2, EfficientNetB7)
    Izmantotā sistemātika

Rezultāti un diskusija
    stiprās un vājās puses - citas datu ģenerēšanas metodes, vai cita modeļa izmantošanu (EfficientNetB7) - mazak eksperimetus varētu izmantot
    Mazjaudīgas ierīces, ko vieglāk var izmantot labratorijas
Secinājumi
Izmantotā literatūra un avoti
Pielikumi



