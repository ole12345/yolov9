import os
from utils.filter_dataset import DatasetConfig, SubsetStrEnum, filter_dataset
import yaml

if __name__ == '__main__':
 root_dir = os.path.dirname(os.path.realpath(__file__))
 dataset_name="coco-2017-person-ball-small"
 category_list=["person","sports ball"]
 yaml_content = {"path":os.path.join("..","datasets",dataset_name)}

 #validation
 subset = SubsetStrEnum.VALIDATION
 max_number_of_images = 10
 cfg= DatasetConfig(root_dir = root_dir,dataset_name=dataset_name, subset = subset, max_number_of_images = max_number_of_images,category_list=category_list)
 cat_id_name_val = filter_dataset(cfg)
 yaml_content["val"] = subset+".txt"

 #train
 subset = SubsetStrEnum.TRAIN
 max_number_of_images = 100
 cfg= DatasetConfig(root_dir = root_dir,dataset_name=dataset_name, subset = subset, max_number_of_images = max_number_of_images,category_list=category_list)
 cat_id_name_train = filter_dataset(cfg)
 yaml_content["train"] = subset+".txt"
 if cat_id_name_train!=cat_id_name_val:
    raise Exception("Inconsistent categories")
 
 #write the yaml file
 yaml_content["names"] = cat_id_name_train
 yaml_content["stuff_names"] = list()
 yaml_content["download"] = None
 with open(os.path.join(root_dir,"data",dataset_name+".yaml"),"w") as file:
    yaml.dump(yaml_content, file, default_flow_style=False)
    file.close()
 



