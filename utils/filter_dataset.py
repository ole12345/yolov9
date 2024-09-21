from pycocotools.coco import COCO
from dataclasses import dataclass
from os.path import join, basename, splitext, exists
from os import makedirs, remove, listdir
from tqdm import tqdm as tqdm
from zipfile import ZipFile
import utils.utils as utils
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class SubsetStrEnum(str, Enum):
    VALIDATION = "val2017"
    TRAIN = "train2017"

@dataclass
class DatasetConfig:
   root_dir: str
   dataset_name : str  
   subset : SubsetStrEnum  
   max_number_of_images : int
   category_list : list

def filter_dataset(cfg : DatasetConfig):
   dataset_filter = coco_category_filter(cfg)
   return dataset_filter.filter()


class coco_category_filter:
 def __init__(self, cfg : DatasetConfig):
    self.cfg = cfg
    self.coco = None
    self.can_have_crowd = False # True not supported
    self.original_annotation_file = "instances_"+cfg.subset+".json"
    self.dataset_dir = join(cfg.root_dir,"datasets",cfg.dataset_name)
    self.image_dir = join(self.dataset_dir,"images",cfg.subset)
    self.labels_dir = join(self.dataset_dir,"labels",cfg.subset)
    self.annotation_dir = join(self.dataset_dir,"annotations") 
    self.original_annotation_dir = join(self.dataset_dir,"original_annotations") 
    self.original_annotation_file = "instances_"+cfg.subset+".json"
    self.original_annotation_file_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


 def __download_zip(self,url:str, dst_path: str):
    filename = basename(url)
    _, ext = splitext(filename)
    filepath = join(dst_path,filename)
    makedirs(dst_path,exist_ok=True)
    is_downloaded = exists(filepath)
    is_extracted = len([name for name in listdir(dst_path) if name != filename])>0
    if not is_extracted and not is_downloaded:
        utils.download_file_with_progress_bar(url,filepath)
    if not is_extracted:
        with ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dst_path)
        is_extracted = True      
    if is_extracted and exists(filepath):
        remove(filepath)

 def __download_images(self,images: list, workers = 256):
    makedirs(self.image_dir,exist_ok=True)

    if workers is None:
        for image in tqdm(iterable = images,unit="images",desc="Downloading images"):
            filepath = join(self.image_dir,image["file_name"])
            if not exists(filepath):
                utils.download_file_without_progress_bar(image["coco_url"],filepath,)
    else:
        tasks = []
        executor = ThreadPoolExecutor(max_workers=256)
        # Use tqdm to create a progress bar
        with tqdm(total=len(images),unit="images",desc="Downloading images") as progress_bar:
            # Submit the download tasks
            for image in images:
                task = executor.submit( utils.download_file_without_progress_bar,image["coco_url"], join(self.image_dir,image["file_name"]))
                tasks.append(task)

            # Process the completed tasks
            for completed_task in as_completed(tasks):
                completed_task.result()
                progress_bar.update(1)

 def __get_imgs_from_json(self):  
     cat_id_name = {}
     for name in self.cfg.category_list:
        id = self.coco.getCatIds(catNms=name)
        cat_id_name[id[0]] = name

     imgIds = list()
     #try to get roughly equal amount of images for each category. Multiple categories per image are not counted.
     numCategories = len(cat_id_name)
     max_number_of_images = len(self.coco.getImgIds()) if self.cfg.max_number_of_images is None else  self.cfg.max_number_of_images
     req_num_images_per_category = max_number_of_images//numCategories
     for id in cat_id_name.keys():
        imgids = self.coco.getImgIds(catIds=id)
        if len(imgids) >req_num_images_per_category:
           imgids = imgids[:req_num_images_per_category]
        imgIds.extend(imgids)           
     
     imgIds = list(set(imgIds))
     print("Selected {} images of '{}' with following class distribution:".format(len(imgIds),self.original_annotation_file))
     for (cat_id,cat_name) in cat_id_name.items():
        print("'{:>12}' : {} images".format(cat_name, len(self.coco.getImgIds(imgIds,cat_id))))
            
     return  self.coco.loadImgs(imgIds), imgIds

 def __write_image_reference_files(self,imgIds):
    with open(join(self.dataset_dir,self.cfg.subset+".txt"),"w") as file:
        for imgId in imgIds:
            image = self.coco.loadImgs(imgId)[0]   
            file.write(join(".","images",self.cfg.subset,image["file_name"])+"\n") 
        file.close()
       
 def __write_labels(self, imgIds):
    makedirs(self.labels_dir, exist_ok=True)
    cat_id_name = {}
    cat_id_new_old = {}
    cat_id_old_new = {}
    # reorder class ids
    for i,name in enumerate(self.cfg.category_list):
        ids = self.coco.getCatIds(catNms=name)
        cat_id_name[i] = name
        cat_id_new_old[i] = ids[0]
        cat_id_old_new[ids[0]] = i
    print("New Category Ids:")
    for id, name in cat_id_name.items():
        print("{}:{}; old id: {}".format(id,name,cat_id_new_old[id]))

    #
    annottmp = []
    all_annotation_ids = []
    
    for imgId in imgIds:
        annottmp.clear()
        image = self.coco.loadImgs(imgId)[0]    
        filename,_ = splitext(image["file_name"])
        width = image["width"]
        height = image["height"]
        newfilename = join(self.labels_dir,filename+".txt")
        for id in list(cat_id_old_new.keys()):
            annottmp.extend(self.coco.getAnnIds(catIds=id,imgIds=imgId, iscrowd=self.can_have_crowd))
        all_annotation_ids.extend(annottmp)
        if len(all_annotation_ids) <1:
           raise Exception("Error: Can't find annotation for image")     
        with open(newfilename,'w') as f:
            for ann in  self.coco.loadAnns(annottmp):   
              for seg in ann["segmentation"]:
                s = "{} ".format(cat_id_old_new[ann["category_id"]])
                s += " ".join(["{:8.6f} {:8.6f}".format(i/width, j/height) for i,j in utils.pairwise(seg)])     
                f.write(s+"\n")
            f.close()       

    #write the new json
    all_annotation_ids = list(set(all_annotation_ids))
    new_annotations = self.coco.loadAnns(all_annotation_ids)
    for annotation in new_annotations:
       annotation["category_id"] = cat_id_old_new[annotation["category_id"]]
    new_categories = []
    for category in [x for x in self.coco.dataset["categories"] if x["id"] in list(cat_id_old_new.keys())]:
       category["id"] = cat_id_old_new[category["id"]]
       new_categories.append(category)
       
    data = {
     "info": self.coco.dataset['info'],
     "licenses": self.coco.dataset['licenses'],
     "images": self.coco.loadImgs(imgIds), 
     "annotations": new_annotations,
     "categories": new_categories 
     }
    makedirs(self.annotation_dir,exist_ok=True)
    with open(join(self.annotation_dir,"instances_"+self.cfg.subset+".json"), 'w') as f:
        json.dump(data, f)
    
    return cat_id_name
 
 def filter(self):
    makedirs(self.dataset_dir,exist_ok=True)
    #download annotation file
    self.__download_zip(self.original_annotation_file_url,self.original_annotation_dir)
    #update original annotation dir 
    self.original_annotation_dir= join(self.original_annotation_dir,"annotations")

    #instantiate coco tools
    self.coco = COCO(join(self.original_annotation_dir,self.original_annotation_file))

    #get list of images with the specified categories
    images, imgIds = self.__get_imgs_from_json()
    
    #download images
    self.__download_images(images)

    #write labels and new annotation file
    cat_id_name = self.__write_labels(imgIds=imgIds)

    #write image txt files with rel image paths
    self.__write_image_reference_files(imgIds=imgIds)

    return cat_id_name