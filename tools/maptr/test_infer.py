from mmdet.datasets.pipelines import Compose
import os
import json
root_dir = 'data/mapfusion/maps'
# 定义测试时的处理流程
test_pipeline = [
    dict(type='LoadMapFile')
]

pipeline = Compose(test_pipeline)


def load_train_file_info():
    ret = []
    for bag_md5 in os.listdir(root_dir):
        bag_train_info_path = f'{root_dir}/{bag_md5}/train_info.json'
        with open(bag_train_info_path) as f:
            bag_train_info = json.load(f)

        for meta in bag_train_info:
            single_meta = {
                "model_input_map1": [],
                "model_input_map2": [],
                "model_output_map": [],
                "meta": meta['meta']
            }
            for key in ['model_input_map1', 'model_input_map2', 'model_output_map']:
            # for key, value in meta.items():
                value = meta[key]
                single_meta[key] = []
                for lane_center in value['lane_center']:
                    single_meta[key].append(
                        {
                            "type": "lane_center",
                            "points": lane_center['points'],
                            "id": lane_center['id']
                        }
                    )
            ret.append(single_meta)   
    return ret
    
data_infos = load_train_file_info()
def get_data_info(index):
    """Get data info according to the given index.
    Args:
        index (int): Index of the sample data to get.
    """
    info = data_infos[index]
    # standard protocal modified from SECOND.Pytorch
    input_dict = dict(
        model_input_map1=info['model_input_map1'],
        model_input_map2=info['model_input_map2'],
        model_output_map=info['model_output_map'],
        meta = info['meta']
    )
    return input_dict
index = 1
input_dict = get_data_info(index)
example = pipeline(input_dict)
example = vectormap_pipeline(example, input_dict)



# 确保键 "img" 存在并且其值是 PyTorch 张量
print("Data keys:", data.keys())
print("Type of 'img':", type(data['img']))

# 将数据移动到 GPU
data['img'] = [img_tensor.cuda() for img_tensor in data['img']]