from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import cv2
import os
import functools
from torch.nn.functional import interpolate
import torch
import yolo
import pprint
import sys

class ImgDataset(Dataset):
    def __init__(self, image_dir, cuda=False):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.split('.')[-1] in ['png', 'jpg']]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)[:,:,::-1]
        return image.copy(), img_path # copy to fix strides


class TimedModule(nn.Module):
    # utility wrapper module with timing and printing logic
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        iscuda = x.device.type == 'cuda'
        if iscuda: # cuda execution calls are asynchronous, sync needed for proper timing
            torch.cuda.synchronize()
            
        start_time = time.time()
        output = self.module(x)
        
        if iscuda: # cuda execution calls are asynchronous, sync needed for proper timing
            torch.cuda.synchronize()
        execution_time = time.time() - start_time
        print(f'{str(self.module)}, {execution_time*1000:.4f}') # display module and time in ms
        return output
    
def replace_modules(model, module_types):
    # crawls through model, wrapping all modules of interest in TimedModule utility wrapper
    for name, module in model.named_children():
        # If the module is an instance of the specified types, wrap it
        if isinstance(module, module_types):
            setattr(model, name, TimedModule(module))
        else:
            # Otherwise, recursively process the current module
            replace_modules(module, module_types)
            
            
# profiling utilities


def time_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = (time.time()  - start_time) * 1000
        print(f'{func.__name__}, {duration:.4f}')
        return result
    return wrapper

def profiled_function(func, *args, **kwargs):
    start_time = time.time()  
    result = func(*args, **kwargs) 
    duration = (time.time() - start_time) * 1000
    if isinstance(func, functools.partial):
        name = func.func.__name__
    else:
        name = func.__name__
    print(f'{name}, {duration:.4f}')
    return result

# preprocessing functions

def resize(img, shape):
    return interpolate(img, shape) # could set interpolation method if performance is critical / use torch implementation
    
def to_01_range(img):
    return img.type(torch.float32)/255.0
    
def to_nchw(img):
    return img.permute(0,3,1,2)

def to_cuda(img):
    return img.cuda()


# postprocessing functions


@time_function
def get_rescale_factor(img, inference_size):
    x_scale_factor = 1/inference_size*img.shape[1]
    y_scale_factor = 1/inference_size*img.shape[0]
    return torch.tensor([x_scale_factor, y_scale_factor, x_scale_factor, y_scale_factor])

@time_function
def scale_and_convert_box(box, rescale_tensor):
    return (box * rescale_tensor).type(torch.int32).numpy()

@time_function
def draw_box(img, box):
    return cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), 255, 2)

@time_function
def move_results_to_cpu(results):
    return results.cpu()

# pipeline and its elements
def postprocess(model_outs, original_image, inference_size):
    boxes = model_outs[0][0]['boxes']
    if boxes.device.type == 'cuda':
        boxes = move_results_to_cpu(boxes)
    rescale_tensor = get_rescale_factor(original_image, inference_size)
    for box in boxes:
        processed_box = scale_and_convert_box(box, rescale_tensor)
        original_image = draw_box(original_image, processed_box)
    return original_image
        
def timed_preprocess(img, pipe):
    for stage in pipe:
        img = profiled_function(stage, img)
    return img        

def run_through_pipe(img, model, preprocess_pipe, img_sizes):
    warmup(model)
    img_orig = img.numpy().squeeze()
    print('Preprocess logs, ')
    img = timed_preprocess(img, preprocess_pipe)
    print('Model logs, ')
    with torch.no_grad():
        results = model(img)
    print('Postprocess logs')
    return postprocess(results, img_orig, img_sizes)

def warmup(model):
    device = next(model.parameters()).device
    # Redirect stdout to suppress print statements - we don't want warmup logs
    with SuppressPrint():
        with torch.no_grad():
            # warmup
            for i in range(100):
                model(torch.rand(1, 3, 224, 224).to(device))

class SuppressPrint: # print suppressor for model warmup
    class _Suppressor:
        def write(self, x):
            pass
        
        def flush(self):
            pass

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self._Suppressor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

def main():
    config = {
    'image_directory': 'data_in',
    'ckpt_path': "yolov5s_official_2cf45318.pth",
    'img_sizes': 672
    }

    # defining pre-processing pipeline
    cpu_preprocess = [to_01_range, to_nchw, functools.partial(resize, shape=(config['img_sizes'], config['img_sizes']))]
    gpu_preprocess = cpu_preprocess + [to_cuda]

    # creating dataset and dataloader
    dataset = ImgDataset(config['image_directory'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
    # batch size set to 1 due to dataloader being bare-bones - could define collate fn or move pre-processing to dataset if
    # profiling batch-processing is required

    # building model to inspect it
    model = yolo.YOLOv5(80, img_sizes=config['img_sizes'], score_thresh=0.3)
    model.eval()
            
    checkpoint = torch.load(config['ckpt_path'])
    model.load_state_dict(checkpoint)

    # defining which low level ops to profile, based on list above - can (should?) be changed depending on needs
    ops_of_interest = (
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.activation.LeakyReLU,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample
    )

    # actually running the replacer
    replace_modules(model, ops_of_interest)
    
    #running profiler through dataset
    for img, img_path in dataloader:
        print("\n\n### CPU PROFILING {} ###\n\n".format(img_path))
        processed = run_through_pipe(img, model.cpu(), cpu_preprocess, config['img_sizes'])
        print("\n\n### GPU PROFILING {} ###\n\n".format(img_path))
        processed = run_through_pipe(img, model.cuda(), gpu_preprocess, config['img_sizes'])

if __name__ == "__main__":
    main()
    