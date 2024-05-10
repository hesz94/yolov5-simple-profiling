import math
from .utils import time_function
import torch

@time_function
def size_matched_idx(wh1, wh2, thresh):
    #area1 = wh1.prod(1)
    #area2 = wh2.prod(1)
    
    #wh = torch.min(wh1[:, None], wh2[None])
    #inter = wh.prod(2)
    #iou = inter / (area1[:, None] + area2 - inter)
    #return torch.where(iou > thresh)
    
    ratios = wh1[:, None] / wh2[None]
    max_ratios = torch.max(ratios, 1. / ratios).max(2)[0]
    return torch.where(max_ratios < thresh)

@time_function
def assign_targets_to_proposals(xy, size, overlap=0.5):
    x, y = xy.T
    ids = [torch.arange(len(xy), device=xy.device)]
    
    ids.append(torch.where((x > 1) & (x % 1 < overlap))[0]) # lt_x
    ids.append(torch.where((y > 1) & (y % 1 < overlap))[0]) # lt_y
    ids.append(torch.where((x < size[1] - 1) & (x % 1 > (1 - overlap)))[0]) # rb_x
    ids.append(torch.where((y < size[0] - 1) & (y % 1 > (1 - overlap)))[0]) # rb_y
    
    offsets = xy.new_tensor([[0, 0], [-overlap, 0], [0, -overlap], [overlap, 0], [0, overlap]])
    coordinates = torch.cat([xy[ids[i]] + offsets[i] for i in range(5)]).long()
    return torch.cat(ids), coordinates


# temporarily not merge box_giou and box_ciou
@time_function
def box_giou(box1, box2): # box format: (cx, cy, w, h)
    cx1, cy1, w1, h1 = box1.T
    cx2, cy2, w2, h2 = box2.T
    
    b1_x1, b1_x2 = cx1 - w1 / 2, cx1 + w1 / 2
    b1_y1, b1_y2 = cy1 - h1 / 2, cy1 + h1 / 2
    b2_x1, b2_x2 = cx2 - w2 / 2, cx2 + w2 / 2
    b2_y1, b2_y2 = cy2 - h2 / 2, cy2 + h2 / 2
    
    ws = torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)
    hs = torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    inter = ws.clamp(min=0) * hs.clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / union
    
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c_area = cw * ch
    return iou - (c_area - union) / c_area

@time_function
def box_ciou(box1, box2): # box format: (cx, cy, w, h)
    cx1, cy1, w1, h1 = box1.T
    cx2, cy2, w2, h2 = box2.T
    
    b1_x1, b1_x2 = cx1 - w1 / 2, cx1 + w1 / 2
    b1_y1, b1_y2 = cy1 - h1 / 2, cy1 + h1 / 2
    b2_x1, b2_x2 = cx2 - w2 / 2, cx2 + w2 / 2
    b2_y1, b2_y2 = cy2 - h2 / 2, cy2 + h2 / 2
   
    ws = torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)
    hs = torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    inter = ws.clamp(min=0) * hs.clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / union
    
    v = (2 / math.pi * (torch.atan(w2 / h2) - torch.atan(w1 / h1))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v)
        
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2
    return iou - (rho2 / c2 + v * alpha)

@time_function
def box_iou(box1, box2): # box format: (x1, y1, x2, y2)
    area1 = torch.prod(box1[:, 2:] - box1[:, :2], 1)
    area2 = torch.prod(box2[:, 2:] - box2[:, :2], 1)
    
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter)

@time_function
def nms(boxes, scores, threshold):
    return torch.ops.torchvision.nms(boxes, scores, threshold)

@time_function
def batched_nms(boxes, scores, labels, threshold, max_size): # boxes format: (x1, y1, x2, y2)
    offsets = labels.to(boxes) * max_size
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, threshold)
    return keep

@time_function
def all_batched_nms(ids, boxes, scores, labels, threshold, max_size): # boxes format: (x1, y1, x2, y2)
    offsets = torch.stack((labels, ids, labels, ids), dim=1) * max_size
    boxes_for_nms = boxes + offsets
    keep = nms(boxes_for_nms, scores, threshold)
    return keep

@time_function
def cxcywh2xyxy(box): # box format: (cx, cy, w, h)
    cx, cy, w, h = box.T
    ws = w / 2
    hs = h / 2
    new_box = torch.stack((cx - ws, cy - hs, cx + ws, cy + hs), dim=1)
    return new_box

@time_function
def xyxy2cxcywh(box): # box format: (x1, y1, x2, y2)
    x1, y1, x2, y2 = box.T
    new_box = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=1)
    return new_box

# profiling utilities
import functools
from torch.nn.functional import interpolate

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
    img_orig = img.numpy().squeeze()
    print('Preprocess logs, ')
    img = timed_preprocess(img, preprocess_pipe)
    print('Model logs, ')
    with torch.no_grad():
        results = model(img)
    print('Postprocess logs')
    return postprocess(results, img_orig, img_sizes)