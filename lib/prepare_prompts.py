'''
prepare for the prompts
there are several methods of generating prompts:
1. load from scene property, i.e. INPUT_POINT, INPUT_BOX
2. load from a json file
3. directly input the prompts
4. from a interactive backend
5. from a text discription, i.e. grounded dino
'''
# import argparse
import torch
import numpy as np
from .scene_property import INPUT_POINT, INPUT_BOX

def get_prompt_points(args, **kwargs):
    prompt_points, boxes, mask_id, masks = None, None, None, None
    
    mask_id = kwargs.get('mask_id', None)
    masks = kwargs.get('masks', None)
    assert args.text is not None, 'please input the text (args.text)'
    # fifth type, from a text discription, i.e. grounded dino
    # TODO
        
    # text prompt
    # masks <class 'torch.Tensor'> torch.Size([2, 1, 756, 1008])
    image = kwargs['init_rgb']
    from .self_prompting import grounding_dino_prompt
    input_boxes = grounding_dino_prompt(image, args.text)
    boxes = torch.tensor(input_boxes)[0:1]
#   transformed_boxes = kwargs['sam'].transform.apply_boxes_torch(input_boxes, image.shape[:2])

#   masks, scores, logits = kwargs['sam'].predict_torch(
#       point_coords=None,
#       point_labels=None,
#       boxes=transformed_boxes,
#       multimask_output=False,
#   )
#   masks = masks[0].detach().cpu().numpy()
#   raise NotImplementedError
    
    return {
        'prompt_points': prompt_points, # points used for segmentation [num_points, 2]
        'boxes': boxes, # boxes used for segmentation
        'mask_id': mask_id, # selected mask id, 0 or 1 or 2
        'masks': masks # one numpy array with shape [3, H, W]
    }


if __name__=="__main__":
    prompt_points = get_prompt_points()
    print(prompt_points)
    print(prompt_points.shape)
