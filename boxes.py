import torch
import torchvision

def get_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def get_valid_boxes(current_boxes, past_boxes, image_size):
    valid_boxes = []
    for current_box in current_boxes:
        # Skip if box is too large (whole image)
        if get_area(current_box) > image_size[0] * image_size[1] * 0.9:
            continue
        
        matched_frames = 0
        for past_frame_boxes in past_boxes[max(-10, -len(past_boxes)):]: 
            ious = torchvision.ops.box_iou(current_box.unsqueeze(0), past_frame_boxes)
            abs_diff = torch.abs(current_box - past_frame_boxes)
            if (ious > 0.3).any() or (abs_diff < 40).all():
                matched_frames += 1
                
        if matched_frames >= 1:
            valid_boxes.append(current_box)
    valid_boxes = torch.stack(valid_boxes) if valid_boxes else torch.empty((0, 4), device=current_boxes.device)
    
    # in the last frame, try to match all boxes with past 2 frames, if there is match, propogate this box
    matched = []
    if len(valid_boxes) == 0 and len(past_boxes) > 3:
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                    
                for i_box in past_boxes[-i]:
                    if (torchvision.ops.box_iou(i_box.unsqueeze(0), past_boxes[-j]) > 0.3).any():
                        matched.append(i_box)
                        
        valid_boxes = torch.stack(matched) if matched else torch.empty((0, 4), device=current_boxes.device)
    return valid_boxes