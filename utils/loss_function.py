import torch

def get_loss(attention_layout, attention_text, input):
    loss = torch.tensor(0.0)  
    sum_box = torch.tensor(0.0)
    sum_all = torch.tensor(0.0)
    for i, (box, location) in enumerate(zip(input['boundingbox'], input['token_location'])):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(16*x1), int(16*y1), int(16*x2), int(16*y2)
        sum_box = torch.sum(attention_layout[y1:y2, x1:x2, location])
        sum_all = torch.sum(attention_layout[:, :, location])
        loss += 1 - (sum_all-sum_box) / sum_all
    for i, (box, location) in enumerate(zip(input['boundingbox'], input['token_location'])):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(16*x1), int(16*y1), int(16*x2), int(16*y2)
        sum_box = torch.sum(attention_text[y1:y2, x1:x2, location])
        sum_all = torch.sum(attention_text[:, :, location])
        loss += 1 - sum_box / sum_all
    return loss.to("cuda")


def get_img(img_layout, img_text, input):
    for box in input['boundingbox']:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(64*x1), int(64*y1), int(64*x2), int(64*y2)
        img_text[:, :, y1:y2, x1:x2] = img_layout[:, :, y1:y2, x1:x2]
    return img_text


def get_img2(img_layout, img_text, input):
    for box in input['boundingbox']:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(64*x1), int(64*y1), int(64*x2), int(64*y2)
        img_layout[:, :, y1:y2, x1:x2] = img_text[:, :, y1:y2, x1:x2]
    return img_layout