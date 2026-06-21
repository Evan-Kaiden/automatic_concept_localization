import torch

def get_text_features(model, tokenizer, concepts, device):
    text_tokens = tokenizer(concepts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

def get_image_features(model, image):
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features

DEFAULT_NEGATIVE_CONCEPT = "a photo of something else"

def get_concept_presence_scores(model, image_features, text_features, negative_text_features):
    logit_scale = model.logit_scale.exp()
    pos_logits = logit_scale * (image_features @ text_features.T)
    neg_logits = logit_scale * (image_features @ negative_text_features.T)
    pair_logits = torch.stack([pos_logits, neg_logits.expand_as(pos_logits)], dim=-1)
    probs = pair_logits.softmax(dim=-1)[..., 0].squeeze(0)
    return probs

