import os
import torch
from PIL import Image
import RISE
import open_clip
import encoder_utils
import visualize
from argparse import ArgumentParser

parser = ArgumentParser()
## CLIP arguments
parser.add_argument('--show_models', action="store_true")
parser.add_argument('--clip-model', type=str, default='ViT-SO400M-14-SigLIP2',
                     help='open_clip model architecture to load, e.g. ViT-B-32, ViT-L-14, ViT-H-14. '
                          'See open_clip.list_pretrained() for valid (model, pretrained) combinations.')
parser.add_argument('--pretrained', type=str, default='webli',
                     help='Pretrained weights tag for --clip-model, e.g. openai, laion2b_s34b_b79k')
parser.add_argument('--image', type=str, required=True,
                     help='Path to the input image')
parser.add_argument('--concepts', type=str, nargs='+', required=True,
                     help='Candidate concept strings, e.g. --concepts "a wheel" "a wing"')

## RISE arguments
parser.add_argument('--n_masks', type=int, default=10000)
parser.add_argument('--p1', type=float, default=0.1,
                     help='Probability that a grid cell is kept (1) before upsampling')
parser.add_argument('--pre_upsample_size', type=int, default=16,
                     help='Side length of the low-res grid that gets upsampled to the image size')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

## visualization arguments
parser.add_argument('--out_dir', type=str, default='results',
                     help='Folder to save the per-concept overlayed saliency maps')


def n_boxes_type(value):
    return value if value == 'auto' else int(value)


parser.add_argument('--n_boxes', type=n_boxes_type, default='auto',
                     help='Number of bounding boxes to draw around the most focused/dense '
                          'saliency regions per concept. Pass an integer, or "auto" to let '
                          'the most relevant regions be selected automatically.')
parser.add_argument('--box_threshold', type=float, default=0.5,
                     help='Fraction of a saliency map\'s max value used to binarize it before '
                          'finding connected regions for bounding boxes')
parser.add_argument('--auto_box_rel_thresh', type=float, default=0.3,
                     help='When --n_boxes auto, keep regions whose saliency mass is at least '
                          'this fraction of the most focused region\'s mass')
args = parser.parse_args()

device = args.device
model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.pretrained, device=device)
tokenizer = open_clip.get_tokenizer(args.clip_model)
model.eval()

if args.show_models:
     print(open_clip.list_pretrained())
     exit()

## get concepts
if os.path.exists(args.concepts[0]):
    print(f"Provided concept file {args.concepts}")
    with open(args.concepts[0], "r") as f:
            concepts = [line.strip() for line in f]

else:
    concepts = args.concepts

# get features for those concepts
text_features = encoder_utils.get_text_features(model, tokenizer, concepts, device)

# get features for a generic negative concept used to score each concept against
negative_text_features = encoder_utils.get_text_features(model, tokenizer, [encoder_utils.DEFAULT_NEGATIVE_CONCEPT], device)

# get features for the input image
image = preprocess(Image.open(args.image).convert('RGB')).unsqueeze(0).to(device)
image_features = encoder_utils.get_image_features(model, image)

# get the present concepts in the image
present_prob = encoder_utils.get_concept_presence_scores(model, image_features, text_features, negative_text_features)
for concept, prob in sorted(zip(concepts, present_prob.tolist()), key=lambda x:x[1], reverse=True):
    print(f'{concept}: {prob:.4f}')

# run the RISE based localization
# ugly hack to get the models input image size
def get_image_size(preprocess, model):
    for t in preprocess.transforms:
        name = t.__class__.__name__
        if name == "CenterCrop":
            return t.size
        if name == "Resize":
            size = t.size
            # torchvision Resize stores size as int or (h, w)
            return size if isinstance(size, int) else size[0]
    # read from model's positional embedding grid
    try:
        patch_size = model.visual.patch_size
        grid = model.visual.grid_size
        return patch_size * grid[0]
    except AttributeError:
        return 378  # hardcoded fallback if you know the input image size

image_size = get_image_size(preprocess, model)
image_size = (image_size, image_size)
rise_masks = RISE.generate_masks(N=args.n_masks, s=args.pre_upsample_size, p1=args.p1, input_size=image_size, device=device)

concept_predictor = lambda x: encoder_utils.get_concept_presence_scores(model, encoder_utils.get_image_features(model, x), text_features, negative_text_features)
explanation = RISE.explain(model=concept_predictor, inp=image, masks=rise_masks, p1=args.p1, batch_size=128, device=device)

# return the location of the present concepts in the image
# save the overlayed saliency map for each concept, one image per concept
display_image = visualize.get_display_image(Image.open(args.image).convert('RGB'), preprocess)
visualize.save_concept_saliency_maps(display_image, explanation.cpu().numpy(), concepts, {k: v for k, v in zip(concepts, present_prob)}, args.out_dir,
                                      n_boxes=args.n_boxes, box_threshold=args.box_threshold,
                                      auto_rel_thresh=args.auto_box_rel_thresh)
print(f'Saved saliency map visualizations to {args.out_dir}')