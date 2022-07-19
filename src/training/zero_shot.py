import logging
import random
from contextlib import suppress

import torch
from torch import einsum
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import tokenize

from .imagenet_zeroshot_data import get_imagenet_classnames, get_imagenet_r_classnames, get_imagenet_a_classnames, get_imagenet_r_cipher, get_imagenet_a_cipher, get_openai_imagenet_template

try:
    from .inat_zeroshot_data import inat_classnames, inat_template
    from .cars_zeroshot_data import cars_classnames, cars_template
    from .flowers_zeroshot_data import flowers_classnames, flowers_template
    from .food_zeroshot_data import food_classnames, food_template
    from .air_zeroshot_data import air_classnames, air_template

except Exception as e:
    print(e)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            if args.zeroshot_scramble:
                res = []
                tlist = [t.split(" ") for t in texts]
                for l in tlist:
                    random.shuffle(l)
                    res.append(" ".join(l).strip())
            texts = tokenize(texts).to(args.device)  # tokenize
            # print("size of texts: ")
            # print(texts.size())
            if args.distributed and not args.horovod:
                if args.model in ["coca"]:
                    images = torch.rand(len(texts), 3, 224, 224).to(args.device)
                    class_embeddings = model.module(texts, images, return_embeddings=True)
                    class_embeddings = class_embeddings[0]
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                elif args.model in ["xclip"]:
                    images = torch.rand(len(texts), 3, model.module.image_size, model.module.image_size).to(args.device)
                    class_embeddings = model.module(texts, images, return_encodings=True)
                    class_embedding = l2norm(model.module.to_text_latent(class_embeddings[0][:, 0])).mean(dim=0)                
                else:
                    class_embeddings = model.module.encode_text(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
            else:         
                if args.model in ["coca"]:
                    images = torch.rand(len(texts), 3, 224, 224).to(args.device)
                    class_embeddings = model(texts, images, return_embeddings=True)
                    class_embeddings = class_embeddings[0]
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                elif args.model in ["xclip"]:
                    images = torch.rand(len(texts), 3, model.image_size, model.image_size).to(args.device)
                    class_embeddings = model(texts, images, return_encodings=True)
                    class_embedding = l2norm(model.to_text_latent(class_embeddings[0][:, 0])).mean(dim=0)
                else:
                    class_embeddings = model.encode_text(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
        # if args.model == "xclip":
        #     zeroshot_weights = einsum('b c d -> c d', zeroshot_weights)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            target = target.to(args.device)
            #FIXME: handle larger batch sizes gracefully with gradient caching
            if args.gc:
                images = images[:min(args.gpumaxbatch, len(images)-1)]
                target = target[:min(args.gpumaxbatch, len(images)-1)]
            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    if args.model == "coca":
                        texts = torch.randint(100, (5, len(images))).to(args.device)
                        image_features = model.module(texts, images, return_embeddings=True)
                        image_features = F.normalize(image_features[1], dim=-1)
                        logits = model.module.temperature.exp() * image_features @ classifier   
                    elif args.model == "xclip":
                        texts = torch.randint(100, (len(images), 5)).to(args.device)
                        image_features = model.module(texts, images, return_encodings=True)
                        image_features = l2norm(model.module.to_visual_latent(image_features[1][:, 0]))
                        logits = model.module.temperature.exp() * image_features @ classifier                             
                    else:
                        image_features = model.module.encode_image(images)
                        image_features = F.normalize(image_features, dim=-1)
                        logits = 100. * image_features @ classifier
                else:
                    if args.model == "coca":
                        texts = torch.randint(100, (5, len(images))).to(args.device)
                        image_features = model(texts, images, return_embeddings=True)
                        image_features = F.normalize(image_features[1], dim=-1)
                        logits = model.temperature.exp() * image_features @ classifier         
                    elif args.model == "xclip":
                        texts = torch.randint(100, (len(images), 5)).to(args.device)
                        image_features = model(texts, images, return_encodings=True)
                        image_features = l2norm(model.to_visual_latent(image_features[1][:, 0]))
                        logits = model.temperature.exp() * image_features @ classifier                             
                    else:
                        image_features = model.encode_image(images)
                        image_features = F.normalize(image_features, dim=-1)
                        logits = 100. * image_features @ classifier

            # measure accuracy
            logging.debug("size of logits: {}, size of target: {}".format(logits.size(), target.size()))
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def to_upper(l):
    return [c.upper() for c in l]

def to_lower(l):
    return [c.lower() for c in l]

def build_imagenet(args, model, in_type=""):
    if in_type == "r":
        if args.zs_upper:
            classifier = zero_shot_classifier(model, to_upper(get_imagenet_r_classnames()), get_openai_imagenet_template(), args)
        elif args.zs_lower:
            classifier = zero_shot_classifier(model, to_lower(get_imagenet_r_classnames()), get_openai_imagenet_template(), args)
        if args.ds_cipher:
            classifier = zero_shot_classifier(model, get_imagenet_r_cipher(), get_openai_imagenet_template(), args)
        else:
            classifier = zero_shot_classifier(model, get_imagenet_r_classnames(), get_openai_imagenet_template(), args)   
    elif in_type == "a":
        if args.zs_upper:
            classifier = zero_shot_classifier(model, to_upper(get_imagenet_a_classnames()), get_openai_imagenet_template(), args)
        elif args.zs_lower:
            classifier = zero_shot_classifier(model, to_lower(get_imagenet_a_classnames()), get_openai_imagenet_template(), args)
        if args.ds_cipher:
            classifier = zero_shot_classifier(model, get_imagenet_a_cipher(), get_openai_imagenet_template(), args)
        else:
            classifier = zero_shot_classifier(model, get_imagenet_a_classnames(), get_openai_imagenet_template(), args)  
    else:
        if args.zs_upper:
            classifier = zero_shot_classifier(model, to_upper(get_imagenet_classnames()), get_openai_imagenet_template(), args)
        elif args.zs_lower:
            classifier = zero_shot_classifier(model, to_lower(get_imagenet_classnames()), get_openai_imagenet_template(), args)
        elif args.ds_cipher:
            classifier = zero_shot_classifier(model, get_imagenet_cipher(), get_openai_imagenet_template(), args)
        else:
            classifier = zero_shot_classifier(model, get_imagenet_classnames(), get_openai_imagenet_template(), args)
    return classifier

def zero_shot_eval(model, data, epoch, args):
    logging.debug(data)
    
    results = {}
    classifier = None

    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'imagenet-r' not in data and 'imagenet-s' not in data and 'imagenet-a' not in data and 'inat2021' not in data and 'stanfordcars' not in data and 'flowers' not in data and 'food' not in data:
        return results
    if args.zeroshot_frequency == 0:
        return results
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return results
    if 'inat2021' in data:
        # if args.zs_upper:
        #     inat_classnames = to_upper(inat_classnames)
        # elif args.zs_lower:
        #     inat_classnames = to_lower(inat_classnames)
        logging.info("Starting zero-shot inat2021.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, inat_classnames, inat_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['inat2021'].dataloader, args)
        results['inat2021-top1'] = top1
        results['inat2021-top5'] = top5

        logging.info('Finished zero-shot inat2021. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'stanfordcars' in data:
        # if args.zs_upper:
        #     cars_classnames = to_upper(cars_classnames)
        # elif args.zs_lower:
        #     cars_classnames = to_lower(cars_classnames)
        logging.info("Starting zero-shot stanfordcars.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, cars_classnames, cars_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['stanfordcars'].dataloader, args)
        results['stanfordcars-top1'] = top1
        results['stanfordcars-top5'] = top5

        logging.info('Finished zero-shot stanfordcars. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'flowers' in data:
        # if args.zs_upper:
        #     flowers_classnames = to_upper(flowers_classnames)
        # elif args.zs_lower:
        #     flowers_classnames = to_lower(flowers_classnames)
        logging.info("Starting zero-shot flowers.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, flowers_classnames, flowers_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['flowers'].dataloader, args)
        results['flowers-top1'] = top1
        results['flowers-top5'] = top5

        logging.info('Finished zero-shot flowers. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'air' in data:
        logging.info("Starting zero-shot FGVC-aircraft.")
        logging.info('Building zero-shot classifier')
        # if args.zs_upper:
        #     air_classnames = to_upper(air_classnames)
        # elif args.zs_lower:
        #     air_classnames = to_lower(air_classnames)
        classifier = zero_shot_classifier(model, air_classnames, air_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['air'].dataloader, args)
        results['FGVC-aircraft-top1'] = top1
        results['FGVC-aircraft-top5'] = top5

        logging.info('Finished zero-shot FGVC-aircraft. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'food' in data:
            logging.info("Starting zero-shot food.")
            logging.info('Building zero-shot classifier')
            # if args.zs_upper:
            #     food_classnames = to_upper(food_classnames)
            # elif args.zs_lower:
            #     food_classnames = to_lower(food_classnames)
            classifier = zero_shot_classifier(model, food_classnames, food_template, args)

            logging.info('Using classifier')
            top1, top5 = run(model, classifier, data['food'].dataloader, args)
            results['food-top1'] = top1
            results['food-top5'] = top5

            logging.info('Finished zero-shot food. Top1 was {}, top5 was {}'.format(top1, top5))
            
    logging.info('Starting zero-shot imagenet.')

    classifier = None
    
    if 'imagenet-val' in data:
        if classifier is None:
            logging.info('Building zero-shot classifier')
            classifier = build_imagenet(args, model)
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot val. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-v2' in data:
        if classifier is None:
            logging.info('Building zero-shot classifier')
            classifier = build_imagenet(args, model)
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot v2. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-s' in data:
        if classifier is None:
            logging.info('Building zero-shot classifier')
            classifier = build_imagenet(args, model)
        top1, top5 = run(model, classifier, data['imagenet-s'].dataloader, args)
        results['imagenets-zeroshot-val-top1'] = top1
        results['imagenets-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot sketch. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-r' in data:
        classifier = build_imagenet(args, model, "r")
        top1, top5 = run(model, classifier, data['imagenet-r'].dataloader, args)
        results['imagenetr-zeroshot-val-top1'] = top1
        results['imagenetr-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot imagenet-r. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-a' in data:
        classifier = build_imagenet(args, model, "a")
        top1, top5 = run(model, classifier, data['imagenet-a'].dataloader, args)
        results['imageneta-zeroshot-val-top1'] = top1
        results['imageneta-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot imagenet-a. Top1 was {}, top5 was {}'.format(top1, top5))

    logging.info('Finished zero-shot evals')

    return results
