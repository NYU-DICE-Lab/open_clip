import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import tokenize
from .imagenet_zeroshot_data import imagenet_classnames, imagenet_r_classnames, imagenet_a_classnames, openai_imagenet_template
try:
    from .inat_zeroshot_data import inat_classnames, inat_template
    from .cars_zeroshot_data import cars_classnames, cars_template
    from .flowers_zeroshot_data import flowers_classnames, flowers_template

except Exception as e:
    print(e)


def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenize(texts).to(args.device)  # tokenize
            if args.distributed and not args.horovod:
                if args.model in ["coca", "xclip"]:
                    images = torch.rand(len(texts), 3, model.image_size, model.image_size).to(args.device)
                    class_embeddings = model.module(texts, images, return_encodings=True)
                    class_embeddings = class_embeddings[0]
                else:
                    class_embeddings = model.module.encode_text(texts)
            else:
                if args.model in ["coca", "xclip"]:
                    images = torch.rand(len(texts), 3, model.image_size, model.image_size).to(args.device)
                    class_embeddings = model(texts, images, return_encodings=True)
                    class_embeddings = class_embeddings[0][:, -1, :]     
                else:
                    class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
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
                    if args.model in ["coca", "xclip"]:
                        texts = torch.rand(len(images), 5).to(args.device)
                        image_features = model.module(texts, images, return_encodings=True)
                        image_features = image_features[1][:, -1, :]
                    else:
                        image_features = model.module.encode_image(images)
                else:
                    if args.model in ["coca", "xclip"]:
                        texts = torch.randint(100, (len(images), 5)).to(args.device)
                        image_features = model(texts, images, return_encodings=True)
                        image_features = image_features[1][:, -1, :]
                    else:
                        image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args):
    logging.debug(data)
    
    results = {}
    classifier = None

    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'imagenet-r' not in data and 'imagenet-s' not in data and 'imagenet-a' not in data and 'inat2021' not in data and 'stanfordcars' not in data and 'flowers' not in data:
        return results
    if args.zeroshot_frequency == 0:
        return results
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return results
    if 'inat2021' in data:
        logging.info("Starting zero-shot inat2021.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, inat_classnames, inat_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['inat2021'].dataloader, args)
        results['inat2021-top1'] = top1
        results['inat2021-top5'] = top5

        logging.info('Finished zero-shot inat2021. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'stanfordcars' in data:
        logging.info("Starting zero-shot stanfordcars.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, cars_classnames, cars_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['stanfordcars'].dataloader, args)
        results['stanfordcars-top1'] = top1
        results['stanfordcars-top5'] = top5

        logging.info('Finished zero-shot stanfordcars. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'flowers' in data:
        logging.info("Starting zero-shot flowers.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, flowers_classnames, flowers_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['flowers'].dataloader, args)
        results['flowers-top1'] = top1
        results['flowers-top5'] = top5

        logging.info('Finished zero-shot flowers. Top1 was {}, top5 was {}'.format(top1, top5))

    logging.info('Starting zero-shot imagenet.')

    classifier = None
    
    if 'imagenet-val' in data:
        if type(classifier) is None:
            logging.info('Building zero-shot classifier')
            classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        if type(classifier) is None:
            logging.info('Building zero-shot classifier')
            classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
    if 'imagenet-s' in data:
        if type(classifier) is None:
            logging.info('Building zero-shot classifier')
            classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)
        top1, top5 = run(model, classifier, data['imagenet-s'].dataloader, args)
        results['imagenets-zeroshot-val-top1'] = top1
        results['imagenets-zeroshot-val-top5'] = top5
    if 'imagenet-r' in data:
        classifier_r = zero_shot_classifier(model, imagenet_r_classnames, openai_imagenet_template, args)
        top1, top5 = run(model, classifier_r, data['imagenet-r'].dataloader, args)
        results['imagenetr-zeroshot-val-top1'] = top1
        results['imagenetr-zeroshot-val-top5'] = top5
    if 'imagenet-a' in data:
        classifier_a = zero_shot_classifier(model, imagenet_a_classnames, openai_imagenet_template, args)
        top1, top5 = run(model, classifier_a, data['imagenet-a'].dataloader, args)
        results['imageneta-zeroshot-val-top1'] = top1
        results['imageneta-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
