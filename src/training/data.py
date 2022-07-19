import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value
from functools import partial

import braceexpand
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # for notebooks
tqdm.pandas()
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('tagsets')
    from nltk.corpus import stopwords, wordnet
    lemmatizer = nltk.stem.WordNetLemmatizer()

except:
    print("nltk load failed, filtering not available")

from open_clip import tokenize

from .imagenet_zeroshot_data import get_imagenet_classnames, get_imagenet_r_classnames, get_imagenet_a_classnames, get_imagenet_r_cipher, get_imagenet_a_cipher, get_openai_imagenet_template

imagenet_classnames = get_imagenet_classnames()

try:
    from .inat_zeroshot_data import inat_classnames, inat_template
    from .cars_zeroshot_data import cars_classnames, cars_template
    from .flowers_zeroshot_data import flowers_classnames, flowers_template
    from .food_zeroshot_data import food_classnames, food_template
    from .air_zeroshot_data import air_classnames, air_template

except Exception as e:
    print(e)

cipher_dict = {
    'a': '@^',
    'b': '#^',
    'c': '$^',
    'd': '%^',
    'e': '@&',
    'f': '#&',
    'g': '$&',
    'h': '%&',
    'i': '@*',
    'j': '#*',
    'k': '$*',
    'l': '%*',
    'm': '@(',
    'n': '#(',
    'o': '$(',
    'p': '%(',
    'q': '@)',
    'r': '#)',
    's': '$)',
    't': '%)',
    'u': '@+',
    'v': '#+',
    'w': '$+',
    'x': '%+',
    'y': '@=',
    'z': '#=',
    ' ': ' ',
    '\'': '\'',
    ',': ',',
    '.': '.',
    '!': '!',
    '?': '?',
}

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, csvfilter, csvscrambled, csvcleaned, dscipher, simplecaptions, strict, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}')
        df = pd.read_csv(input_filename, sep=sep)
        if dscipher:
            csvcleaned=True
        if csvcleaned:
            logging.debug("cleaning captions")
            df.loc[:, caption_key] = df.title.progress_apply(clean_captions)
            logging.debug("Done.")
            logging.debug(df.head())
        if dscipher or simplecaptions:
            logging.debug('Filtering and enciphering captions. Original dataset size is {}'.format(len(df)))
            df['title'] = df[caption_key].progress_apply(synset_ds, ngram=3, ds=csvfilter, cipher=dscipher, simplecaptions=simplecaptions, strict=strict)
            logging.debug(df['title'].head())
            df = df[df['title'].str.len() > 0]
            logging.debug("Done. Length is now {}".format(len(df)))
            logging.debug(df.head())            
        elif csvfilter != "":
            logging.debug('Filtering captions. Original dataset size is {}'.format(len(df)))
            df['is_synset'] = df[caption_key].progress_apply(synset_ds, ngram=3, ds=csvfilter, cipher=False, simplecaptions=False, strict=strict)
            logging.debug(df['is_synset'].head())
            df = df[df['is_synset']].drop(columns=['is_synset'])
            logging.debug("Done. Length is now {}".format(len(df)))
            logging.debug(df.head())
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.scrambled = csvscrambled
        logging.debug('Done loading data')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        try:
            images = self.transforms(Image.open(str(self.images[idx])))
            texts = str(self.captions[idx])
            if self.scrambled:
                texts = scramble_txt(texts)
            texts = tokenize(texts)[0]
        except Exception as e:
            logging.debug("Missing or unreadable image at {}, generating dummy image and caption.".format(str(self.images[idx])))
            logging.debug("error message {}".format(e))
            imarray = np.random.rand(224,224,3) * 255
            images = self.transforms(
                Image.fromarray(imarray.astype('uint8')).convert('RGBA')
                )
            texts = tokenize(["NONE"])[0]
        return images, texts

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_txt(text):
    try:
        text = str(text)
        return tokenize([text])[0]
    except Exception as e:
        print(e)
        return ""

def filter_preprocess_txt(text, ds, scrambled, dscipher, simplecaptions, strict):
    try:
        text = clean_captions(str(text))
        if ds != "":
            if dscipher and simplecaptions:
                text = synset_ds(text, 3, ds, True, True, strict)
            elif dscipher:
                text = synset_ds(text, 3, ds, True, False, strict)
            elif simplecaptions:
                text = synset_ds(text, 3, ds, False, True, strict)
            elif synset_ds(text, 3, ds, False, False, strict):
                text = text
                if scrambled:
                    text = scramble_txt(text)
            else:
                text = ""
        #logging.debug("leaving filter_preprocess")
        #logging.debug(text)
        return text
    except Exception as e:
        print("in filter preprocess: ")
        print(e)
        return ""

def scramble_txt(text):
    tlist = text.split(" ")
    random.shuffle(tlist)
    text = " ".join(tlist).strip()   
    return text 

"""
Synset builder

Dataset argument expects a list of class names as strings -- any dataset can be used
Strict follows the methodology of Fang et al ... multiple matches -> no match
nva uses parts of speech for all of wordnet, instead of matching on some list from a dataset
WARNING: can return string or bool, depending on arguments provided
"""

def synset_ds(s, ngram=3, ds=None, cipher=False, simplecaptions=False, strict=False):
    try:
        flag = False
        s = [lemmatizer.lemmatize(t) for t in s.split(" ")]
        # logging.debug('s is now {}'.format(s))
        str_s = " ".join(w for w in s)
        # logging.debug('str_s is now {}'.format(str_s))
        for count, word in enumerate(s):
            if strict and flag:
                break
            grams = []
            for i in range(ngram):
                if count + i - 1 > len(s):
                    continue
                gram = " ".join(w for w in s[count:count+i+1] if len(w) > 2)
                grams.append(gram)
            for i, gram in enumerate(grams):
                if strict and flag:
                    break
                gram_t = gram
                if cipher:
                    k = ""
                    for c in gram:
                        nextc = cipher_dict.get(c) or c
                        k = k + nextc
                    gram_t = k
                if ds and gram_t in ds:
                    # logging.debug("k is now {}".format(k))
                    if simplecaptions and not flag:
                        str_s = "An image of " + gram_t
                    elif simplecaptions and flag and str_s.find(gram) == -1:
                        str_s += " {}".format(gram)
                    flag = True
                    if cipher:
                        # logging.debug("str_s before cipher replacement: {}".format(str_s))
                        str_s = str_s.replace(gram, k)
                        # logging.debug("str_s after cipher replacement: {}".format(str_s))
                elif simplecaptions and not ds:
                    d = wordnet.synsets(gram)
                    if d in stopwords.words('english'):
                        continue
                    elif d and not flag:
                        str_s = "An image of " + gram
                    elif d and str_s.find(gram) == -1:
                        str_s += " {}".format(gram)
                    flag=True
    
        if cipher or simplecaptions:
            if not flag:
                str_s = ""
            logging.debug("Returning {}".format(str_s))
            return str_s
        return flag
    except Exception as e:
        print("Exception in synset ds: ")
        print(e)
        return []

def clean_captions(x):
    try:
        return x.lower().translate({ord(i): None for i in '&<^*>\\|+=[]~`\"@/\'\©#)("'}).translate({ord(i): " " for i in ':;-_.,!?'}).replace(" www ", " ").replace(" com ", " ").replace(" photo ", " ").replace(" photos ", " ").replace(" flickr ", " ").replace(" camera ", " ").replace(" st ", " street ").replace(" de ", "").strip()
    except Exception as e:
        print("in clean captions: ")
        print(e)
        return ""

def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2", "r", "a", "s"]
    is_train = (split == "train")
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    elif is_train:
        data_path = args.imagenet_train
        preprocess_fn = preprocess_train
    else:
        if split == "val":
            data_path = args.imagenet_val
        if split == "r":
            data_path = args.imagenet_r
        if split == "a":
            data_path = args.imagenet_a
        if split == "s":
            data_path = args.imagenet_s
        preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)
    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_torchvision(args, preprocess_fns, ds):
    _, preprocess_val = preprocess_fns
    preprocess_fn = preprocess_val
    if ds == "stanfordcars":
        data_path = args.stanfordcars
        dataset = datasets.StanfordCars(root = data_path, split = 'test', transform = preprocess_fn, download = False)
    elif ds == "flowers":
        data_path = args.flowers
        dataset = datasets.Flowers102(root = data_path, split = 'test', transform = preprocess_fn, download = False)
    elif ds == "air":
        data_path = args.air
        dataset = datasets.FGVCAircraft(root=data_path, split = 'val', annotation_level = 'family', transform = preprocess_fn, download = False)
    elif ds == "food":
        data_path = args.food
        dataset = datasets.Food101(root = data_path, split = 'test', transform = preprocess_fn, download = False)
    elif ds == "inat2021":
        data_path = args.inat2021
        dataset = datasets.INaturalist(root = data_path, version = "2021_valid", transform = preprocess_fn, download = False)
    elif ds == "inat2018":
        data_path = args.inat2018
        dataset = datasets.INaturalist(root = data_path, version = "2018", transform = preprocess_fn, download = False)
    elif ds == "inat2017":
        data_path = args.inat2017
        dataset = datasets.INaturalist(root = data_path, version = "2017", transform = preprocess_fn, download = False)
    sampler = None
    dataloader = torch.utils.data.DataLoader(
    dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler
    )

    return DataInfo(dataloader, sampler)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption(sample):
    return 'txt' in sample

def filter_no_caption_text(sample):
    return sample["text"] != ""

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        shl = wds.SimpleShardList(input_shards)
        pipeline = [shl]

    # at this point we have an iterator over all the shards
    logging.debug("get_wds_dataset, is_train")
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png", text="txt"),
    ])
    if args.ds_filter != "" or args.csv_scrambled:
        logging.debug("get_wds_dataset, filtering")
        pipeline.extend([
            wds.map_dict(text=lambda x : filter_preprocess_txt(x, args.ds_filter, args.csv_scrambled, args.ds_cipher, args.simplecaptions, args.strict)),
            #wds.map_dict(text=partial(filter_preprocess_txt, args.ds_filter, args.csv_scrambled)),
            wds.select(filter_no_caption_text),
        ])
    pipeline.extend([
        wds.map_dict(image=preprocess_img, text=preprocess_txt),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=args.ds_filter != "" or not is_train)
    ])
    dataset = wds.DataPipeline(*pipeline)
    logging.debug("get_wds_dataset, pipeline assembled")
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def my_collate(batch):
    # logging.debug("batch contents: {}".format(batch))
    len_batch = len(batch) # original batch length
    # logging.debug("Before filter, batch length is {}".format(len_batch))
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    # logging.debug("After filter, batch length is {}".format(len(batch)))
    if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)

def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if args.ds_cipher:
        args.ds_filter = imagenet_cipher
    elif args.ds_filter != "":
        var_names = globals()
        args.ds_filter = var_names[args.ds_filter]
        args.ds_filter = [clean_captions(a) for a in args.ds_filter]
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        csvfilter=args.ds_filter,
        csvscrambled=args.csv_scrambled,
        csvcleaned=args.csv_cleaned,
        dscipher=args.ds_cipher,
        simplecaptions=args.simplecaptions,
        strict=args.strict,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=my_collate
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
#    try:
    # logging.debug("{}".format(next(iter(dataloader))))
#    except Exception as e:
#        logging.debug("could not load from dataloader: {}".format(e))
    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type in ["csv"]:
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    if args.imagenet_r is not None:
        data["imagenet-r"] = get_imagenet(args, preprocess_fns, "r")    
    
    if args.imagenet_s is not None:
        data["imagenet-s"] = get_imagenet(args, preprocess_fns, "s")   
    
    if args.imagenet_a is not None:
        data["imagenet-a"] = get_imagenet(args, preprocess_fns, "a")   

    if args.inat2021 is not None:
        data["inat2021"] = get_torchvision(args, preprocess_fns, "inat2021")

    if args.stanfordcars is not None:
        data["stanfordcars"] = get_torchvision(args, preprocess_fns, "stanfordcars")
    
    if args.flowers is not None:
        data["flowers"] = get_torchvision(args, preprocess_fns, "flowers")

    if args.air is not None:
        data["air"] = get_torchvision(args, preprocess_fns, "air")

    if args.food is not None:
        data["food"] = get_torchvision(args, preprocess_fns, "food")

    return data
