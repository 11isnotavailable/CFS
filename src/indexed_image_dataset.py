"""Byte-preserving image packs: avoid millions of ZFS pathname lookups."""
import io
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class IndexedImages(Dataset):
    def __init__(self, root, split, transform=None):
        self.root=Path(root); self.split=split; self.transform=transform
        self.index=np.load(self.root/f'{split}_index.npy',mmap_mode='r')
        self.class_to_idx=json.loads((self.root/'classes.json').read_text())
        storage=json.loads((self.root/'storage.json').read_text()) if (self.root/'storage.json').exists() else {}
        self.raw_pixels=storage.get(split)=='chw_uint8'; self.file_prefix=storage.get(split+'_prefix',split)
        self._fds={}; self._pid=None; self._last_index=None; self._last_image=None

    def __len__(self): return len(self.index)

    def __getitem__(self,i):
        if self._pid!=os.getpid(): self._fds={}; self._pid=os.getpid()
        r=self.index[i]; shard=int(r['shard'])
        # RepeatAugSampler emits adjacent copies of an index. Decode once, but
        # always run a fresh random transform on an independent image copy.
        if self._last_index==i:
            x=self._last_image.copy()
            return (self.transform(x) if self.transform else x),int(r['label'])
        if shard not in self._fds:
            self._fds[shard]=os.open(self.root/f'{self.file_prefix}_{shard:04d}.images',os.O_RDONLY|getattr(os,'O_BINARY',0))
        if hasattr(os,'pread'):
            raw=os.pread(self._fds[shard],int(r['length']),int(r['offset']))
        else:  # Windows unit tests; each DataLoader worker owns its descriptors.
            os.lseek(self._fds[shard],int(r['offset']),os.SEEK_SET)
            raw=os.read(self._fds[shard],int(r['length']))
        if len(raw)!=int(r['length']): raise IOError('Truncated packed image')
        if self.raw_pixels:
            if self.transform is not None: raise ValueError('Validation pixels are already deterministically cropped')
            return np.frombuffer(raw,dtype=np.uint8).reshape(3,224,224).copy(),int(r['label'])
        with Image.open(io.BytesIO(raw)) as image: x=image.convert('RGB')
        self._last_index=i; self._last_image=x.copy()
        if self.transform: x=self.transform(x)
        return x,int(r['label'])

    def __getstate__(self):
        state=self.__dict__.copy(); state['_fds']={}; state['_pid']=None
        state['_last_index']=None; state['_last_image']=None; return state

    def __del__(self):
        for fd in getattr(self,'_fds',{}).values():
            try: os.close(fd)
            except OSError: pass
