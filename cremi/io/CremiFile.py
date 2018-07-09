from __future__ import print_function

from abc import ABCMeta
from six import add_metaclass

import h5py
import numpy as np
from .. import Annotations
from .. import Volume

try:
    import z5py
except ImportError:
    z5py = False

try:
    unicode_dtype = h5py.special_dtype(vlen=unicode)
except NameError:
    unicode_dtype = h5py.special_dtype(vlen=str)


@add_metaclass(ABCMeta)
class AbstractCremiFile(object):
    def _create_group(self, group):

        path = "/"
        for d in group.split("/"):
            path += d + "/"
            try:
                self.file.create_group(path)
            except ValueError:
                pass

    def _create_dataset(self, path, data, dtype, compression=None):
        """Wrapper around h5py's create_dataset. Creates the group, if not
        existing. Deletes a previous dataset, if existing and not compatible.
        Otherwise, replaces the dataset.
        """

        self._replace_compatible_dataset(path, data, dtype)
        self.file.create_dataset(path, data=data, dtype=dtype, compression=compression)

    def _replace_compatible_dataset(self, path, data, dtype):
        group = "/".join(path.split("/")[:-1])
        ds_name = path.split("/")[-1]

        self._create_group(group)

        if ds_name in self.file[group]:

            ds = self.file[path]
            if ds.dtype == dtype and ds.shape == np.asarray(data).shape:
                print("overwriting existing dataset")
                self.file[path][:] = data[:]
                return

            del self.file[path]

    def _spatial(self, sequence):
        """Involutory function for transforming coordinates between python and file"""
        return sequence

    def write_volume(self, volume, ds_name, dtype):

        self._create_dataset(ds_name, data=volume.data, dtype=dtype, compression="gzip")
        self.file[ds_name].attrs["resolution"] = self._spatial(volume.resolution)
        self.file[ds_name].attrs["offset"] = self._spatial(volume.offset)
        if volume.comment is not None:
            self.file[ds_name].attrs["comment"] = str(volume.comment)

    def read_volume(self, ds_name):

        volume = Volume(self.file[ds_name])

        volume.resolution = self._spatial(self.file[ds_name].attrs["resolution"])
        if "offset" in self.file[ds_name].attrs:
            volume.offset = self._spatial(self.file[ds_name].attrs["offset"])
        if "comment" in self.file[ds_name].attrs:
            volume.comment = self.file[ds_name].attrs["comment"]

        return volume

    def _has_volume(self, ds_name):

        return ds_name in self.file

    def write_raw(self, raw):
        """Write a raw volume.
        """

        self.write_volume(raw, "/volumes/raw", np.uint8)

    def write_neuron_ids(self, neuron_ids):
        """Write a volume of segmented neurons.
        """

        self.write_volume(neuron_ids, "/volumes/labels/neuron_ids", np.uint64)

    def write_clefts(self, clefts):
        """Write a volume of segmented synaptic clefts.
        """

        self.write_volume(clefts, "/volumes/labels/clefts", np.uint64)

    def write_annotations(self, annotations):
        """Write pre- and post-synaptic site annotations.
        """

        if len(annotations.ids()) == 0:
            return

        self._create_group("/annotations")
        self.file["/annotations"].attrs["offset"] = self._spatial(annotations.offset)

        self._create_dataset("/annotations/ids", data=list(annotations.ids()), dtype=np.uint64)
        self._create_dataset("/annotations/types",
                             data=list(annotations.types()),
                             dtype=unicode_dtype,
                             compression="gzip")
        self._create_dataset(
            "/annotations/locations",
            data=[self._spatial(loc) for loc in annotations.locations()],
            dtype=np.double
        )

        if len(annotations.comments) > 0:
            keys = []
            values = []
            for id_ in annotations.ids():
                if id_ in annotations.comments:
                    keys.append(id_)
                    values.append(annotations.comments[id_])
            self._create_dataset("/annotations/comments/target_ids", data=keys, dtype=np.uint64)
            self._create_dataset("/annotations/comments/comments",
                                 data=values,
                                 dtype=unicode_dtype)

        if len(annotations.pre_post_partners) > 0:
            self._create_dataset("/annotations/presynaptic_site/partners",
                                 data=annotations.pre_post_partners,
                                 dtype=np.uint64)

    def has_raw(self):
        """Check if this file contains a raw volume.
        """
        return self._has_volume("/volumes/raw")

    def has_neuron_ids(self):
        """Check if this file contains neuron ids.
        """
        return self._has_volume("/volumes/labels/neuron_ids")

    def has_neuron_ids_confidence(self):
        """Check if this file contains confidence information about neuron ids.
        """
        return self._has_volume("/volumes/labels/neuron_ids_confidence")

    def has_clefts(self):
        """Check if this file contains synaptic clefts.
        """
        return self._has_volume("/volumes/labels/clefts")

    def has_annotations(self):
        """Check if this file contains synaptic partner annotations.
        """
        return "/annotations" in self.file

    def has_segment_annotations(self):
        """Check if this file contains segment annotations.
        """
        return "/annotations" in self.file

    def read_raw(self):
        """Read the raw volume.
        Returns a Volume.
        """

        return self.read_volume("/volumes/raw")

    def read_neuron_ids(self):
        """Read the volume of segmented neurons.
        Returns a Volume.
        """

        return self.read_volume("/volumes/labels/neuron_ids")

    def read_neuron_ids_confidence(self):
        """Read confidence information about neuron ids.
        Returns Confidences.
        """

        # FIXME Confidences is undefined
        confidences = Confidences(num_levels=2)
        if not self.has_neuron_ids_confidence():
            return confidences

        data = self.file["/volumes/labels/neuron_ids_confidence"]
        i = 0
        while i < len(data):
            level = data[i]
            i += 1
            num_ids = data[i]
            i += 1
            confidences.add_all(level, data[i:i + num_ids])
            i += num_ids

        return confidences

    def read_clefts(self):
        """Read the volume of segmented synaptic clefts.
        Returns a Volume.
        """

        return self.read_volume("/volumes/labels/clefts")

    def read_annotations(self):
        """Read pre- and post-synaptic site annotations.
        """

        annotations = Annotations()

        if "/annotations" not in self.file:
            return annotations

        offset = (0.0, 0.0, 0.0)
        if "offset" in self.file["/annotations"].attrs:
            offset = self._spatial(self.file["/annotations"].attrs["offset"])
        annotations.offset = offset

        ids = self.file["/annotations/ids"]
        types = self.file["/annotations/types"]
        locations = self.file["/annotations/locations"]
        for idx, ann_type, location in zip(ids, types, locations):
            annotations.add_annotation(idx, ann_type.decode("utf-8"), self._spatial(location))

        if "comments" in self.file["/annotations"]:
            ids = self.file["/annotations/comments/target_ids"]
            comments = self.file["/annotations/comments/comments"]
            for (id, comment) in zip(ids, comments):
                annotations.add_comment(id, comment.decode("utf-8"))

        if "presynaptic_site/partners" in self.file["/annotations"]:
            pre_post = self.file["/annotations/presynaptic_site/partners"]
            for (pre, post) in pre_post:
                annotations.set_pre_post_partners(pre, post)

        return annotations

    def close(self):

        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CremiFile(AbstractCremiFile):
    def __init__(self, filename, mode="a", *args, **kwargs):
        self.file = h5py.File(filename, mode)
        if mode == "w" or mode == "a":
            self.file["/"].attrs["file_format"] = "0.2"

    @property
    def h5file(self):
        return self.file


class CremiN5(AbstractCremiFile):
    def __init__(self, filename, chunks=None, *args, **kwargs):
        if not z5py:
            raise RuntimeError("N5 files not supported; install z5py")
        self.file = z5py.N5File(filename)
        self.file.attrs["file_format"] = "0.2"
        self.chunks = chunks

    @property
    def n5file(self):
        return self.file

    def _spatial(self, sequence):
        return sequence[::-1]

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _create_dataset(self, path, data, dtype, compression=None):
        """Wrapper around h5py's create_dataset. Creates the group, if not
        existing. Deletes a previous dataset, if existing and not compatible.
        Otherwise, replaces the dataset.
        """
        data = np.asarray(data)
        self._replace_compatible_dataset(path, data, dtype)
        chunks = [max(c, s) for c, s in zip(self.chunks, data.shape)]

        self.file.create_dataset(path, data=data, dtype=dtype, compression=compression, chunks=chunks)
