from copy import deepcopy


class Annotations:

    def __init__(self, offset = (0.0, 0.0, 0.0)):
        self.__ids = []
        self.__types = {}
        self.__locations = {}
        self.comments = {}
        self.pre_post_partners = []
        self.offset = offset

    def __check(self, id):
        if not id in self.__types.keys():
            raise "there is no annotation with id " + str(id)

    def add_annotation(self, id, type, location):
        """Add a new annotation.

        Parameters
        ----------

            id: int
                The ID of the new annotation.

            type: string
                A string denoting the type of the annotation. Use 
                "presynaptic_site" or "postsynaptic_site" for pre- and 
                post-synaptic annotations, respectively.

            location: tuple, float
                The location of the annotation, relative to the offset.
        """
        self.__ids.append(id)
        self.__types[id] = type.encode('utf8')
        self.__locations[id] = location

    def add_comment(self, id, comment):
        """Add a comment to an annotation.
        """

        self.__check(id)
        self.comments[id] = comment.encode('utf8')

    def set_pre_post_partners(self, pre_id, post_id):
        """Mark two annotations as pre- and post-synaptic partners.
        """

        self.__check(pre_id)
        self.__check(post_id)
        self.pre_post_partners.append((pre_id, post_id))

    def ids(self):
        """Get the ids of all annotations.
        """

        return list(self.__ids)

    def types(self):
        """Get the types of all annotations.
        """

        return [self.__types[idx] for idx in self.__ids]

    def locations(self):
        """Get the locations of all annotations. Locations are in world units, 
        relative to the offset.
        """

        return [self.__locations[idx] for idx in self.__ids]

    def get_annotation(self, id):
        """Get the type and location of an annotation by its id.
        """

        self.__check(id)
        return (self.__types[id], self.__locations[id])

    def sort(self, key_fn=None, reverse=False, in_place=False):
        """Sorts the annotations.
        key_fn must take 1 positional argument (id) and **kwargs.
        It may make use of "type", "location", and "comment" kwargs.
        By default, sorts annotations by ID"""
        if not in_place:
            clone = deepcopy(self)
            clone.sort(key_fn=key_fn, reverse=reverse, in_place=True)
            return clone

        if key_fn is None:
            def key_fn(id, **kwargs):
                return id

        def full_key(id_):
            return key_fn(
                id_, type=self.__types[id_], location=self.__locations[id_], comment=self.comments.get(id_, '')
            )

        self.__ids.sort(key=full_key, reverse=reverse)
