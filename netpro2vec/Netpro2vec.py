# @Time    : 10/12/2025
# @File    : Netpro2vec.py

from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
from joblib import wrap_non_picklable_objects
from netpro2vec.ProbDocExtractor import ProbDocExtractor
from netpro2vec.DistributionGenerator import DistributionGenerator, \
    probability_aggregator_cutoff
from netpro2vec import utils
from gensim.utils import simple_preprocess
from gensim import models, corpora
from gensim import __version__ as gensimversion
import numpy as np
import itertools
import igraph as ig
from typing import *
import os
import pickle as pk

"""
Netpro2vec.py
====================================
The core module of Netpro2vec software.
"""


class Netpro2vec:
    """The class implementation of Netpro2vec model for whole-graph embedding.
    from the IEEE TCBB '20 paper "Netpro2vec: a Graph Embedding Framework for Biological Networks". The procedure
    uses probability distribution representations of graphs and a skip-gram model (Doc2Vec).

    Args:
        format (str)             : graph format. Default "graphml" ("graphml" or "edgelist").
        dimensions (int)         : embedding dimension. Default 128.
        prob_type (list of str)  : list of probability types, e.g. ["tm1"], ["ndd"], ["fndd"].
        extractor (list of int)  : extractor mode for each prob_type (1 or 2).
        cut_off (list of float)  : cut-off thresholds for each prob_type.
        agg_by (list of int)     : aggregators for "ndd"/"fndd" (1–10), 0 for "tm" types.
        min_count (int)          : Doc2Vec min_count.
        down_sampling (float)    : Doc2Vec sampling.
        workers (int)            : number of parallel workers.
        epochs (int)             : Doc2Vec epochs.
        learning_rate (float)    : Doc2Vec initial learning rate.
        remove_inf (bool)        : remove inf bin in distributions.
        vertex_attribute (str)   : vertex attribute name (used also by "fndd").
        feature_sigma (float)    : bandwidth for Gaussian similarity in "fndd".
        similarity (str)         : "gaussian" or "cosine" similarity for "fndd".
        seed (int)               : random seed.
        verbose (bool)           : verbose logging.
        encodew (bool)           : encode words as hashes.
        save_probs, load_probs   : save/load probability matrices.
        save_vocab, load_vocab   : save/load vocabulary.
    """

    def __init__(self, format="graphml", dimensions=128, prob_type: List[str] = ["tm1"],
                 extractor=[1], cut_off=[0.01], agg_by=[0],
                 min_count=5, down_sampling=0.0001, workers=4, epochs=10, learning_rate=0.025,
                 remove_inf=False, vertex_attribute=None,
                 feature_sigma=1.0, similarity="gaussian",
                 seed=0, verbose=False, encodew=True,
                 save_probs=False, load_probs=False, save_vocab=False, load_vocab=False):

        # basic shape consistency
        if len({len(i) for i in [prob_type, extractor, cut_off, agg_by]}) != 1:
            raise Exception("Probability type list must be equal-sized wrt extractor, agg_by and cut_off.")

        for i, a in enumerate(agg_by):
            if prob_type[i] in ["ndd", "fndd"] and (int(a) < 1 or int(a) > 10):
                raise Exception("Aggregators values for %s must be in the range [1,10]" % prob_type[i])
            if prob_type[i] not in ["ndd", "fndd"] and int(a) != 0:
                raise Exception("Aggregators values for %s must be 0 (disabled)" % prob_type[i])

        if any(e not in (1, 2) for e in extractor):
            raise Exception("Supported extractor modes are 1 (single cut-off) and 2 (multiple cut-offs)")

        if dimensions <= 0:
            raise Exception("Dimensions must be > 0 (default 128)")

        if format not in ["graphml", "edgelist"]:
            raise Exception("graph format can be graphml or edgelist (default graphml)")

        self.prob_type = prob_type
        self.extractor = extractor
        self.cut_off = cut_off
        self.agg_by = agg_by
        self.dimensions = dimensions
        self.remove_inf = remove_inf
        self.vertex_attribute = vertex_attribute
        self.feature_sigma = feature_sigma
        self.similarity = similarity

        self.vertex_attribute_list = []
        self.embedding = None
        self.probmats = {}
        self.encodew = encodew
        self.saveprobs = save_probs
        self.loadprobs = load_probs
        self.savevocab = save_vocab
        self.loadvocab = load_vocab

        if not os.path.exists('.np2vec'):
            os.makedirs('.np2vec')
        self.probmatfile = os.path.join('.np2vec', 'probmats.pkl')
        self.vocabfile = os.path.join('.np2vec', 'vocab.pkl')

        self.min_count = min_count
        self.down_sampling = down_sampling
        self.workers = workers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.randomseed = seed
        self.document_collections = []
        self.document_collections_list = []
        self.verbose = verbose
        self.tqdm = tqdm if self.verbose else utils.nop
        self.model = None

    # ------------------------------------------------------------------
    # Text corpus access (for TF-IDF / LDA)
    # ------------------------------------------------------------------
    def get_dictionary_corpus(self):
        diction = corpora.Dictionary(
            [simple_preprocess(" ".join(line)) for line in
             self.document_collections_list[-1]])
        corpus = [diction.doc2bow(simple_preprocess(" ".join(line))) for line in
                  self.document_collections_list[-1]]
        return diction, corpus

    # ------------------------------------------------------------------
    # Sentence / document generation for new graphs
    # ------------------------------------------------------------------
    def get_sentences(self, graphs: List[ig.Graph]):
        probmats = self.__generate_probabilities_newsample(graphs)
        if self.vertex_attribute is not None:
            self.get_vertex_attributes(graphs)
        docs = self.__get_document_collections_newsample(probmats, encodew=self.encodew)
        return docs

    def get_documents(self, graphs: List[ig.Graph]):
        """Generate plain-text documents for graphs."""
        self.num_graphs = len(graphs)
        self.__generate_probabilities(graphs)
        if self.vertex_attribute is not None:
            self.get_vertex_attributes(graphs)
        self.__get_document_collections(tag_doc=False, encodew=self.encodew)
        return [" ".join(doc) for doc in self.document_collections_list[-1]]

    # ------------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------------
    def fit(self, graphs: List[ig.Graph]):
        self.format = "graphml"
        self.num_graphs = len(graphs)
        if self.loadvocab:
            try:
                utils.vprint("Loading vocabulary...", end='\n', verbose=self.verbose)
                with open(self.vocabfile, 'rb') as infile:
                    self.document_collections_list = pk.load(infile)
            except Exception as e:
                utils.vprint("Cannot load vocabulary...%s" % e, end='\n', verbose=self.verbose)
                utils.vprint("...Let's generate it from scratch!", end='\n', verbose=self.verbose)
                self.__generate_probabilities(graphs)
                if self.vertex_attribute is not None:
                    self.get_vertex_attributes(graphs)
                self.__get_document_collections(encodew=self.encodew)
        else:
            self.__generate_probabilities(graphs)
            if self.vertex_attribute is not None:
                self.get_vertex_attributes(graphs)
            self.__get_document_collections(encodew=self.encodew)

        if self.savevocab:
            try:
                utils.vprint("Saving vocabulary...", end='\n', verbose=self.verbose)
                with open(self.vocabfile, 'wb') as outfile:
                    pk.dump(self.document_collections_list, outfile)
            except Exception as e:
                raise Exception("Cannot save vocabulary...", e, e.args)

        self.__run_d2v(dimensions=self.dimensions,
                       min_count=self.min_count,
                       down_sampling=self.down_sampling,
                       workers=self.workers,
                       epochs=self.epochs,
                       learning_rate=self.learning_rate)
        return self

    # ------------------------------------------------------------------
    # Infer vector for new graphs (gensim 4 style signature)
    # ------------------------------------------------------------------
    def infer_vector(self, graphs: List[ig.Graph], epochs=None, alpha=None):
        """Infer embeddings for new graphs."""
        probmats = self.__generate_probabilities_newsample(graphs)
        if self.vertex_attribute is not None:
            self.get_vertex_attributes(graphs)
        docs_list = self.__get_document_collections_newsample(probmats, encodew=self.encodew)

        idx = 0
        if len(self.prob_type) > 1:
            idx = len(self.prob_type) - 1

        documents = [d.words for d in docs_list[idx]]
        if epochs is None:
            epochs = self.epochs
        if alpha is None:
            alpha = self.learning_rate

        utils.vprint("Doc2Vec inferring (epochs=%d, alpha=%f) in progress..." % (epochs, alpha),
                     end='', verbose=self.verbose)
        embedding_list = [self.model.infer_vector(doc, epochs=epochs, alpha=alpha)
                          for doc in documents]
        utils.vprint("Done!", verbose=self.verbose)
        return embedding_list

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_embedding(self):
        return np.array(self.embedding)

    def get_memberships(self):
        return np.array(self.document_collections_list[-1])

    # ------------------------------------------------------------------
    # Probability generation
    # ------------------------------------------------------------------
    def __generate_probabilities(self, graphs: List[ig.Graph]):
        if self.loadprobs:
            try:
                utils.vprint("Loading prob mats...", end='\n', verbose=self.verbose)
                with open(self.probmatfile, 'rb') as infile:
                    self.probmats = pk.load(infile)
                return
            except Exception as e:
                utils.vprint("Cannot load saved probs...%s" % e, end='\n', verbose=self.verbose)
                utils.vprint("...Let's generate them from scratch!", end='\n', verbose=self.verbose)

        # (Loadprobs failed or disabled → generate fresh)
        self.probmats = {}

        for prob_type in self.prob_type:
            if prob_type == "fndd":
                self.probmats[prob_type] = DistributionGenerator(
                    prob_type, graphs,
                    verbose=self.verbose,
                    vertex_attribute=self.vertex_attribute,
                    feature_sigma=self.feature_sigma,
                    similarity=self.similarity
                ).get_distributions()
            else:
                self.probmats[prob_type] = DistributionGenerator(
                    prob_type, graphs,
                    verbose=self.verbose
                ).get_distributions()

        if self.saveprobs:
            try:
                utils.vprint("Saving prob mats...", end='\n', verbose=self.verbose)
                with open(self.probmatfile, 'wb') as outfile:
                    pk.dump(self.probmats, outfile)
            except Exception as e:
                raise Exception("Cannot save probs...", e, e.args)

    def __generate_probabilities_newsample(self, graphs: List[ig.Graph]):
        probmats = {}
        for prob_type in self.prob_type:
            if prob_type == "fndd":
                probmats[prob_type] = DistributionGenerator(
                    prob_type, graphs,
                    verbose=self.verbose,
                    vertex_attribute=self.vertex_attribute,
                    feature_sigma=self.feature_sigma,
                    similarity=self.similarity
                ).get_distributions()
            else:
                probmats[prob_type] = DistributionGenerator(
                    prob_type, graphs,
                    verbose=self.verbose
                ).get_distributions()
        return probmats

    # ------------------------------------------------------------------
    # Feature extraction per graph
    # ------------------------------------------------------------------
    @delayed
    @wrap_non_picklable_objects
    def __batch_feature_extractor(self, probability_distrib_matrix, name, word_tag=None, tag=True,
                                  aggregate=0, cut=0, encodew=True, extractor=1):
        """Generate a document collection describing a single graph."""
        if self.vertex_attribute is not None:
            vertex_labels = self.vertex_attribute_list[int(name)]
        else:
            vertex_labels = None

        if aggregate > 0 or cut > 0:
            probability_distrib_matrix = probability_aggregator_cutoff(
                probability_distrib_matrix, cut_off=cut,
                agg_by=aggregate, return_prob=True,
                remove_inf=self.remove_inf)

        document_collections = ProbDocExtractor(
            probability_distrib_matrix,
            name, word_tag,
            extractor=extractor,
            tag=tag, encodew=encodew,
            vertex_labels=vertex_labels
        )
        return document_collections

    def get_vertex_attributes(self, graphs):
        if self.vertex_attribute in graphs[0].vs.attributes():
            self.vertex_attribute_list = [graphs[x].vs[self.vertex_attribute]
                                          for x in range(len(graphs))]
        else:
            raise Exception('The graph does not have the provided vertex attribute specified in -A!')

    # ------------------------------------------------------------------
    # Document creation for training
    # ------------------------------------------------------------------
    def __get_document_collections(self, workers=4, tag_doc=True, encodew=True):
        document_collections_all = []
        for prob_idx, prob_type in enumerate(self.prob_type):
            if (len(self.prob_type) > 1) or (tag_doc is False):
                tag = False
            else:
                tag = True

            prob_mats = self.probmats[prob_type]
            utils.vprint("Building vocabulary for %s..." % prob_type, verbose=self.verbose)

            with parallel_backend('threading', n_jobs=workers):
                document_collections = Parallel()(
                    self.__batch_feature_extractor(
                        p, str(i), prob_type, tag=tag,
                        extractor=self.extractor[prob_idx],
                        encodew=encodew,
                        cut=self.cut_off[prob_idx],
                        aggregate=self.agg_by[prob_idx])
                    for i, p in enumerate(self.tqdm(prob_mats))
                )

            document_collections = [
                document_collections[x].graph_document for x in range(len(document_collections))
            ]
            document_collections_all.append(document_collections)

        if len(self.prob_type) > 1:
            doc_merge = [e for e in zip(*document_collections_all)]
            merged_document = [list(itertools.chain.from_iterable(doc)) for doc in doc_merge]
            document_collections_all.append(merged_document)

            if tag_doc:
                for prob_type_doc in document_collections_all:
                    self.document_collections_list.append([
                        models.doc2vec.TaggedDocument(
                            words=prob_type_doc[x], tags=["g_%d" % x])
                        for x in range(len(prob_type_doc))
                    ])
            else:
                self.document_collections_list = document_collections_all
        else:
            # single prob_type; when tag_doc=True we already have TaggedDocuments
            self.document_collections_list.append(document_collections_all[0])

    # ------------------------------------------------------------------
    # Document creation for new samples
    # ------------------------------------------------------------------
    def __get_document_collections_newsample(self, probmats, workers=4, tag_doc=True, encodew=True):
        document_collections_list = []
        document_collections_all = []

        for prob_idx, prob_type in enumerate(self.prob_type):
            if (len(self.prob_type) > 1) or (tag_doc is False):
                tag = False
            else:
                tag = True

            prob_mats = probmats[prob_type]
            utils.vprint("Building vocabulary for %s..." % prob_type, verbose=self.verbose)

            with parallel_backend('threading', n_jobs=workers):
                document_collections = Parallel()(
                    self.__batch_feature_extractor(
                        p, str(i), prob_type, tag=tag,
                        extractor=self.extractor[prob_idx],
                        encodew=encodew,
                        cut=self.cut_off[prob_idx],
                        aggregate=self.agg_by[prob_idx])
                    for i, p in enumerate(self.tqdm(prob_mats))
                )

            document_collections = [
                document_collections[x].graph_document for x in range(len(document_collections))
            ]
            document_collections_all.append(document_collections)

        if len(self.prob_type) > 1:
            doc_merge = [e for e in zip(*document_collections_all)]
            merged_document = [list(itertools.chain.from_iterable(doc)) for doc in doc_merge]
            document_collections_all.append(merged_document)

            if tag_doc:
                for prob_type_doc in document_collections_all:
                    document_collections_list.append([
                        models.doc2vec.TaggedDocument(
                            words=prob_type_doc[x], tags=["g_%d" % x])
                        for x in range(len(prob_type_doc))
                    ])
            else:
                document_collections_list = document_collections_all
        else:
            document_collections_list.append(document_collections_all[0])

        return document_collections_list

    # ------------------------------------------------------------------
    # Doc2Vec training (unified for gensim 3 & 4)
    # ------------------------------------------------------------------
    def __run_d2v(self, dimensions=128, min_count=5, down_sampling=0.0001,
                  workers=4, epochs=10, learning_rate=0.025):
        """
        Gensim-safe Doc2Vec training:
          1. Initialize model with NO corpus
          2. build_vocab(docs)
          3. train(docs)
        Works for gensim 3.x and 4.x
        """

        # If multiple prob types, use last one
        idx = len(self.prob_type) - 1 if len(self.prob_type) > 1 else 0
        docs = self.document_collections_list[idx]

        if not docs:
            raise Exception("Document list is empty — cannot train Doc2Vec")

        utils.vprint("Doc2Vec embedding in progress...", verbose=self.verbose)

        # 1) Create empty model
        model = models.doc2vec.Doc2Vec(
            vector_size=dimensions,
            window=0,
            min_count=min_count,
            dm=0,
            sample=down_sampling,
            workers=workers,
            alpha=learning_rate,
            seed=self.randomseed
        )

        # 2) Build vocabulary
        model.build_vocab(docs)

        # 3) Train model
        model.train(
            docs,
            total_examples=len(docs),
            epochs=epochs
        )

        self.model = model

        # 4) Extract embedding matrix
        try:
            # gensim 4
            self.embedding = model.docvecs.vectors
        except AttributeError:
            # gensim 3
            self.embedding = model.docvecs.doctag_syn0

        utils.vprint("Done!", verbose=self.verbose)
