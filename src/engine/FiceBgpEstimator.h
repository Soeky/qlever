// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_ENGINE_FICEBGPESTIMATOR_H
#define QLEVER_SRC_ENGINE_FICEBGPESTIMATOR_H

#ifdef QLEVER_WITH_FICE

#include <torch/script.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "engine/IBgpCardinalityEstimator.h"

namespace qlever::bgp {

// Memory-mapped file helper (RAII).
struct MmapFile {
  void* data = nullptr;
  size_t size = 0;
  int fd = -1;

  ~MmapFile();
  MmapFile() = default;
  MmapFile(MmapFile&& o) noexcept;
  MmapFile& operator=(MmapFile&& o) noexcept;
  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;
};

MmapFile mmapOpen(const std::string& path);

// FICE cardinality estimator — uses a pre-trained GNN model to predict
// the cardinality of a BGP (set of triple patterns).
//
// Lifecycle: construct once at server startup with the path to the
// inference artifacts directory (output of prepare.py). Each call to
// estimate() builds a query graph from the triple patterns and runs
// the TorchScript model (~0.5ms per query).
class FiceBgpEstimator : public IBgpCardinalityEstimator {
 public:
  // `artifactsDir` must contain: metadata.json, uri_to_id.json,
  // query_gnn_scripted.pt (or query_gnn_traced.pt), entity_embeddings.bin,
  // relation_embeddings.bin, entity_occurrences.bin, relation_occurrences.bin.
  explicit FiceBgpEstimator(const std::string& artifactsDir);

  uint64_t estimate(std::span<const SparqlTriple> triples,
                    const EstimatorContext& ctx) override;

  void warmup(const EstimatorContext& ctx) override;

 private:
  torch::jit::script::Module model_;
  int embeddingDim_;
  int featDim_;
  int graphNumEdges_;

  std::unordered_map<std::string, int> entityUriToId_;
  std::unordered_map<std::string, int> relationUriToId_;

  MmapFile entityEmbMmap_;
  MmapFile relationEmbMmap_;
  MmapFile entityOccMmap_;
  MmapFile relationOccMmap_;

  const float* entityEmb_;
  const float* relationEmb_;
  const float* entityOcc_;
  const float* relationOcc_;

  // Extract a bare URI string from a TripleComponent (strip angle brackets).
  static std::string extractUri(const TripleComponent& tc);
};

}  // namespace qlever::bgp

#endif  // QLEVER_WITH_FICE

#endif  // QLEVER_SRC_ENGINE_FICEBGPESTIMATOR_H
