// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_ENGINE_IBGPCARDINALITYESTIMATOR_H
#define QLEVER_SRC_ENGINE_IBGPCARDINALITYESTIMATOR_H

#include <cstdint>
#include <span>

#include "parser/SparqlTriple.h"

class QueryExecutionContext;

namespace qlever::bgp {

// Minimal context passed to the estimator. Provides access to the index,
// vocabulary, cost factors, and located triples state via the QEC.
struct EstimatorContext {
  QueryExecutionContext* qec_;
};

// Interface for cardinality estimation of arbitrary subsets of a BGP.
// Given a set of triple patterns (possibly a single triple, possibly
// multiple triples representing a sub-join), return an estimated
// result cardinality.
class IBgpCardinalityEstimator {
 public:
  virtual ~IBgpCardinalityEstimator() = default;

  // Estimate the cardinality of the result of joining the given triples.
  // For a single triple, this is the scan cardinality.
  // For multiple triples, this is the estimated join result size.
  virtual uint64_t estimate(std::span<const SparqlTriple> triples,
                            const EstimatorContext& ctx) = 0;

  // Optional: pre-load model weights, embeddings, etc.
  virtual void warmup(const EstimatorContext& /*ctx*/) {}

  // Optional: clear any internal caches.
  virtual void clearCache() {}
};

}  // namespace qlever::bgp

#endif  // QLEVER_SRC_ENGINE_IBGPCARDINALITYESTIMATOR_H
