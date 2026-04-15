// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_ENGINE_DEFAULTBGPCARDINALITYESTIMATOR_H
#define QLEVER_SRC_ENGINE_DEFAULTBGPCARDINALITYESTIMATOR_H

#include "engine/IBgpCardinalityEstimator.h"

namespace qlever::bgp {

// Wraps QLever's existing IndexScan-based estimation for single triples.
// For multi-triple subsets, throws (not needed until Phase 4).
class DefaultBgpCardinalityEstimator : public IBgpCardinalityEstimator {
 public:
  uint64_t estimate(std::span<const SparqlTriple> triples,
                    const EstimatorContext& ctx) override;
};

}  // namespace qlever::bgp

#endif  // QLEVER_SRC_ENGINE_DEFAULTBGPCARDINALITYESTIMATOR_H
