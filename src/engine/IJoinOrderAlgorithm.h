// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_ENGINE_IJOINORDERALGORITHM_H
#define QLEVER_SRC_ENGINE_IJOINORDERALGORITHM_H

#include <cstddef>
#include <memory>
#include <span>
#include <variant>

#include "engine/IBgpCardinalityEstimator.h"

class QueryExecutionContext;

namespace qlever::bgp {

// A lightweight, algorithm-agnostic intermediate representation of a join
// order as a binary tree. Leaves reference triples by index into the
// original BGP triple list. Internal nodes represent joins.
//
// This IR intentionally carries NO physical operator choices (hash join
// vs merge join, permutation selection, etc.). Those decisions are made
// by the Phase 3 adapter that translates JoinPlan -> SubtreePlan.
struct JoinPlan {
  struct Leaf {
    size_t tripleIndex;  // Index into the original span<SparqlTriple>
  };
  struct JoinNode {
    std::unique_ptr<JoinPlan> left;
    std::unique_ptr<JoinPlan> right;
  };

  std::variant<Leaf, JoinNode> node;

  // Convenience factory methods.
  static JoinPlan makeLeaf(size_t idx) { return JoinPlan{Leaf{idx}}; }

  static JoinPlan makeJoin(JoinPlan left, JoinPlan right) {
    return JoinPlan{JoinNode{std::make_unique<JoinPlan>(std::move(left)),
                             std::make_unique<JoinPlan>(std::move(right))}};
  }

  bool isLeaf() const { return std::holds_alternative<Leaf>(node); }
  const Leaf& asLeaf() const { return std::get<Leaf>(node); }
  const JoinNode& asJoin() const { return std::get<JoinNode>(node); }
};

// Context passed to the join order algorithm.
struct PlannerContext {
  QueryExecutionContext* qec_;
};

// Interface for a BGP join ordering algorithm.
// Given a set of triple patterns and a cardinality estimator, produce
// a JoinPlan that specifies the join order as a binary tree.
class IJoinOrderAlgorithm {
 public:
  virtual ~IJoinOrderAlgorithm() = default;

  // Produce a join plan for the given triples.
  // The estimator can be called to get cardinality estimates for
  // arbitrary subsets of the triples.
  virtual JoinPlan plan(std::span<const SparqlTriple> triples,
                        IBgpCardinalityEstimator& estimator,
                        const PlannerContext& ctx) = 0;
};

}  // namespace qlever::bgp

#endif  // QLEVER_SRC_ENGINE_IJOINORDERALGORITHM_H
