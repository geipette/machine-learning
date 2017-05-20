(ns machine-learning.network-test
  (:require [midje.sweet :refer :all]
            [machine-learning.network :refer :all]
            [clojure.core.matrix :as m])
  (:import (java.util Random)))

(facts "about 'shape'"
       (fact "nil input returns []"
             (shape nil) => [])
       (fact "empty vector returns a empty vector"
             (shape []) => [])
       (fact "vector with unnested elements results in a IllegalArgumentException"
             (shape [1 2 3]) => (throws IllegalArgumentException)
             (shape [[1 2] 3]) => (throws IllegalArgumentException))
       (fact "returns expected output for given input"
             (shape [[1 2] [1 2 3]]) => [2 3]
             (shape [[1 2] [1 2 3] [] [1]]) => [2 3 0 1]))

(facts "about 'sample-gaussian'"
       (fact "returns expected numbers with defined random seed"
             (first (sample-gaussian 1 (Random. 1000))) => 1.6925177840650305
             (sample-gaussian 2 (Random. 1000)) => [1.6925177840650305 0.6026210756731758])
       (fact "supports vector input"
             (sample-gaussian [2] (Random. 1000)) => [[1.6925177840650305 0.6026210756731758]]
             (sample-gaussian [2 2] (Random. 1000)) => [[1.6925177840650305 0.6026210756731758] [-0.719106498075259 -2.8712814721590734]]))


(facts "about 'sigmoid'"
       (fact "supports single input"
             (sigmoid 1) => 2.718281828459045
             (sigmoid 0) => 1.0)
       (fact "support vector input"
             (sigmoid [-1 0 1]) => [0.36787944117144233 1.0 2.718281828459045]))

(facts "about 'create-network'"
       (fact "Fails when sizes contains one or less elements"
             (create-network []) => (throws IllegalArgumentException)
             (create-network nil) => (throws IllegalArgumentException)
             (create-network [1]) => (throws IllegalArgumentException))
       (fact "Number of layers is present and correct"
             (:num-layers (create-network [1 2 3])) => 3
             (:num-layers (create-network [800 40 40 10])) => 4)
       (fact "Sizes should be present and correct"
             (:sizes (create-network [1 2 3])) => [1 2 3]
             (:sizes (create-network [30 40 2 2])) => [30 40 2 2])
       (fact "Biases should be present and have the correct structure, (one per node excluding first layer)"
             (shape (:biases (create-network [2 3 4]))) => [3 4]
             (shape (:biases (create-network [2 3 4 5]))) => [3 4 5])
       (fact "Weights shold be a vector with weights between layers"
             (vector? (:weights (create-network [1 2 1]))) => truthy
             (count (first (:weights (create-network [2 3 2])))) => 2
             (count (ffirst (:weights (create-network [2 3 2])))) => 3
             (count (nth (:weights (create-network [2 3 2])) 1)) => 3
             (count (first (nth (:weights (create-network [2 3 2])) 1))) => 2))

(def test-network
  {:num-layers 3
   :sizes      [2 3 2]
   :biases     [[0 0 0] [0 0]]
   :weights    [[[0 0 0] [0 0 0]]
                [[0 0] [0 0] [0 0]]]})

(facts "about 'feed-forward'"
       (fact "returns expected output structure"
             (count (feed-forward (create-network [2 3 2]) [1 1])) => 2)
       (fact "return expected output for given networks"
             (feed-forward test-network [0 0]) => [1.0 1.0]
             (feed-forward test-network [1 1]) => [1.0 1.0]
             (feed-forward (assoc test-network :biases [[-1 1 -1] [0 0]]) [1 1]) => [1.0 1.0]
             (feed-forward (assoc test-network :biases [[-1 1 -1] [1 1]]) [1 1]) => [2.718281828459045 2.718281828459045]
             ))

