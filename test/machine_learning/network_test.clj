(ns machine-learning.network-test
  (:require [midje.sweet :refer :all]
            [machine-learning.network :refer :all]
            [clojure.core.matrix :as m])
  (:import (java.util Random)))

(facts "about 'sample-gaussian'"
       (fact "returns expected numbers with defined random seed"
             (first (sample-gaussian 1 (Random. 1000))) => 1.6925177840650305
             (sample-gaussian 2 (Random. 1000)) => [1.6925177840650305 0.6026210756731758])
       (fact "supports vector input"
             (sample-gaussian [2] (Random. 1000)) => [[1.6925177840650305 0.6026210756731758]]
             (sample-gaussian [2 2] (Random. 1000)) => [[1.6925177840650305 0.6026210756731758] [-0.719106498075259 -2.8712814721590734]]))


(facts "about 'sigmoid'"
       (fact "supports single input"
             (sigmoid 1) => 0.7310585786300049
             (sigmoid 0) => 0.5)
       (fact "support vector input"
             (sigmoid [-1 0 1]) => [0.2689414213699951 0.5 0.7310585786300049]))

(facts "about 'sigmoid-prime"
       (fact "supports single input"
             (sigmoid-prime 1) => truthy)
       (fact "calculates expected results"
             (sigmoid-prime 1) => 0.19661193324148185
             (sigmoid-prime [1 1]) => [0.19661193324148185 0.19661193324148185]
             (sigmoid-prime [1 0]) => [0.19661193324148185 0.25]))

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
       (fact "Biases should be present and have the correct structure"
             (m/shape (:biases (create-network [2 3 4]))) => [2 3 1]
             (m/shape (:biases (create-network [2 3 4 5]))) => [3 3 1])
       (fact "Weights shold be a matrix-array with weights between layers"
             (m/matrix? (first (:weights (create-network [2 3 2])))) => truthy
             (m/row-count (first (:weights (create-network [2 3 2])))) => 3
             (m/row-count (ffirst (:weights (create-network [2 3 2])))) => 2
             (m/row-count (nth (:weights (create-network [2 3 2])) 1)) => 2
             (m/row-count (first (nth (:weights (create-network [2 3 2])) 1))) => 3
             )
       )

(def test-network-1
  {:num-layers 3
   :sizes      [2 3 2]
   :biases     (list (m/matrix [[0] [0] [0]])
                     (m/matrix [[0] [0]]))
   :weights    (list (m/matrix [[0 0] [0 0] [0 0]])
                     (m/matrix [[0 0 0] [0 0 0]]))})

(def test-network-2
  {:num-layers 3
   :sizes      [2 3 2]
   :biases     (list (m/matrix [[0.67959424] [-0.50075735] [-0.1517778]])
                     (m/matrix [[-0.92451521] [0.89157524]]))
   :weights    (list (m/matrix [[-1.62339141 -0.65800186]
                                [0.96918883 -0.89932594]
                                [1.51664486 0.30323217]])
                     (m/matrix [[0.62191559 -0.40955908 1.42636803]
                                [0.50692284 -0.64028764 -2.13522075]]))})

(facts "about 'feed-forward'"
       (fact "returns expected output structure"
             (count (feed-forward (create-network [2 3 2]) [[1] [1]])) => 2)
       (fact "return expected output for given networks"
             (feed-forward test-network-1 [[0] [0]]) => [[0.5] [0.5]]
             (feed-forward test-network-1 [[1] [1]]) => [[0.5] [0.5]]
             (feed-forward (assoc test-network-1 :biases [[[-1] [1] [-1]] [[0] [0]]]) [[1] [1]]) => [[0.5] [0.5]]
             (feed-forward (assoc test-network-1 :biases [[[-1] [1] [-1]] [[1] [1]]]) [[1] [1]]) => [[0.7310585786300049] [0.7310585786300049]]
             (feed-forward test-network-2 [[1] [1]]) => [[0.5544095665798978] [0.2550182532563072]]))

;(facts "about 'collect-activations'"
;       (fact "returns expected output structure"
;             (let [output (collect-activations (create-network [2 3 2]) [[1] [1]])
;                   activations (first output)
;                   zs (second output)]
;               (count activations) => 3
;               (count (first activations)) => 2
;               (count (second activations)) => 3
;               (count (nth activations 2)) => 2
;               (count zs) => 2
;               (count (first zs)) => 3
;               (count (second zs)) => 2))
;       (fact "returns expected values"
;             (let [output (collect-activations test-network-1 [[1] [1]])
;                   activations (first output)
;                   zs (second output)]
;               (first activations) => [[1] [1]]
;               (second activations) => [[0.5] [0.5] [0.5]]
;               (nth activations 2) => [[0.5] [0.5]]
;               (first zs) => [[0] [0] [0]]
;               (second zs) => [[0.0] [0.0]]))
; )

;(facts "about 'backprop'"
;       (fact "returns epected value for given input"
;             (backprop test-network-2 [[[1] [1]] [[0] [1]]]) => [[(m/matrix [[0.0018749460781700626] [0.00824368314946285] [0.0664244302809047]])
;                                                                  (m/matrix [[0.1369611170454698] [-0.1415345702773751]])]
;                                                                 [(m/matrix [[0.0018749460781700626 0.0018749460781700626]
;                                                                    [0.00824368314946285 0.00824368314946285]
;                                                                    [0.0664244302809047 0.0664244302809047]])
;                                                                  (m/matrix [[0.022972532780817807 0.05395073204148325 0.1152284330469081]
;                                                                    [-0.023739639581331136 -0.05575227364059902 -0.11907618093984336]])]]))

(facts "about 'index-of-max'"
       (fact "returns index of highest output value"
             (index-of-max [[1] [2] [3]]) => 2
             (index-of-max [[-1] [0] [-1]]) = 1
             (index-of-max [[1] [0] [-1]]) = 1))

(facts "about 'sgd"
       (let [network (create-network [2 3 2])
             training_data [[[0] [0]] 0
                            [[0] [1]] 1
                            [[1] [0]] 1
                            [[1] [1]] 0]
             trained_network (sgd network training_data 5 4 100)]
         (fact "trained network solves XOR"
               (run-test network [[0] [0]]) => 0
               (run-test network [[0] [1]]) => 1
               (run-test network [[1] [0]]) => 1
               (run-test network [[1] [1]]) => 0
               )))


